"""Transformer detector of the original study: head, initialization, batching, loss and Trainer setup.

The response is encoded once per line, with an embedding marking the line's tokens, and their mean is classified.
"""
from __future__ import annotations

from contextlib import nullcontext
import weakref

import torch
from torch import nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments

LINE_CHUNK_SIZE = 8                  # lines encoded at once; affects memory only
OFFLOAD = {"enabled": False, "stats": {}}


class TransformerClaimHead(nn.Module):
    """Non-causal Transformer encoder, then mean pooling over a line's tokens."""

    def __init__(self, input_dim: int, *, n_heads: int = 64, n_layers: int = 2, dropout: float = .05,
                 head_dim: int | None = None):
        super().__init__()
        model_dim = int(head_dim or input_dim)
        if model_dim % n_heads:
            raise ValueError("head dimension must be divisible by attention heads")
        self.projection = nn.Sequential(
            nn.Linear(input_dim, model_dim * 2),
            nn.LayerNorm(model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )
        self.entity_embedding = nn.Embedding(2, model_dim)
        layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )


def build_head(input_dim: int, config: dict) -> nn.Module:
    return TransformerClaimHead(input_dim, n_heads=int(config.get("n_heads", 64)),
                                n_layers=int(config.get("n_layers", 2)), dropout=float(config.get("dropout", .05)),
                                head_dim=config.get("head_dim"))


def initialize(module):
    # As in the original training script; MultiheadAttention keeps its own init (in_proj_weight, not .weight).
    if hasattr(module, 'weight') and module.weight is not None:
        if not (hasattr(module, 'name') and 'positional_encoding' in module.name):
            if module.weight.ndim >= 2:
                nn.init.xavier_uniform_(module.weight)
            else:
                nn.init.uniform_(module.weight)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.zeros_(module.bias)


def saved_tensors_on_cpu(enabled=True, stats=None, *, pin_memory=False):
    """Keep the tensors saved for backward in host memory, losslessly (for large inputs on small GPUs)."""
    if not enabled:
        return nullcontext()
    cache = {}
    if stats is None:
        stats = {}
    for name in ('copies', 'reused', 'copied_bytes', 'pinned_bytes'):
        stats.setdefault(name, 0)

    def pack(tensor):
        if tensor.device.type != 'cuda':
            return (False, tensor)
        shape, stride = tensor.size(), tensor.stride()
        owner = tensor
        while owner._base is not None:
            owner = owner._base
        key = (owner._cdata, tensor.storage_offset(), shape, stride, tensor.dtype, tensor._version)
        previous = cache.get(key)
        if previous is not None and previous[0]() is owner:
            stats['reused'] += 1
            return previous[1]
        # Copy only the unique values of a broadcast tensor; unpack restores its strides.
        unique = tensor[tuple(slice(0, 1) if s == 0 else slice(None) for s in stride)]
        host = torch.empty_like(unique, device='cpu', pin_memory=pin_memory)
        host.copy_(unique)
        packed = (True, host, tensor.device, shape, stride)
        cache[key] = (weakref.ref(owner), packed)
        stats['copies'] += 1
        stats['copied_bytes'] += host.numel() * host.element_size()
        if host.is_pinned():
            stats['pinned_bytes'] += host.numel() * host.element_size()
        return packed

    def unpack(packed):
        if not packed[0]:
            return packed[1]
        _, host, device, shape, stride = packed
        unique_shape = tuple(1 if s == 0 else n for n, s in zip(shape, stride))
        restored = torch.empty_strided(unique_shape, stride, dtype=host.dtype, device=device)
        restored.copy_(host, non_blocking=pin_memory)
        return restored.expand(shape)

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


def collate(records):
    if not records: raise ValueError('empty batch')
    tokens=max(r['features'].shape[0] for r in records)
    claims=max(len(r['labels']) for r in records)
    dim=records[0]['features'].shape[1]
    features=torch.zeros(len(records),tokens,dim)
    masks=torch.zeros(len(records),claims,tokens,dtype=torch.bool)
    labels=torch.zeros(len(records),claims)
    valid=torch.zeros(len(records),claims,dtype=torch.bool)
    token_mask=torch.zeros(len(records),tokens,dtype=torch.bool)
    for i,rec in enumerate(records):
        t,d=rec['features'].shape
        c=len(rec['labels'])
        if not t or not c or d!=dim or rec['claim_masks'].shape!=(c,t):
            raise ValueError('unaligned batch member')
        features[i,-t:]=rec['features']
        masks[i,:c,-t:]=rec['claim_masks']
        labels[i,:c]=rec['labels']
        valid[i,:c]=True
        token_mask[i,-t:]=True
    return dict(features=features,claim_masks=masks,labels=labels,claim_valid=valid,token_mask=token_mask)


class ClaimModel(nn.Module):
    def __init__(self,input_dim,head_config,pos_weight):
        super().__init__()
        self.head=build_head(input_dim,head_config)
        self.head.apply(initialize)
        self.register_buffer('positive_weight',torch.tensor([float(pos_weight)]))
        self.training_forwards=0

    def forward(self,features,claim_masks,labels,claim_valid,token_mask):
        if not torch.isfinite(features).all(): raise FloatingPointError('nonfinite features')
        with saved_tensors_on_cpu(OFFLOAD['enabled'] and self.training and torch.is_grad_enabled(),OFFLOAD['stats']):
            projected=self.head.projection(features.float())
            values=[]
            for i in range(len(features)):
                count=int(claim_valid[i].sum())
                if count<1: raise ValueError('empty claims must be filtered before batching')
                chunks=[]
                for start in range(0,count,LINE_CHUNK_SIZE):
                    entity=claim_masks[i,start:min(start+LINE_CHUNK_SIZE,count)].long()
                    n=len(entity)
                    x=projected[i].unsqueeze(0).repeat(n,1,1)+self.head.entity_embedding(entity)
                    padding=(~token_mask[i]).unsqueeze(0).repeat(n,1)
                    encoded=self.head.encoder(x,src_key_padding_mask=padding)
                    pooled=(encoded*entity.unsqueeze(-1)).sum(1)/entity.sum(1).clamp_min(1).unsqueeze(-1)
                    chunks.append(self.head.classifier(pooled).squeeze(-1))
                    del x,encoded,pooled,entity,padding
                result=torch.cat(chunks)
                values.append(F.pad(result,(0,claim_valid.shape[1]-count),value=-100))
            logits=torch.stack(values)
            loss=F.binary_cross_entropy_with_logits(logits[claim_valid].float(),labels[claim_valid].float(),
                                                    pos_weight=self.positive_weight)
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError('nonfinite detector output/loss')
        if self.training: self.training_forwards+=1
        return {'loss':loss,'logits':logits}


def make_trainer(model,train_dataset,val_dataset,recipe,output_dir,*,use_cuda=True,callbacks=None):
    batch=int(recipe['batch_size'])
    if batch not in (1,4): raise ValueError('unexpected published batch size')
    args=TrainingArguments(output_dir=str(output_dir),num_train_epochs=int(recipe['epochs']),
        per_device_train_batch_size=batch,per_device_eval_batch_size=batch,
        gradient_accumulation_steps=int(recipe['gradient_accumulation_steps']),eval_accumulation_steps=4,
        learning_rate=float(recipe['learning_rate']),weight_decay=float(recipe['weight_decay']),
        max_grad_norm=float(recipe['max_grad_norm']),warmup_ratio=float(recipe['warmup_ratio']),
        lr_scheduler_type='linear',fp16=bool(recipe['fp16']) and use_cuda,fp16_full_eval=False,
        load_best_model_at_end=False,eval_strategy='epoch',logging_strategy='epoch',save_strategy='no',
        report_to=[],gradient_checkpointing=False,dataloader_num_workers=0,dataloader_pin_memory=False,
        remove_unused_columns=False,seed=int(recipe['seed']),use_cpu=not use_cuda,disable_tqdm=True,
        optim='adamw_torch',label_names=['labels'],skip_memory_metrics=True)
    return Trainer(model=model,args=args,train_dataset=train_dataset,eval_dataset=val_dataset,
                   data_collator=collate,callbacks=callbacks or [])
