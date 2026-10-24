from __future__ import annotations

import hashlib
from difflib import SequenceMatcher, ndiff


def diff_flags(reply_claims, answer_claims):
    """The public composer's ndiff rules, including its '?' replacement heuristic."""
    diff = list(ndiff(reply_claims, answer_claims))
    flags, position = [0] * len(reply_claims), 0
    for index, line in enumerate(diff):
        if line.startswith("- "):
            if index + 1 < len(diff) and diff[index+1].startswith("? "):
                position += 1
                continue
            if position < len(flags):
                flags[position] = 1
                position += 1
        elif line.startswith("  "):
            if index > 0 and diff[index-1].startswith("+ "):
                pointer = index-1
                while pointer >= 0 and diff[pointer].startswith("+ "):
                    pointer -= 1
                if pointer >= 0 and diff[pointer].startswith("  "):
                    flags[position] = 1
            position += 1
    if position != len(flags):
        raise ValueError("diff did not cover the response claims")
    return flags


def select_reference(reply, references):
    if not references or any(not isinstance(r, str) or not r for r in references):
        raise ValueError("at least one non-empty reference is required")
    # Selects a textual reference, never a passed model output.
    return min(set(references), key=lambda r: (-SequenceMatcher(None, reply, r, autojunk=False).ratio(), hashlib.sha256(r.encode()).hexdigest()))


def _claims(tokenizer, ids):
    tokens = tokenizer.convert_ids_to_tokens(list(ids))
    specials = set(getattr(tokenizer, "all_special_ids", []))
    separator = "<0x0A>" if "codellama" in tokenizer.name_or_path.lower() else "Ċ"
    groups, newline_previous = [], False
    for index, (token_id, token) in enumerate(zip(ids, tokens)):
        if int(token_id) in specials:
            continue
        newline = separator in token
        if not groups or newline_previous and not newline:
            groups.append([])
        groups[-1].append(index)
        newline_previous = newline
    result, character_offset = [], 0
    reply = tokenizer.decode(list(ids), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    for indices in groups:
        diff_text = tokenizer.decode([ids[i] for i in indices], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        prefix = tokenizer.decode(list(ids[:indices[-1]+1]), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if not reply.startswith(prefix) or len(prefix) < character_offset:
            raise ValueError("claim boundary falls inside an undecodable character")
        text = reply[character_offset:len(prefix)]
        result.append({"start": indices[0], "end": indices[-1], "aligned_token_ids": indices,
                       "claim_text": [tokens[i] for i in indices], "text": text, "diff_text": diff_text,
                       "char_start": character_offset, "char_end": character_offset + len(text),
                       "line_start": reply[:character_offset].count("\n") + 1,
                       "line_end": reply[:character_offset + len(text.rstrip('\n'))].count("\n") + 1})
        character_offset += len(text)
    if "".join(c["text"] for c in result) != reply:
        raise ValueError("token-claim decoding does not reconstruct the exact response text")
    return result


def annotate_response(tokenizer, response_ids, reference_text):
    claims = _claims(tokenizer, response_ids)
    answer = _claims(tokenizer, tokenizer.encode(reference_text, add_special_tokens=False))
    verified = diff_flags([c["diff_text"] for c in claims], [c["diff_text"] for c in answer])
    if verified:
        # As in the original code, the first and last lines are never hallucinated.
        verified[0] = verified[-1] = 0
    labels = [-100] * len(response_ids)
    for claim, flag in zip(claims, verified):
        for index in claim["aligned_token_ids"]:
            labels[index] = flag
    return {"method": "paper_diff_proxy", "ground_truth_status": "proxy_not_semantic_error_truth",
            "reference_sha256": hashlib.sha256(reference_text.encode()).hexdigest(),
            "reference_claim_texts": [c["diff_text"] for c in answer],
            "special_token_positions": [i for i, value in enumerate(labels) if value == -100],
            "claims": claims, "verified": verified, "token_labels": labels}


def validate_labels(labels, record):
    claims, flags, tokens = labels["claims"], labels["verified"], labels["token_labels"]
    if labels["method"] != "paper_diff_proxy" or labels["ground_truth_status"] != "proxy_not_semantic_error_truth":
        raise ValueError("label source must remain an explicit diff proxy")
    if len(tokens) != len(record["generated_ids"]) or len(flags) != len(claims):
        raise ValueError("label length mismatch")
    if record.get("claims", claims) != claims or "".join(c["text"] for c in claims) != record["raw_reply"]:
        raise ValueError("claims do not reconstruct the saved response")
    expected = diff_flags([c["diff_text"] for c in claims], labels["reference_claim_texts"])
    if expected:
        expected[0] = expected[-1] = 0
    if flags != expected or any(type(v) is not int or v not in (-100,0,1) for v in tokens):
        raise ValueError("diff label formula mismatch")
    seen, chars = [], 0
    for claim, flag in zip(claims,flags):
        indices = claim["aligned_token_ids"]
        if not indices or indices != sorted(set(indices)) or min(indices) < 0 or max(indices) >= len(tokens):
            raise ValueError("claim token indices mismatch")
        if claim["start"] != indices[0] or claim["end"] != indices[-1]:
            raise ValueError("claim token boundary mismatch")
        if any(tokens[i] != flag for i in indices) or claim["char_start"] != chars:
            raise ValueError("claim labels/character offsets mismatch")
        chars += len(claim["text"])
        if claim["char_end"] != chars:
            raise ValueError("claim character end mismatch")
        seen.extend(indices)
    if seen != sorted(set(seen)) or sorted(seen+labels["special_token_positions"]) != list(range(len(tokens))):
        raise ValueError("claims/special tokens do not cover exact generated ids")
    if any(tokens[i] != -100 for i in labels["special_token_positions"]):
        raise ValueError("special tokens must be unlabelled")
