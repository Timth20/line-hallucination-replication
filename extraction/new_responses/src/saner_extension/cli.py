from __future__ import annotations

import argparse
import json

from .config import load_config
from .pipeline import prepare, run, status


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generation, line labels and token-level features of the new responses")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "status", "verify"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", required=True)
        sub.add_argument("--output-dir", default="../outputs_new")
        sub.add_argument("--cache-dir", default="../cache")
        if command == "run":
            sub.add_argument("--device", default="cuda:0")
            sub.add_argument("--sample-start", type=int, default=0)
            sub.add_argument("--max-samples", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "prepare":
        result = prepare(config,args.output_dir,args.cache_dir)
    elif args.command == "run":
        result = run(config,args.output_dir,args.cache_dir,args.device,args.sample_start,args.max_samples)
    else:
        result = status(config,args.output_dir)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 2 if args.command == "verify" and result["status"] != "complete" else 0
