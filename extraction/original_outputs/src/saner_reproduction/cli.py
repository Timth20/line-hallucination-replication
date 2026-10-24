from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="shared outputs root")
    parser.add_argument("--cache-dir", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Token-level features of the original models on the released outputs")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="download, verify, and index released data")
    _add_common_arguments(prepare)

    run = commands.add_parser("run", help="resume feature extraction")
    _add_common_arguments(run)
    run.add_argument("--max-samples", type=int)
    run.add_argument("--sample-start", type=int, default=0)
    run.add_argument("--device", default="cuda:0", help="CUDA device in cuda:N form")

    status = commands.add_parser("status", help="show completion counts")
    _add_common_arguments(status)

    verify = commands.add_parser("verify", help="verify completed sample artifacts")
    _add_common_arguments(verify)
    return parser


def _cache_dir(args: argparse.Namespace, config: Mapping[str, Any]) -> Path:
    configured = config.get("cache_dir")
    if configured is None and isinstance(config.get("runtime"), Mapping):
        configured = config["runtime"].get("cache_dir")
    return Path(args.cache_dir or configured or PROJECT_ROOT / ".cache").expanduser()


def _print_result(value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (Mapping, list, tuple)):
        print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(value)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    cache_dir = _cache_dir(args, config)

    from . import pipeline

    if args.command == "prepare":
        result = pipeline.prepare(config, args.output_dir, cache_dir)
    elif args.command == "run":
        result = pipeline.run(
            config,
            args.output_dir,
            cache_dir,
            max_samples=args.max_samples,
            sample_start=args.sample_start,
            device=args.device,
        )
    elif args.command == "status":
        result = pipeline.status(config, args.output_dir, cache_dir)
    else:
        result = pipeline.verify(config, args.output_dir, cache_dir)
    _print_result(result)
    if args.command == "verify" and result["invalid_samples"]:
        return 1
    return 0
