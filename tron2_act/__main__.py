"""CLI: prepare -> train -> evaluate. None of these commands control hardware."""
import argparse
import json
from pathlib import Path

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Audit HDF5, split whole episodes and compute train-only statistics")
    prepare.add_argument("--config", default="configs/tron2_act.yaml")
    prepare.add_argument("--data-root")
    prepare.add_argument("--foreground-root", help="Reviewed causal SAM3 caches; no fallback to raw RGB")
    prepare.add_argument("--allow-experimental-foreground", action="store_true", default=None, help="Offline pipeline checks only; explicit opt-in for experimental model outputs")
    prepare.add_argument("--deduplicate-identical-timestamps", action="store_true", default=None, help="Read-only removal of verified identical repeated observations; conflicting or reversed times still fail")
    prepare.add_argument("--task")
    prepare.add_argument("--episodes", type=int, nargs="+")
    prepare.add_argument("--output", required=True)
    train = commands.add_parser("train", help="Train ACT with validation using the prior, not future labels")
    train.add_argument("--prepared", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--config", help="Optional model/training overrides; data semantics must match prepared data")
    train.add_argument("--resume", help="Trusted local checkpoint; resumes optimizer and weights")
    for name in ["max-steps", "batch-size", "workers", "eval-every", "checkpoint-every", "hidden-dim",
                 "heads", "encoder-layers", "decoder-layers", "feedforward-dim"]:
        train.add_argument("--" + name, type=int)
    train.add_argument("--device")
    train.add_argument("--no-pretrained", action="store_true", help="Do not download/load ImageNet weights")
    train.add_argument("--eval-batches", type=int, help="Smoke-test only: truncate validation; omit for real training")
    evaluate = commands.add_parser("evaluate", help="Offline prediction errors and hold-current-state baseline")
    evaluate.add_argument("--prepared", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--split", choices=["val", "test"], default="test")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--max-batches", type=int, help="Smoke-test only; omit for full evaluation")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .data import prepare
        c = load_config(args.config, data_root=args.data_root, foreground_root=args.foreground_root, allow_experimental_foreground=args.allow_experimental_foreground, deduplicate_identical_timestamps=args.deduplicate_identical_timestamps, task=args.task, episodes=args.episodes)
        report = prepare(c, args.output)
    elif args.command == "train":
        from .engine import train
        overrides = {}
        if args.config:
            # Load only declared keys; an override file need not repeat the data config.
            import yaml
            overrides = yaml.safe_load(Path(args.config).read_text())
            if not isinstance(overrides, dict):
                parser.error("Training override config must be a mapping")
        for key in ["max_steps", "batch_size", "workers", "eval_every", "checkpoint_every", "hidden_dim",
                    "heads", "encoder_layers", "decoder_layers", "feedforward_dim", "device"]:
            if getattr(args, key) is not None:
                overrides[key] = getattr(args, key)
        if args.no_pretrained:
            overrides["pretrained_backbone"] = False
        if args.eval_batches is not None and args.eval_batches < 1:
            parser.error("--eval-batches must be positive")
        report = train(args.prepared, args.output, overrides, args.resume, args.eval_batches)
    else:
        from .engine import evaluate
        if args.max_batches is not None and args.max_batches < 1:
            parser.error("--max-batches must be positive")
        report = evaluate(args.prepared, args.checkpoint, args.output, args.split,
                          args.device, args.batch_size, args.max_batches)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
