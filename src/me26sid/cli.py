from __future__ import annotations

import argparse
import sys
from pathlib import Path

from me26sid.calibrate import calibrate_main
from me26sid.eval import eval_main
from me26sid.predict import export_submission_main
from me26sid.train import inspect_data_main, train_main

SCRIPT_TO_COMMAND = {
    "me26sid-train": "train",
    "me26sid-eval": "eval",
    "me26sid-calibrate": "calibrate",
    "me26sid-export-submission": "export-submission",
    "me26sid-inspect-data": "inspect-data",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="me26sid")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("inspect-data", "train", "eval", "calibrate", "export-submission"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--run-name", type=str)
        if command == "eval":
            subparser.add_argument("--checkpoint", type=Path)
        elif command == "calibrate":
            subparser.add_argument("--predictions", type=Path)
        elif command == "export-submission":
            subparser.add_argument("--checkpoint", type=Path)
            subparser.add_argument("--threshold", type=Path)
            subparser.add_argument("--output", type=Path)

    return parser


def main() -> None:
    script_name = Path(sys.argv[0]).name
    argv = sys.argv[1:]
    if script_name in SCRIPT_TO_COMMAND:
        argv = [SCRIPT_TO_COMMAND[script_name], *argv]

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect-data":
        inspect_data_main(args.config, run_name_override=args.run_name)
    elif args.command == "train":
        train_main(args.config, run_name_override=args.run_name)
    elif args.command == "eval":
        eval_main(
            args.config,
            checkpoint_override=args.checkpoint,
            run_name_override=args.run_name,
        )
    elif args.command == "calibrate":
        calibrate_main(
            args.config,
            predictions_path=args.predictions,
            run_name_override=args.run_name,
        )
    elif args.command == "export-submission":
        export_submission_main(
            args.config,
            checkpoint_override=args.checkpoint,
            threshold_override=args.threshold,
            output_override=args.output,
            run_name_override=args.run_name,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported command: {args.command}")
