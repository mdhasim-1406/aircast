"""Run the separately authorized probabilistic candidate training workflow."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="run candidate training; no promotion is performed")
    args = parser.parse_args()
    if not args.train:
        parser.error("explicit --train required; candidate creation is never a scheduled refresh side effect")
    from scripts.train_probabilistic import main as train_main
    try:
        result = train_main()
    except SystemExit as exc:
        return int(exc.code or 0)
    if isinstance(result, int) and result != 0:
        return result
    summary = result if isinstance(result, dict) else {"result": str(result)}
    print(json.dumps({"candidate_training": "completed", **summary}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
