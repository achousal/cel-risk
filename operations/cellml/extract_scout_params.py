#!/usr/bin/env python3
"""CLI shim for extracting scout warm-start params (CellML factorial workflow).

The implementation lives in ``ced_ml.utils.optuna_warmstart`` so it can be
reused by other operations or notebooks. This file is just the CLI surface
used by ``submit_factorial.sh``.

Usage:
    python operations/cellml/extract_scout_params.py \\
        --storage-dir /path/to/optuna/ \\
        --output scout_top_params.json \\
        --top-k 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ced_ml.utils.optuna_warmstart import extract_top_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract scout params for warm-starting")
    parser.add_argument(
        "--storage-dir",
        required=True,
        type=Path,
        help="Directory with .optuna.journal files",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of best params per model (default: 5)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    results = extract_top_params(args.storage_dir, args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logging.getLogger(__name__).info("Wrote %d models to %s", len(results), args.output)


if __name__ == "__main__":
    main()
