"""
Orchestrate the IMS feature pipeline.

Default: steps 2-6 (refine through targets). Extraction from raw
waveforms is long-running and opt-in via --with-extract.

    python run_pipeline.py
    python run_pipeline.py --with-extract
    python run_pipeline.py --from-step refine --to-step targets
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

STEPS: list[tuple[str, str, str]] = [
    ("extract", "extract_features.py", "Raw IMS snapshots -> features.csv"),
    ("refine", "refine_features.py", "Drop redundant columns"),
    ("temporal", "add_temporal_features.py", "Deltas / rolling / gap flags"),
    ("select_temporal", "select_temporal_features.py", "Keep selected temporal set"),
    ("select_model", "select_model_features.py", "Train-only low-var drop"),
    ("targets", "create_targets.py", "Labels + train/val/test CSVs"),
]

TRAIN_STEPS: list[tuple[str, str, str]] = [
    ("train_if", "train_isolation_forest.py", "Isolation Forest"),
    ("train_clf", "train_failure_classifier.py", "Failure classifier"),
    ("train_rul", "train_rul_regressor.py", "RUL regressor"),
]


def run_script(script: str) -> None:
    path = PROJECT_ROOT / script
    if not path.exists():
        raise FileNotFoundError(path)
    print()
    print("=" * 70)
    print(f"RUNNING {script}")
    print("=" * 70)
    completed = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"{script} failed with exit code {completed.returncode}"
        )


def main() -> int:
    names = [name for name, _, _ in STEPS]
    parser = argparse.ArgumentParser(description="Run the IMS feature pipeline.")
    parser.add_argument(
        "--with-extract",
        action="store_true",
        help="Include extract_features.py (slow; reads every raw snapshot).",
    )
    parser.add_argument("--from-step", choices=names, default="refine")
    parser.add_argument("--to-step", choices=names, default="targets")
    parser.add_argument(
        "--train",
        action="store_true",
        help="After features/targets, retrain IF + classifier + RUL.",
    )
    args = parser.parse_args()

    start = names.index(args.from_step)
    end = names.index(args.to_step)
    if end < start:
        print("--to-step must be at or after --from-step")
        return 1

    selected = STEPS[start : end + 1]
    if not args.with_extract:
        selected = [s for s in selected if s[0] != "extract"]
        if args.from_step == "extract":
            print("Skipping extract (pass --with-extract to run it).")

    print("Pipeline steps:")
    for name, script, desc in selected:
        print(f"  - {name:16} {script}  ({desc})")

    for _name, script, _desc in selected:
        run_script(script)

    if args.train:
        for _name, script, _desc in TRAIN_STEPS:
            run_script(script)

    print()
    print("Pipeline finished.")
    print("Next: python smoke_test_ml.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
