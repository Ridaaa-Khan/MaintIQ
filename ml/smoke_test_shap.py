"""
Smoke test for Feature 4 SHAP explanation artifacts.

Verifies required outputs for failure classifier, RUL regressor, and
Isolation Forest. Optionally regenerates with a small sample when
--regenerate is passed.

Run:
    python smoke_test_shap.py
    python smoke_test_shap.py --regenerate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

TARGETS = {
    "failure_classifier": {
        "dir": PROJECT_ROOT / "models" / "failure_classifier" / "shap_explanations",
        "module": "explain_failure_shap",
    },
    "rul_regressor": {
        "dir": PROJECT_ROOT / "models" / "rul_regressor" / "shap_explanations",
        "module": "explain_rul_shap",
    },
    "isolation_forest": {
        "dir": PROJECT_ROOT / "models" / "isolation_forest" / "shap_explanations",
        "module": "explain_isolation_forest_shap",
    },
}

REQUIRED_FILES = [
    "global_shap_importance.csv",
    "individual_shap_explanation.csv",
    "shap_summary.png",
    "individual_shap_waterfall.png",
    "shap_report.json",
]


def check_target(name: str, output_dir: Path) -> list[str]:
    failures: list[str] = []

    if not output_dir.exists():
        return [f"{name}: missing directory {output_dir}"]

    for filename in REQUIRED_FILES:
        path = output_dir / filename
        if not path.exists():
            failures.append(f"{name}: missing {filename}")
            continue
        if path.stat().st_size <= 0:
            failures.append(f"{name}: empty {filename}")
            print(f"FAIL {name}: empty {filename}")
        else:
            print(f"OK  {name}: {filename}")

    report_path = output_dir / "shap_report.json"
    if report_path.exists():
        with report_path.open(encoding="utf-8") as f:
            report = json.load(f)
        additivity = report.get("additivity", {})
        if not additivity.get("passed", False):
            failures.append(
                f"{name}: additivity check failed in shap_report.json"
            )
            print(f"FAIL {name}: additivity not passed")
        else:
            print(
                f"OK  {name}: additivity "
                f"max_error={additivity.get('max_additivity_error')}"
            )

        top = report.get("top_features") or []
        if not top:
            failures.append(f"{name}: shap_report.json has empty top_features")
        else:
            print(f"OK  {name}: top feature = {top[0].get('feature')}")

    return failures


def regenerate() -> int:
    import explain_failure_shap
    import explain_isolation_forest_shap
    import explain_rul_shap

    common = [
        "--background-size",
        "50",
        "--explanation-size",
        "80",
        "--random-state",
        "42",
    ]

    for label, module in [
        ("failure_classifier", explain_failure_shap),
        ("rul_regressor", explain_rul_shap),
        ("isolation_forest", explain_isolation_forest_shap),
    ]:
        print(f"\n--- regenerating {label} ---")
        code = module.main(common)
        if code != 0:
            print(f"FAIL regenerate {label} exited {code}")
            return code
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test SHAP artifacts.")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Re-run SHAP scripts with a small sample before checking.",
    )
    args = parser.parse_args(argv)

    print("=" * 60)
    print("SHAP EXPLANATION SMOKE TEST")
    print("=" * 60)

    if args.regenerate:
        code = regenerate()
        if code != 0:
            return code

    failures: list[str] = []
    for name, meta in TARGETS.items():
        print(f"\n[{name}]")
        failures.extend(check_target(name, meta["dir"]))

    print()
    if failures:
        print("SMOKE TEST FAILED")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
