"""Parse an uploaded CSV, TXT, or ZIP and report only what the file contains."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pandas as pd


def result(ok: bool, message: str, rows=None, columns=None, kind="unknown") -> dict:
    return {
        "ok": bool(ok),
        "status": "parsed" if ok else "rejected",
        "message": message[:1000],
        "rows": None if rows is None else int(rows),
        "columns": None if columns is None else int(columns),
        "kind": kind,
    }


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def describe_table(frame: pd.DataFrame, label: str) -> dict:
    rows, columns = frame.shape
    names = [str(name) for name in frame.columns[:12]]
    preview = ", ".join(names)
    if len(frame.columns) > 12:
        preview += ", …"
    return result(
        True,
        f"{label}: {rows} data rows, {columns} columns ({preview}). No model scores were added.",
        rows,
        columns,
        "table",
    )


def describe_text(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    return result(
        True,
        f"Plain text file with {len(lines)} non-empty lines. It is not a feature table, so no scores were produced.",
        len(lines),
        None,
        "text",
    )


def validate_path(path: Path) -> dict:
    if not path.is_file():
        return result(False, "Uploaded file was not found on disk.")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            return describe_table(read_table(path), path.name)
        except Exception as exc:
            return result(False, f"CSV could not be read: {exc}")
    if suffix == ".txt":
        try:
            frame = read_table(path)
            if frame.shape[1] > 1:
                return describe_table(frame, path.name)
        except Exception:
            pass
        return describe_text(path)
    if suffix == ".zip":
        try:
            with zipfile.ZipFile(path) as archive:
                names = [name for name in archive.namelist() if not name.endswith("/")]
                tables = [name for name in names if name.lower().endswith((".csv", ".txt"))]
                if not tables:
                    return result(
                        False,
                        f"ZIP contains {len(names)} files and no CSV or TXT member.",
                    )
                member = tables[0]
                extracted = path.with_suffix("") 
                target = extracted.parent / f"{path.stem}__{Path(member).name}"
                target.write_bytes(archive.read(member))
                try:
                    if member.lower().endswith(".csv"):
                        report = describe_table(read_table(target), member)
                    else:
                        report = validate_path(target)
                finally:
                    target.unlink(missing_ok=True)
                report["message"] = (
                    f"ZIP has {len(names)} files. Read {member}. {report['message']}"
                )[:1000]
                return report
        except zipfile.BadZipFile:
            return result(False, "File is not a valid ZIP archive.")
        except Exception as exc:
            return result(False, f"ZIP could not be read: {exc}")
    return result(False, "Only CSV, ZIP, and TXT files are accepted.")


def main() -> None:
    if len(sys.argv) != 2:
        print(json.dumps(result(False, "No file path was provided.")))
        return
    print(json.dumps(validate_path(Path(sys.argv[1]))))


if __name__ == "__main__":
    main()
