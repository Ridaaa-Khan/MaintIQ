"""HTTP wrapper around validate_dataset.py. Started by the Express app when FastAPI is installed."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile

from validate_dataset import validate_path

app = FastAPI(title="MaintIQ upload check")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/validate")
async def validate(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.bin").suffix.lower()
    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(data)
        temp_path = Path(handle.name)
    try:
        return validate_path(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)
