"""Lightweight output selection; importing this module never loads dynamics."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


class OutputSelection:
    def __init__(self, root: Path) -> None:
        self.path = root / "runtime" / "osc_output_mode.json"
        self.mode = "cpv"
        self.error = None
        if self.path.exists():
            try:
                self.mode = self.validate(json.loads(self.path.read_text(encoding="utf-8-sig"))["mode"])
            except (ValueError, TypeError, KeyError, OSError) as exc:
                self.error = f"output selection invalid; restored CPV: {exc}"

    @staticmethod
    def validate(mode: str) -> str:
        if mode not in {"cpv", "impedance"}:
            raise ValueError("mode must be cpv or impedance")
        return mode

    def save(self, mode: str) -> None:
        mode = self.validate(mode)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="osc-output-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"schema_version": 1, "mode": mode}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.mode, self.error = mode, None
