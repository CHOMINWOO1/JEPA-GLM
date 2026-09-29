from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CsvLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields: list[str] | None = None

    def log(self, row: dict[str, Any]) -> None:
        if self._fields is None:
            self._fields = list(row)
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._fields)
                writer.writeheader()
                writer.writerow(row)
            return
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self._fields).writerow(row)
