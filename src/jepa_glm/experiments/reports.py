from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_step_metrics_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path: str | Path, summary: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_markdown_table(path: str | Path, rows: list[dict[str, Any]], title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\nNo rows available.\n", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    lines = [f"# {title}", "", "|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(_format_cell(row.get(column, "")) for column in columns) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paper_report(
    path: str | Path,
    table_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]] | None = None,
    title: str = "JEPA-GLM Reproducibility Report",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "## Methods Snapshot",
        "",
        "This report summarizes the reproducible artifacts emitted by the JEPA-GLM pipeline: configuration manifests, step-wise training metrics, fixed benchmark splits, downstream linear-probe metrics, and zero-shot variant scores.",
        "",
        "## Results Table",
        "",
    ]
    if table_rows:
        columns: list[str] = []
        for row in table_rows:
            for column in row:
                if column not in columns:
                    columns.append(column)
        lines.append("|" + "|".join(columns) + "|")
        lines.append("|" + "|".join(["---"] * len(columns)) + "|")
        for row in table_rows:
            lines.append("|" + "|".join(_format_cell(row.get(column, "")) for column in columns) + "|")
    else:
        lines.append("No result rows available.")
    if audit_rows is not None:
        lines.extend(["", "## Artifact Audit", "", "|status|name|detail|", "|---|---|---|"])
        for row in audit_rows:
            lines.append(f"|{row.get('status', '')}|{row.get('name', '')}|{row.get('detail', '')}|")
    lines.extend(
        [
            "",
            "## Reporting Notes",
            "",
            "- Report AUROC, AUPRC, MCC, and confidence intervals for class-imbalanced tasks.",
            "- Preserve the generated manifest and split files with any submitted result table.",
            "- Interpret effect sizes cautiously when the number of seeds is small.",
            "- Treat synthetic smoke results as software validation, not biological evidence.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
