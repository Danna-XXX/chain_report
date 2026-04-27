"""Parse uploaded structured files and private materials."""

from __future__ import annotations

import csv
import json
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None

TEXT_EXTS = {".txt", ".md", ".json"}
SPREADSHEET_EXTS = {".xlsx", ".xls"}


def _safe_preview(text: str, limit: int = 400) -> str:
    return " ".join(text.split())[:limit]


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".json":
        try:
            return json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)
        except Exception:
            pass
    return path.read_text(encoding="utf-8", errors="ignore")


def _parse_excel(path: Path) -> dict:
    if pd is None:
        return {"name": path.name, "format": "excel", "rows": None, "columns": [], "preview": "需安装 pandas/openpyxl"}
    wb = pd.ExcelFile(path)
    total, columns, previews = 0, set(), []
    for sheet in wb.sheet_names[:3]:
        df = pd.read_excel(wb, sheet_name=sheet)
        total += len(df)
        columns.update(str(c) for c in df.columns[:16])
        if not df.empty:
            previews.append(df.head(2).astype(str).to_dict(orient="records"))
    return {
        "name": path.name,
        "format": "excel",
        "rows": total,
        "columns": sorted(columns),
        "preview": _safe_preview(json.dumps(previews, ensure_ascii=False)),
    }


def _parse_csv(path: Path) -> dict:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0] if rows else []
    samples = rows[1:4] if len(rows) > 1 else []
    return {
        "name": path.name,
        "format": "csv",
        "rows": max(len(rows) - 1, 0),
        "columns": header,
        "preview": _safe_preview(" | ".join(",".join(r) for r in samples)),
    }


def summarize_structured_files(files: list[dict], target: str) -> dict:
    items = []
    for fi in files:
        path = Path(fi["path"])
        ext = path.suffix.lower()
        try:
            if ext in SPREADSHEET_EXTS:
                items.append(_parse_excel(path))
            elif ext == ".csv":
                items.append(_parse_csv(path))
            else:
                items.append({"name": path.name, "format": ext.lstrip("."), "rows": None, "columns": [], "preview": "已接收"})
        except Exception as exc:
            items.append({"name": path.name, "format": ext.lstrip("."), "rows": None, "columns": [], "preview": f"解析失败: {exc}"})

    all_cols = sorted({c for item in items for c in item.get("columns", [])})[:30]
    return {
        "overview": {
            "target": target,
            "file_count": len(items),
            "file_names": [i["name"] for i in items],
            "detected_columns": all_cols,
        },
        "items": items,
    }


def summarize_private_materials(files: list[dict], target: str) -> dict:
    items = []
    for fi in files:
        path = Path(fi["path"])
        ext = path.suffix.lower()
        try:
            if ext in TEXT_EXTS or ext == ".csv":
                text = _read_text(path) if ext != ".csv" else path.read_text(encoding="utf-8", errors="ignore")
                items.append({
                    "name": path.name,
                    "format": ext.lstrip("."),
                    "preview": _safe_preview(text, 500),
                    "detail": text[:5000],
                })
            elif ext in SPREADSHEET_EXTS:
                info = _parse_excel(path)
                items.append({
                    "name": path.name,
                    "format": "excel",
                    "preview": info["preview"],
                    "detail": json.dumps(info, ensure_ascii=False),
                })
            else:
                items.append({"name": path.name, "format": ext.lstrip("."), "preview": "文件已接收", "detail": ""})
        except Exception as exc:
            items.append({"name": path.name, "format": ext.lstrip("."), "preview": f"解析失败: {exc}", "detail": ""})

    return {"overview": {"target": target, "file_count": len(items), "file_names": [i["name"] for i in items]}, "items": items}
