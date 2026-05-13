"""Shared API key loader.

Priority order:
  1. OS environment variable
  2. .env file  (KEY=VALUE format, standard dotenv)
  3. DASHSCOPE_API_KEY legacy file (label-line + value-line format)
"""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ── .env parser (no external dependency) ────────────────────────────────────

def _parse_dotenv(path: Path) -> dict[str, str]:
    """Parse a standard KEY=VALUE .env file."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {}
    result: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            result[key] = value
    return result


# ── Legacy multi-key file parser (label line + value line) ──────────────────

def _parse_legacy_file(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {}
    result: dict[str, str] = {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    i = 0
    while i < len(lines) - 1:
        label = lines[i].lower().rstrip(":").replace(" ", "_")
        value = lines[i + 1].strip()
        is_key = (
            value.startswith("sk-")
            or value.startswith("tvly-")
            or (len(value) > 20 and " " not in value)
        )
        if is_key:
            result[label] = value
            i += 2
        else:
            i += 1
    return result


# ── label → .env variable name mapping ──────────────────────────────────────

_LABEL_TO_ENV: dict[str, str] = {
    "aliyun_api_key":      "ALIYUN_API_KEY",
    "siliconflow_api_key": "SILICONFLOW_API_KEY",
    "tavily_api_key":      "TAVILY_API_KEY",
}


def load_key(label: str, env_var: str = "") -> str:
    """
    Load an API key by priority:
      1. OS environment variable (env_var, or the canonical name from _LABEL_TO_ENV)
      2. .env file in project root
      3. Legacy DASHSCOPE_API_KEY file
    """
    label = label.lower()

    # 1. OS env var
    candidates = [env_var] if env_var else []
    canonical = _LABEL_TO_ENV.get(label)
    if canonical and canonical not in candidates:
        candidates.append(canonical)
    for var in candidates:
        val = os.getenv(var, "").strip()
        if val:
            return val

    # 2. .env file
    dotenv = _parse_dotenv(_PROJECT_ROOT / ".env")
    for var in ([canonical] if canonical else []) + ([] if not env_var else [env_var]):
        if var and var in dotenv:
            return dotenv[var]

    # 3. Legacy file
    legacy = _parse_legacy_file(_PROJECT_ROOT / "DASHSCOPE_API_KEY")
    return legacy.get(label, "")
