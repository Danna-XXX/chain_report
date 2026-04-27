"""LLM client — DashScope OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


class LLMConfig:
    def __init__(self) -> None:
        self.base_url = os.getenv(
            "REPORT_AGENT_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = os.getenv("REPORT_AGENT_MODEL", "qwen-plus")
        self.temperature = float(os.getenv("REPORT_AGENT_TEMPERATURE", "0.5"))
        self.max_tokens = int(os.getenv("REPORT_AGENT_MAX_TOKENS", "3000"))
        self.timeout_sec = int(os.getenv("REPORT_AGENT_TIMEOUT_SEC", "90"))
        self.retries = int(os.getenv("REPORT_AGENT_RETRIES", "2"))
        self.api_key = self._load_api_key()

    @staticmethod
    def _read_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    def _load_api_key(self) -> str:
        env_key = os.getenv("REPORT_AGENT_API_KEY", "").strip()
        if env_key:
            return env_key
        # Look for key file in project root (parent of backend/)
        project_root = Path(__file__).resolve().parent.parent.parent
        for name in ("DASHSCOPE_API_KEY", ".api_key"):
            key = self._read_file(project_root / name)
            if key:
                return key
        return ""


def _post_json(url: str, headers: dict, payload: dict, timeout_sec: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
        return json.loads(body)


def chat_completion(system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> dict:
    cfg = LLMConfig()
    if not cfg.api_key:
        return {"ok": False, "error": "missing_api_key", "content": "", "usage": {}}

    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "max_tokens": max_tokens or cfg.max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    last_error = ""
    for attempt in range(cfg.retries + 1):
        try:
            t0 = time.time()
            result = _post_json(url, headers, payload, cfg.timeout_sec)
            elapsed_ms = int((time.time() - t0) * 1000)
            choices = result.get("choices", [])
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            return {
                "ok": True,
                "content": content or "",
                "usage": result.get("usage", {}),
                "latency_ms": elapsed_ms,
                "model": cfg.model,
            }
        except urllib.error.HTTPError as exc:
            last_error = f"http_{exc.code}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.8 * (attempt + 1))

    return {"ok": False, "error": last_error or "request_failed", "content": "", "usage": {}}


def llm_enabled() -> bool:
    key = LLMConfig().api_key
    # Accept keys that look like real API keys (sk- prefix or long alphanumeric strings)
    if not key:
        return False
    if key.startswith("请将") or key.startswith("请在") or "替换" in key or "API Key" in key:
        return False
    return len(key) > 10
