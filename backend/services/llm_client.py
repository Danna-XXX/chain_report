"""LLM client — dual LLM routing.

Structured tasks (Planner, Critic, Summarizer): SiliconFlow GLM-5.1 (primary) → DashScope Qwen (fallback)
Long-form generation (Executor): DashScope Qwen (primary) → SiliconFlow GLM-5.1 (fallback)
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .key_loader import load_key


# ─────────────────────────────────────────────
# Config classes
# ─────────────────────────────────────────────

class LLMConfig:
    """DashScope / Qwen — primary for long-form generation."""

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
        self.api_key = load_key("aliyun_api_key", env_var="REPORT_AGENT_API_KEY")


class SiliconFlowConfig:
    """SiliconFlow GLM-5.1 — primary for structured tasks (Planner / Critic / Summarizer)."""

    BASE_URL = "https://api.siliconflow.cn/v1"
    MODEL = "Pro/zai-org/GLM-5.1"

    def __init__(self) -> None:
        self.base_url = self.BASE_URL
        self.model = self.MODEL
        self.temperature = 0.3
        self.max_tokens = 2000
        self.timeout_sec = 60
        self.retries = 1
        self.api_key = load_key("siliconflow_api_key", env_var="SILICONFLOW_API_KEY")


# ─────────────────────────────────────────────
# Low-level HTTP
# ─────────────────────────────────────────────

def _post_json(url: str, headers: dict, payload: dict, timeout_sec: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
        return json.loads(body)


def _call_config(cfg, system_prompt: str, user_prompt: str, max_tokens: int | None) -> dict:
    """Single attempt to call one LLM config. Returns standardized result dict."""
    if not cfg.api_key:
        return {"ok": False, "error": "missing_api_key", "content": "", "usage": {}, "model": cfg.model}

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
        if attempt < cfg.retries:
            time.sleep(0.8 * (attempt + 1))

    return {"ok": False, "error": last_error or "request_failed", "content": "", "usage": {}, "model": cfg.model}


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def chat_completion(system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> dict:
    """Long-form generation: Qwen primary → GLM-5.1 fallback."""
    primary = LLMConfig()
    result = _call_config(primary, system_prompt, user_prompt, max_tokens)
    if result.get("ok") and result.get("content", "").strip():
        return result

    # Fallback to SiliconFlow if primary failed for a non-key reason
    if result.get("error") != "missing_api_key":
        fallback = SiliconFlowConfig()
        if fallback.api_key:
            fb_result = _call_config(fallback, system_prompt, user_prompt, max_tokens)
            if fb_result.get("ok") and fb_result.get("content", "").strip():
                return fb_result

    return result


def chat_completion_structured(system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> dict:
    """Structured tasks (Planner, Critic, Summarizer): GLM-5.1 primary → Qwen fallback."""
    primary = SiliconFlowConfig()
    result = _call_config(primary, system_prompt, user_prompt, max_tokens)
    if result.get("ok") and result.get("content", "").strip():
        return result

    # Fallback to DashScope
    if result.get("error") != "missing_api_key":
        fallback = LLMConfig()
        if fallback.api_key:
            fb_result = _call_config(fallback, system_prompt, user_prompt, max_tokens)
            if fb_result.get("ok") and fb_result.get("content", "").strip():
                return fb_result

    return result


def llm_enabled() -> bool:
    qwen_key = LLMConfig().api_key
    sf_key = SiliconFlowConfig().api_key

    def _valid(key: str) -> bool:
        if not key:
            return False
        if key.startswith("请将") or key.startswith("请在") or "替换" in key or "API Key" in key:
            return False
        return len(key) > 10

    return _valid(qwen_key) or _valid(sf_key)


def active_models() -> dict:
    """Return which models are available, for status endpoint."""
    qwen = LLMConfig()
    sf = SiliconFlowConfig()
    return {
        "generation": qwen.model if qwen.api_key else (sf.model if sf.api_key else None),
        "structured": sf.model if sf.api_key else (qwen.model if qwen.api_key else None),
    }
