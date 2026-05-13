"""Web search — Tavily (primary) → DuckDuckGo → Bing HTML fallback, with URL content extraction."""

from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request

from .key_loader import load_key


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _fetch(url: str, timeout_sec: int = 15) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _load_tavily_key() -> str:
    return load_key("tavily_api_key", env_var="TAVILY_API_KEY")


# ─────────────────────────────────────────────
# Search backends
# ─────────────────────────────────────────────

def _tavily_search(query: str, api_key: str, limit: int) -> list[dict]:
    url = "https://api.tavily.com/search"
    payload = json.dumps({
        "api_key": api_key,
        "query": query,
        "max_results": limit,
        "include_answer": True,
        "include_raw_content": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="ignore"))

    results = []
    # Prepend the synthesized answer if present
    answer = (data.get("answer") or "").strip()
    if answer:
        results.append({
            "title": f"{query} — Tavily 综合摘要",
            "url": "",
            "snippet": answer[:400],
            "source": "tavily_answer",
        })
    for item in data.get("results", []):
        snippet = (item.get("content") or item.get("snippet") or "").strip()
        results.append({
            "title": (item.get("title") or "")[:100],
            "url": item.get("url", ""),
            "snippet": snippet[:400],
            "source": "tavily",
        })
        if len(results) >= limit:
            break
    return results[:limit]


def _ddg_search(query: str, limit: int) -> list[dict]:
    encoded = urllib.parse.quote(query)
    url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_redirect=1&no_html=1"
    payload = json.loads(_fetch(url, timeout_sec=12))
    results = []
    abstract = (payload.get("AbstractText") or "").strip()
    if abstract:
        results.append({
            "title": (payload.get("Heading") or query)[:100],
            "url": payload.get("AbstractURL") or "",
            "snippet": abstract[:300],
            "source": "ddg_instant",
        })
    for item in payload.get("RelatedTopics", []):
        if isinstance(item, dict) and item.get("Text"):
            results.append({
                "title": (item.get("Text") or "").split(" - ")[0][:80],
                "url": item.get("FirstURL", ""),
                "snippet": (item.get("Text") or "")[:300],
                "source": "ddg_related",
            })
        if len(results) >= limit:
            break
    return results[:limit]


def _bing_search(query: str, limit: int) -> list[dict]:
    encoded = urllib.parse.quote(query)
    url = f"https://cn.bing.com/search?q={encoded}"
    html_text = _fetch(url, timeout_sec=15)
    results = []
    for m in re.finditer(
        r'<h2[^>]*>\s*<a[^>]*href="(.*?)"[^>]*>(.*?)</a>\s*</h2>', html_text, flags=re.S
    ):
        raw_url, raw_title = m.groups()
        if not raw_url or raw_url.startswith("/"):
            continue
        tail = html_text[m.end(): m.end() + 800]
        sm = re.search(r"<p[^>]*>(.*?)</p>", tail, flags=re.S)
        snippet = re.sub(r"<.*?>", "", sm.group(1)).strip() if sm else ""
        title = re.sub(r"<.*?>", "", raw_title).strip()
        results.append({
            "title": html.unescape(title)[:100],
            "url": html.unescape(raw_url),
            "snippet": html.unescape(snippet)[:300],
            "source": "bing_html",
        })
        if len(results) >= limit:
            break
    return results


# ─────────────────────────────────────────────
# URL content extraction
# ─────────────────────────────────────────────

def _fetch_url_content(url: str, timeout: int = 8) -> str:
    """Extract readable text from a URL's HTML (first ~800 chars of paragraph content)."""
    if not url or url.startswith("https://api."):
        return ""
    try:
        html_text = _fetch(url, timeout_sec=timeout)
        # Strip scripts / styles
        html_text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html_text, flags=re.S | re.I)
        # Extract <p> paragraphs
        paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html_text, flags=re.S | re.I)
        text_parts = []
        total = 0
        for p in paragraphs:
            clean = re.sub(r"<[^>]+>", "", p).strip()
            clean = html.unescape(clean)
            clean = re.sub(r"\s+", " ", clean)
            if len(clean) > 30:
                text_parts.append(clean)
                total += len(clean)
            if total >= 800:
                break
        return " ".join(text_parts)[:800]
    except Exception:
        return ""


def _enrich_with_content(results: list[dict], top_n: int = 3) -> list[dict]:
    """Fetch full content for top_n results and store in full_snippet."""
    enriched = list(results)
    fetched = 0
    for r in enriched:
        if fetched >= top_n:
            break
        if r.get("url") and not r.get("full_snippet"):
            content = _fetch_url_content(r["url"])
            if content:
                r["full_snippet"] = content
                fetched += 1
    return enriched


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def search_web(query: str, limit: int = 5) -> dict:
    """Search with Tavily → DDG → Bing priority chain, then enrich top results."""
    tavily_key = _load_tavily_key()

    # Priority 1: Tavily
    if tavily_key:
        try:
            results = _tavily_search(query, tavily_key, limit)
            if results:
                results = _enrich_with_content(results, top_n=3)
                return {"ok": True, "query": query, "results": results}
        except Exception:
            pass

    # Priority 2: DuckDuckGo
    try:
        results = _ddg_search(query, limit)
        if results:
            results = _enrich_with_content(results, top_n=3)
            return {"ok": True, "query": query, "results": results}
    except Exception:
        pass

    # Priority 3: Bing HTML
    try:
        results = _bing_search(query, limit)
        if results:
            results = _enrich_with_content(results, top_n=3)
            return {"ok": True, "query": query, "results": results}
    except Exception as exc:
        return {"ok": False, "query": query, "results": [], "error": str(exc)}

    return {"ok": False, "query": query, "results": [], "error": "no_results"}


def search_for_report(target: str, section_titles: list[str], report_type: str) -> dict:
    """Run one broad search + per-section searches, return merged evidence."""
    suffix_map = {
        "chain_entirety": "产业链 现状 规模 趋势",
        "trade_data": "产业链 交易 结构 规模",
        "company": "企业 经营 产业链 竞争",
    }
    suffix = suffix_map.get(report_type, "产业链 分析")

    overview = search_web(f"{target}产业链 {suffix}", limit=6)

    important_sections = section_titles[:6]
    by_section: list[dict] = []
    for title in important_sections:
        q = f"{target} {title} 行业分析"
        res = search_web(q, limit=4)
        by_section.append({"section": title, **res})

    all_results = list(overview.get("results", []))
    seen_urls: set[str] = {r["url"] for r in all_results}
    for sec in by_section:
        for r in sec.get("results", []):
            if r["url"] not in seen_urls:
                all_results.append(r)
                seen_urls.add(r["url"])

    return {
        "enabled": True,
        "ok": overview.get("ok") or any(s.get("ok") for s in by_section),
        "results": all_results[:15],
        "by_section": by_section,
    }


def format_results_as_context(results: list[dict], max_items: int = 5) -> str:
    if not results:
        return ""
    lines = []
    for r in results[:max_items]:
        title = r.get("title", "")
        # Prefer full_snippet if available, else snippet
        snippet = r.get("full_snippet") or r.get("snippet", "")
        url = r.get("url", "")
        if snippet:
            source_part = f"（来源：{url}）" if url else ""
            lines.append(f"【{title}】{snippet}{source_part}")
    return "\n".join(lines)
