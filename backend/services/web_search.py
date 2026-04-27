"""Web search — DuckDuckGo instant API with Bing HTML fallback."""

from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request


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


def search_web(query: str, limit: int = 5) -> dict:
    try:
        results = _ddg_search(query, limit)
        if results:
            return {"ok": True, "query": query, "results": results}
    except Exception:
        pass
    try:
        results = _bing_search(query, limit)
        if results:
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

    # Broad overview search
    overview = search_web(f"{target}产业链 {suffix}", limit=6)

    # Per-section targeted searches (limit to key sections to save time)
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
        "results": all_results[:12],
        "by_section": by_section,
    }


def format_results_as_context(results: list[dict], max_items: int = 5) -> str:
    if not results:
        return ""
    lines = []
    for r in results[:max_items]:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        url = r.get("url", "")
        if snippet:
            lines.append(f"【{title}】{snippet}（来源：{url}）")
    return "\n".join(lines)
