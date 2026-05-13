"""报告生成智能体 V3 — Flask 后端入口"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

try:
    from backend.services.llm_client import LLMConfig, active_models, llm_enabled
    from backend.services.material_parser import summarize_private_materials, summarize_structured_files
    from backend.services.report_catalog import (
        build_catalog_payload,
        get_default_outline,
        get_output_filename,
        normalize_outline,
    )
    from backend.services.report_engine import build_report, build_report_stream
    from backend.services.web_search import search_for_report
except ImportError:
    from services.llm_client import LLMConfig, active_models, llm_enabled
    from services.material_parser import summarize_private_materials, summarize_structured_files
    from services.report_catalog import (
        build_catalog_payload,
        get_default_outline,
        get_output_filename,
        normalize_outline,
    )
    from services.report_engine import build_report, build_report_stream
    from services.web_search import search_for_report

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
TEMP_DIR = PROJECT_DIR / "temp"

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
os.makedirs(TEMP_DIR, exist_ok=True)


def _save_files(files, dest: Path) -> list[dict]:
    saved = []
    dest.mkdir(parents=True, exist_ok=True)
    for idx, fs in enumerate(files, 1):
        if not fs or not fs.filename:
            continue
        origin = Path(fs.filename).name
        suffix = Path(origin).suffix.lower()
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(origin).stem).strip("._-") or f"file_{idx}"
        full_path = dest / f"{stem}{suffix}"
        fs.save(full_path)
        saved.append({"name": origin, "path": str(full_path), "suffix": suffix, "size": full_path.stat().st_size})
    return saved


def _parse_versions(raw: str) -> list[str]:
    return [t.strip() for t in raw.split(",") if ":" in t.strip()]


def _parse_outlines(raw: str, keys: list[str]) -> dict[str, list[str]]:
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        payload = {}
    return {k: normalize_outline(payload.get(k), get_default_outline(k)) for k in keys}


@app.route("/")
def index():
    with open(FRONTEND_DIR / "index.html", "r", encoding="utf-8") as f:
        return Response(f.read(), mimetype="text/html")


@app.route("/api/config")
def config():
    cfg = LLMConfig()
    return jsonify({
        "success": True,
        "catalog": build_catalog_payload(),
        "llm_enabled": llm_enabled(),
        "model": cfg.model,
    })


@app.route("/api/status")
def status():
    models = active_models()
    return jsonify({
        "status": "running",
        "version": "3.0.0",
        "llm": {
            "enabled": llm_enabled(),
            "generation_model": models.get("generation"),
            "structured_model": models.get("structured"),
        },
        "features": [
            "planner_executor_critic",
            "inter_section_context",
            "llm_paragraph_generation",
            "web_search_tavily",
            "url_content_extraction",
            "multi_report_generation",
            "editable_outline",
            "structured_file_upload",
            "private_material_upload",
            "markdown_preview_download",
            "sse_streaming",
        ],
    })


# ─────────────────────────────────────────────
# Original synchronous endpoint (backward compatible)
# ─────────────────────────────────────────────

@app.route("/api/generate", methods=["POST"])
def generate():
    target = request.form.get("target", "").strip()
    type_param = request.form.get("type", "")
    enable_web = request.form.get("enable_web_search", "true").lower() == "true"
    selected_keys = _parse_versions(type_param)
    custom_outlines = _parse_outlines(request.form.get("custom_outlines", ""), selected_keys)

    if not target:
        return jsonify({"success": False, "message": "请填写产业链或公司名称。"})
    if not selected_keys:
        return jsonify({"success": False, "message": "请至少选择一个报告版本。"})

    session_dir = TEMP_DIR / str(uuid.uuid4())
    try:
        structured_files = _save_files(request.files.getlist("files"), session_dir / "structured")
        private_files = _save_files(request.files.getlist("private_files"), session_dir / "private")

        structured_summary = summarize_structured_files(structured_files, target)
        private_summary = summarize_private_materials(private_files, target)

        results = []
        for key in selected_keys:
            report_type, version = key.split(":", 1)
            outline = custom_outlines[key]

            web_data = None
            if enable_web:
                try:
                    web_data = search_for_report(target, outline, report_type)
                except Exception:
                    web_data = None

            report = build_report(
                report_type=report_type,
                version=version,
                target=target,
                outline=outline,
                structured_summary=structured_summary,
                private_summary=private_summary,
                web_search_data=web_data,
            )

            results.append({
                "success": True,
                "report_type": report_type,
                "version": version,
                "filename": get_output_filename(report_type, version, target),
                "content": report["markdown"],
                "outline": outline,
                "llm_used": report["llm_used"],
                "elapsed_sec": report["elapsed_sec"],
                "sections_meta": report["sections_meta"],
                "web_search_ok": bool(web_data and web_data.get("ok")),
            })

        return jsonify({
            "success": True,
            "count": len(results),
            "results": results,
            "message": f"已生成 {len(results)} 份报告",
        })

    except Exception as exc:
        return jsonify({"success": False, "message": f"生成失败：{exc}"})
    finally:
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)


# ─────────────────────────────────────────────
# Streaming SSE endpoint
# ─────────────────────────────────────────────

@app.route("/api/generate/stream", methods=["POST"])
def generate_stream():
    target = request.form.get("target", "").strip()
    type_param = request.form.get("type", "")
    enable_web = request.form.get("enable_web_search", "true").lower() == "true"
    selected_keys = _parse_versions(type_param)
    custom_outlines = _parse_outlines(request.form.get("custom_outlines", ""), selected_keys)

    if not target:
        def _err():
            yield f"data: {json.dumps({'type': 'error', 'msg': '请填写产业链或公司名称。'}, ensure_ascii=False)}\n\n"
        return Response(stream_with_context(_err()), mimetype="text/event-stream")

    if not selected_keys:
        def _err():
            yield f"data: {json.dumps({'type': 'error', 'msg': '请至少选择一个报告版本。'}, ensure_ascii=False)}\n\n"
        return Response(stream_with_context(_err()), mimetype="text/event-stream")

    session_dir = TEMP_DIR / str(uuid.uuid4())

    # Save uploaded files eagerly (before streaming starts)
    try:
        structured_files = _save_files(request.files.getlist("files"), session_dir / "structured")
        private_files = _save_files(request.files.getlist("private_files"), session_dir / "private")
        structured_summary = summarize_structured_files(structured_files, target)
        private_summary = summarize_private_materials(private_files, target)
    except Exception as exc:
        shutil.rmtree(session_dir, ignore_errors=True)
        def _err():
            yield f"data: {json.dumps({'type': 'error', 'msg': f'文件处理失败：{exc}'}, ensure_ascii=False)}\n\n"
        return Response(stream_with_context(_err()), mimetype="text/event-stream")

    def event_stream():
        try:
            for report_idx, key in enumerate(selected_keys):
                report_type, version = key.split(":", 1)
                outline = custom_outlines[key]

                if len(selected_keys) > 1:
                    yield f"data: {json.dumps({'type': 'report_start', 'idx': report_idx + 1, 'total': len(selected_keys), 'key': key}, ensure_ascii=False)}\n\n"

                # Web search
                web_data = None
                if enable_web:
                    yield f"data: {json.dumps({'type': 'stage', 'stage': 'searching', 'msg': f'正在联网检索：{target}…'}, ensure_ascii=False)}\n\n"
                    try:
                        web_data = search_for_report(target, outline, report_type)
                    except Exception:
                        web_data = None

                for event in build_report_stream(
                    report_type=report_type,
                    version=version,
                    target=target,
                    outline=outline,
                    structured_summary=structured_summary,
                    private_summary=private_summary,
                    web_search_data=web_data,
                ):
                    # Skip the internal searching stage (already emitted above)
                    if event.get("type") == "stage" and event.get("stage") == "searching":
                        continue
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'msg': f'生成失败：{exc}'}, ensure_ascii=False)}\n\n"
        finally:
            shutil.rmtree(session_dir, ignore_errors=True)

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    print("=" * 54)
    print("报告生成智能体 V3")
    print("访问地址: http://localhost:5002")
    print("=" * 54)
    app.run(host="0.0.0.0", port=5002, debug=False)
