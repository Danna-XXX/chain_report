"""Report type catalog and default outlines."""

from __future__ import annotations

from copy import deepcopy

CATALOG = {
    "chain_entirety": {
        "label": "产业链整体报告",
        "icon": "🏢",
        "description": "适合行业研究、产业图谱梳理和趋势判断。",
        "versions": {
            "no_data": {
                "label": "无数据版",
                "description": "纯联网检索 + LLM 生成，无需上传数据。",
                "outline": [
                    "产业链定义与概述",
                    "产业链发展历程与现状",
                    "产业链细分领域分析",
                    "产业链竞争格局分析",
                    "产业链发展趋势与展望",
                    "结论与建议",
                ],
            },
            "data": {
                "label": "数据版",
                "description": "结合上传结构化数据的增强分析版本。",
                "outline": [
                    "产业链定义与概述",
                    "产业链发展历程与现状",
                    "产业链细分领域分析",
                    "产业链交易特征分析",
                    "产业链竞争格局分析",
                    "产业链发展趋势与展望",
                    "结论与建议",
                ],
            },
        },
    },
    "trade_data": {
        "label": "产业链交易报告",
        "icon": "📊",
        "description": "偏交易结构与风控视角，适合银行授信场景。",
        "versions": {
            "bank": {
                "label": "详细版（银行版）",
                "description": "适合风控、授信、尽调场景。",
                "outline": [
                    "报告摘要",
                    "产业链整体交易情况",
                    "交易结构分析",
                    "交易对手分析",
                    "交易稳定性分析",
                    "风险提示与授信建议",
                ],
            },
            "frontend": {
                "label": "简略版",
                "description": "适合前端展示和汇报。",
                "outline": [
                    "报告摘要",
                    "核心指标总览",
                    "交易趋势判断",
                    "重点风险观察",
                    "结论",
                ],
            },
        },
    },
    "company": {
        "label": "公司具体报告",
        "icon": "🏭",
        "description": "企业尽调、上下游分析和风险判断。",
        "versions": {
            "default": {
                "label": "默认版",
                "description": "融合联网信息与上传材料的企业深度分析。",
                "outline": [
                    "公司基本信息与业务概况",
                    "产业链定位分析",
                    "供应商与上游分析",
                    "客户与下游分析",
                    "竞争优势与核心壁垒",
                    "财务与经营风险分析",
                    "技术实力与创新能力",
                    "行业地位与发展前景",
                    "尽调总结与投资建议",
                ],
            }
        },
    },
}


def build_catalog_payload() -> dict:
    return deepcopy(CATALOG)


def get_default_outline(version_key: str) -> list[str]:
    report_type, version = version_key.split(":", 1)
    return deepcopy(CATALOG[report_type]["versions"][version]["outline"])


def normalize_outline(outline: list | None, fallback: list | None = None) -> list[str]:
    source = outline or fallback or []
    seen: set[str] = set()
    result = []
    for item in source:
        title = str(item).strip()
        if title and title not in seen:
            seen.add(title)
            result.append(title)
    return result


def get_version_meta(report_type: str, version: str) -> dict:
    return CATALOG[report_type]["versions"][version]


def get_type_meta(report_type: str) -> dict:
    return CATALOG[report_type]


def get_output_filename(report_type: str, version: str, target: str) -> str:
    label = get_version_meta(report_type, version)["label"]
    if report_type == "company":
        return f"{target}分析报告_{label}.md"
    if report_type == "trade_data":
        return f"{target}产业链交易报告_{label}.md"
    return f"{target}产业链深度分析报告_{label}.md"
