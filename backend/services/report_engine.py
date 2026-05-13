"""
V3 报告生成引擎 — Planner-Executor-Critic 多步骤 Agent 架构。

流程：
  1. Planner  — GLM-5.1 规划每节写作要点
  2. Executor — Qwen 逐节生成，携带前文摘要（节间上下文）
  3. Critic   — GLM-5.1 审查全文，找出弱节
  4. Revise   — Qwen 针对 Critic 意见重写弱节
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime

from .llm_client import chat_completion, chat_completion_structured, llm_enabled
from .report_catalog import get_type_meta, get_version_meta
from .web_search import format_results_as_context

# ─────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "你是一位资深产业链研究分析师，曾供职于头部咨询机构和商业银行研究部，"
    "拥有丰富的行业研究和信贷分析经验。"
    "你的任务是撰写一份关于【{target}】的专业产业链分析报告中的具体章节。\n\n"
    "【写作规范 — 必须严格遵守】\n\n"
    "1. 段落形式：核心分析内容必须用连贯的分析段落表达。每个小节至少写 3 段，每段 120~220 字。\n"
    "   - 正确：用段落分析背景、现状、结论，段间有逻辑衔接\n"
    "   - 错误：全文都是 bullet list，每行一个短句\n\n"
    "2. 段落结构：遵循 背景/定义 -> 现状/规模 -> 深度分析 -> 结论/展望 的逻辑框架\n\n"
    "3. 数据具体化：\n"
    "   - 如果联网检索提供了具体数据，请直接引用（市场规模、增速、占比等）\n"
    "   - 如果没有检索数据，可基于行业通识给出合理估算，注明（据行业估计）\n"
    "   - 引用具体企业名称、政策文件名等\n\n"
    "4. 局部列举：列举企业名单、政策文件、技术类型等条目时可用 bullet 或表格，"
    "但列举前后必须有段落分析，不能只有列举。\n\n"
    "5. 写作风格：专业、客观、严谨，符合银行业研究报告或咨询机构产业报告的语言风格；"
    "避免口语化；不使用第一人称。\n\n"
    "6. 格式要求：\n"
    "   - 直接开始写内容，不要重复输出章节标题（标题在外层处理）\n"
    "   - 可以用 ### 小标题组织子结构，但小标题下必须有段落内容\n"
    "   - 不要用 markdown 代码块包裹正文\n\n"
    "报告背景：{report_type_context}"
)

_REPORT_TYPE_CONTEXT = {
    "chain_entirety": (
        "本报告为产业链整体分析报告，读者为产业研究人员、机构投资者和政府政策研究人员，"
        "需要从宏观到微观全面了解该产业链的定义、格局、竞争态势和发展趋势。"
    ),
    "trade_data": (
        "本报告为产业链交易分析报告，读者为商业银行授信/风控人员，"
        "重点关注交易规模、结构、集中度、稳定性和信贷风险评估。"
    ),
    "company": (
        "本报告为公司具体分析报告，读者为尽调团队和投资委员会，"
        "重点关注企业基本面、产业链地位、核心竞争力和风险因素。"
    ),
}

# ─────────────────────────────────────────────
# Per-section user prompts (unchanged)
# ─────────────────────────────────────────────

_SECTION_PROMPTS: dict[str, str] = {
    "产业链定义与概述": (
        "请撰写【{target}产业链】的定义与概述章节。\n\n"
        "涵盖以下内容（以段落形式展开）：\n"
        "1. {target}产业链的定义、研究边界和核心概念\n"
        "2. 产业链上中下游各环节的划分，各环节的主要产品/服务及其功能\n"
        "3. 该产业链的战略地位、经济意义和关键特征（如技术密集度、政策敏感性等）\n\n"
        "{web_context}"
    ),
    "产业链发展历程与现状": (
        "请撰写【{target}产业链】的发展历程与现状章节。\n\n"
        "涵盖以下内容（以段落形式展开）：\n"
        "1. 全球{target}产业的发展阶段划分（萌芽期 -> 成长期 -> 成熟期 -> 当前阶段）"
        "，每阶段有具体时间节点和标志事件\n"
        "2. 中国{target}产业的发展历程与当前市场规模（引用具体数据）\n"
        "3. 当前产业发展的主要特点、政策环境及核心驱动因素\n\n"
        "{web_context}"
    ),
    "产业链细分领域分析": (
        "请撰写【{target}产业链】的细分领域分析章节。\n\n"
        "涵盖以下内容（以段落形式展开，可用 ### 小标题区分上中下游）：\n"
        "1. 上游环节：核心原材料/零部件/技术，主要供应商，技术壁垒，国产化现状\n"
        "2. 中游环节：核心产品或制造环节，技术路线，主要企业，产能与竞争格局\n"
        "3. 下游环节：主要应用场景或客户群体，市场需求特点，驱动因素\n"
        "4. 各环节的技术难点和发展机遇（合并为一段综合分析）\n\n"
        "{web_context}"
    ),
    "产业链竞争格局分析": (
        "请撰写【{target}产业链】的竞争格局分析章节。\n\n"
        "涵盖以下内容（以段落形式展开）：\n"
        "1. 全球竞争格局：主要国家/地区的市场地位，代表企业及其优势，市场份额估计\n"
        "2. 中国市场竞争格局：国产企业 vs 外资企业的竞争态势，市场集中度\n"
        "3. 主要竞争维度分析：技术壁垒、资金壁垒、品牌壁垒、渠道壁垒\n"
        "4. 竞争趋势：国产替代进度、新进入者、潜在颠覆性技术\n\n"
        "{web_context}"
    ),
    "产业链发展趋势与展望": (
        "请撰写【{target}产业链】的发展趋势与展望章节。\n\n"
        "涵盖以下内容（以段落形式展开）：\n"
        "1. 技术发展趋势：未来3-5年的核心技术演进方向，代表性技术路线\n"
        "2. 市场规模预测：未来3-5年的增速预判，增长驱动因素\n"
        "3. 应用场景拓展：新兴应用领域，跨界融合机会\n"
        "4. 政策环境展望：国家政策方向，可能的支持或限制因素\n"
        "5. 主要挑战：技术挑战、市场挑战、供应链安全等\n\n"
        "{web_context}"
    ),
    "产业链交易特征分析": (
        "请撰写【{target}产业链】的交易特征分析章节。\n\n"
        "涵盖以下内容（以段落形式展开）：\n"
        "1. 交易规模与增速：整体交易金额，近年变化趋势\n"
        "2. 交易结构分析：上中下游交易占比，主要交易类型\n"
        "3. 交易集中度与稳定性：主要交易对手集中度，长期合作关系比例\n"
        "4. 交易风险特征：账期、账款质量、季节性波动\n\n"
        "{web_context}"
    ),
    "结论与建议": (
        "请撰写【{target}产业链】的结论与建议章节。\n\n"
        "涵盖以下内容（以段落形式展开，每条建议用一段展开，不要只列标题）：\n"
        "1. 主要结论：3-5条核心判断，每条用一段展开分析\n"
        "2. 投资/信贷/政策建议：3-5条具体可操作的建议，说明理由\n"
        "3. 风险提示：2-3个需要重点关注的风险点\n\n"
        "{web_context}"
    ),
    "报告摘要": (
        "请撰写【{target}产业链交易分析报告】的摘要章节。\n\n"
        "以3-4段段落形式写出：报告背景与分析目的；主要发现（交易规模、结构特征）；"
        "核心风险点；主要建议。语言精炼，信息密度高，适合快速阅读。\n\n"
        "{web_context}"
    ),
    "产业链整体交易情况": (
        "请撰写【{target}产业链】整体交易情况章节。\n\n"
        "以段落形式分析：整体交易规模与趋势、上中下游交易分布、主要交易特征。\n\n"
        "{web_context}"
    ),
    "交易结构分析": (
        "请撰写【{target}产业链】交易结构分析章节。\n\n"
        "以段落形式分析：交易类型分布、地区分布、季节性规律、交易集中度。\n\n"
        "{web_context}"
    ),
    "交易对手分析": (
        "请撰写【{target}产业链】交易对手分析章节。\n\n"
        "以段落形式分析：主要交易对手特征、集中度、依赖风险、多元化程度。\n\n"
        "{web_context}"
    ),
    "交易稳定性分析": (
        "请撰写【{target}产业链】交易稳定性分析章节。\n\n"
        "以段落形式分析：长期合作关系比例、客户/供应商保留率、收入波动性。\n\n"
        "{web_context}"
    ),
    "风险提示与授信建议": (
        "请撰写【{target}产业链】风险提示与授信建议章节。\n\n"
        "以段落形式分析：主要信贷风险类型及程度评估；具体授信建议（额度方向、期限、担保）；"
        "需重点监控的风险指标。\n\n"
        "{web_context}"
    ),
    "核心指标总览": (
        "请撰写【{target}产业链】核心指标总览章节。\n\n"
        "以段落形式介绍：关键财务和业务指标的现状与含义。\n\n"
        "{web_context}"
    ),
    "交易趋势判断": (
        "请撰写【{target}产业链】交易趋势判断章节。\n\n"
        "以段落形式分析：近期交易量趋势、驱动因素、未来预判。\n\n"
        "{web_context}"
    ),
    "重点风险观察": (
        "请撰写【{target}产业链】重点风险观察章节。\n\n"
        "以段落形式分析3-4个核心风险点，每个风险点说明成因、当前程度和影响。\n\n"
        "{web_context}"
    ),
    "结论": (
        "请撰写【{target}产业链交易分析报告】的结论章节。\n\n"
        "以2-3段段落形式总结核心发现，给出明确的风险评级和行动建议。\n\n"
        "{web_context}"
    ),
    "公司基本信息与业务概况": (
        "请撰写【{target}】公司分析报告的基本信息与业务概况章节。\n\n"
        "以段落形式介绍：公司成立背景、主营业务范围、核心产品/服务、规模体量（收入、员工、市值等）、"
        "近年业务发展轨迹。\n\n"
        "{web_context}"
    ),
    "产业链定位分析": (
        "请撰写【{target}】在产业链中的定位分析章节。\n\n"
        "以段落形式分析：公司在产业链的位置（上/中/下游）、与上下游的关系、"
        "产业链中的话语权和议价能力。\n\n"
        "{web_context}"
    ),
    "供应商与上游分析": (
        "请撰写【{target}】供应商与上游分析章节。\n\n"
        "以段落形式分析：主要供应商构成、上游依赖度、供应链稳定性、原材料/零部件风险。\n\n"
        "{web_context}"
    ),
    "客户与下游分析": (
        "请撰写【{target}】客户与下游分析章节。\n\n"
        "以段落形式分析：主要客户构成、客户集中度、客户黏性、下游市场需求。\n\n"
        "{web_context}"
    ),
    "竞争优势与核心壁垒": (
        "请撰写【{target}】竞争优势与核心壁垒章节。\n\n"
        "以段落形式分析：技术壁垒、规模效应、品牌效应、渠道优势、人才优势等，"
        "并评估这些优势的可持续性。\n\n"
        "{web_context}"
    ),
    "财务与经营风险分析": (
        "请撰写【{target}】财务与经营风险分析章节。\n\n"
        "以段落形式分析：主要财务风险（流动性、杠杆、盈利能力）、经营风险（市场、技术、政策）、"
        "已暴露的风险信号。\n\n"
        "{web_context}"
    ),
    "技术实力与创新能力": (
        "请撰写【{target}】技术实力与创新能力章节。\n\n"
        "以段落形式分析：核心技术领域、专利布局、研发投入、技术合作与并购。\n\n"
        "{web_context}"
    ),
    "行业地位与发展前景": (
        "请撰写【{target}】行业地位与发展前景章节。\n\n"
        "以段落形式分析：当前市场份额与排名、与竞争对手对比、未来3年增长预判。\n\n"
        "{web_context}"
    ),
    "尽调总结与投资建议": (
        "请撰写【{target}】尽调总结与投资建议章节。\n\n"
        "以段落形式写出：综合评价（优势、劣势、机会、风险）；明确的投资/授信建议；"
        "需重点关注的后续跟踪事项。\n\n"
        "{web_context}"
    ),
}

_DEFAULT_SECTION_PROMPT = (
    "请为【{target}产业链】报告中的【{section_title}】章节撰写内容。\n\n"
    "要求：以3-4个连贯分析段落展开，每段120-200字，逻辑层次清晰，"
    "结合行业特点给出有深度的专业判断。\n\n"
    "{web_context}"
)

CN_NUMS = [
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
    "十一", "十二", "十三", "十四", "十五",
]


def _cn(index: int) -> str:
    return CN_NUMS[index - 1] if 1 <= index <= len(CN_NUMS) else str(index)


def _toc(outline: list[str]) -> str:
    lines = []
    for i, title in enumerate(outline, 1):
        anchor = (
            _cn(i)
            + title.replace("（", "").replace("）", "").replace(" ", "").replace("/", "")
        )
        lines.append(f"{i}. [{_cn(i)}、{title}](#{anchor})")
    return "\n".join(lines)


def _web_context_block(results: list[dict]) -> str:
    if not results:
        return ""
    ctx = format_results_as_context(results, max_items=5)
    if not ctx:
        return ""
    return (
        "\n\n**联网检索到的参考信息（请合理引用具体数据，注明来源）：**\n"
        + ctx
        + "\n"
    )


def _build_user_prompt(
    section_title: str,
    target: str,
    web_results: list[dict],
    prior_context: str = "",
    section_plan: str = "",
    critic_note: str = "",
) -> str:
    web_block = _web_context_block(web_results)
    template = _SECTION_PROMPTS.get(section_title, _DEFAULT_SECTION_PROMPT)
    prompt = template.format(
        target=target,
        section_title=section_title,
        web_context=web_block,
    )
    if prior_context:
        prompt += (
            "\n\n**前文已覆盖的主要内容（请勿重复，注意与前文衔接）：**\n"
            + prior_context
        )
    if section_plan:
        prompt += (
            "\n\n**本节写作重点（Planner 规划）：**\n"
            + section_plan
        )
    if critic_note:
        prompt += (
            "\n\n**审稿意见（请针对以下问题修订，重新撰写本节）：**\n"
            + critic_note
        )
    return prompt


# ─────────────────────────────────────────────
# Planner
# ─────────────────────────────────────────────

def _plan_report(
    target: str,
    outline: list[str],
    report_type: str,
    web_results: list[dict],
) -> dict[str, str]:
    """One GLM call to produce per-section writing keys. Returns {} on failure (graceful degradation)."""
    if not llm_enabled():
        return {}

    web_ctx = format_results_as_context(web_results, max_items=6)
    sections_str = "\n".join(f"- {t}" for t in outline)

    system = (
        "你是一位资深产业链报告策划师，负责为分析师规划报告的写作方向。"
        "请严格按照要求输出 JSON，不要输出任何其他内容。"
    )
    user = (
        f"请为【{target}】产业链分析报告规划写作要点。\n\n"
        f"报告类型：{_REPORT_TYPE_CONTEXT.get(report_type, '产业链分析')}\n\n"
        f"章节列表：\n{sections_str}\n\n"
        f"联网检索摘要：\n{web_ctx}\n\n"
        "请输出一个 JSON 对象，key 为章节标题，value 为该章节的写作要点（50-100字），"
        "要点应包含：重点分析角度、建议引用的数据维度、与其他章节的差异化方向。\n\n"
        "输出格式示例：\n"
        '{"产业链定义与概述": "重点梳理上中下游边界，引用市场规模数据，避免与发展历程章节重叠", ...}\n\n'
        "只输出 JSON，不要有任何说明文字或代码块标记。"
    )

    result = chat_completion_structured(system, user, max_tokens=1500)
    if not result.get("ok"):
        return {}

    content = result.get("content", "").strip()
    # Strip markdown code fences if present
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        plan = json.loads(content)
        if isinstance(plan, dict):
            return {k: str(v) for k, v in plan.items()}
    except (json.JSONDecodeError, ValueError):
        pass
    return {}


# ─────────────────────────────────────────────
# Section summarizer (for inter-section context)
# ─────────────────────────────────────────────

def _summarize_section(title: str, content: str) -> str:
    """Return a 1-2 sentence summary of a generated section. Falls back to truncation."""
    if not llm_enabled():
        return content[:200]

    system = "你是一位文本摘要助手，请用1-2句话概括以下章节的核心观点，不超过100字。只输出摘要，无需任何说明。"
    user = f"章节标题：{title}\n\n内容：\n{content[:2000]}"

    result = chat_completion_structured(system, user, max_tokens=150)
    if result.get("ok") and result.get("content", "").strip():
        return result["content"].strip()
    return content[:200]


# ─────────────────────────────────────────────
# Critic
# ─────────────────────────────────────────────

def _critique_report(
    target: str,
    report_type: str,
    sections_data: list[dict],
) -> dict[str, str]:
    """GLM reviews all sections, returns {title: issue} for up to 3 weak sections."""
    if not llm_enabled() or not sections_data:
        return {}

    # Compact representation: title + first 300 chars of content
    sections_compact = "\n\n".join(
        f"【{s['title']}】\n{s['content'][:300]}…"
        for s in sections_data
    )

    system = (
        "你是一位严格的产业报告审稿专家。请审查报告各章节，找出需要修订的问题。"
        "请严格按照要求输出 JSON，不要输出任何其他内容。"
    )
    user = (
        f"请审查以下【{target}】产业链分析报告的各章节（仅显示摘要），"
        "找出最多3个需要修订的章节，给出具体问题描述。\n\n"
        f"{sections_compact}\n\n"
        "审查标准：\n"
        "1. 分析过于泛泛，缺乏具体数据支撑\n"
        "2. 内容与其他章节严重重复\n"
        "3. 未覆盖章节应有的核心内容\n\n"
        "输出格式（JSON）：\n"
        '{"章节标题": "具体问题：…，修订建议：…"}\n\n'
        "若所有章节质量合格，输出空对象 {}。\n"
        "只输出 JSON，不要有任何说明文字或代码块标记。"
    )

    result = chat_completion_structured(system, user, max_tokens=600)
    if not result.get("ok"):
        return {}

    content = result.get("content", "").strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        critique = json.loads(content)
        if isinstance(critique, dict):
            # Limit to 3 revisions
            return dict(list(critique.items())[:3])
    except (json.JSONDecodeError, ValueError):
        pass
    return {}


# ─────────────────────────────────────────────
# Section generator (Executor)
# ─────────────────────────────────────────────

def _generate_section(
    section_title: str,
    report_type: str,
    target: str,
    web_results: list[dict],
    prior_context: str = "",
    section_plan: str = "",
    critic_note: str = "",
) -> tuple[str, str]:
    """Return (content, source) where source is 'llm', 'llm_revised', or 'fallback_...'."""
    if not llm_enabled():
        return _fallback_section(section_title, target), "fallback_no_key"

    system_prompt = _SYSTEM_PROMPT.format(
        target=target,
        report_type_context=_REPORT_TYPE_CONTEXT.get(report_type, ""),
    )
    user_prompt = _build_user_prompt(
        section_title, target, web_results,
        prior_context=prior_context,
        section_plan=section_plan,
        critic_note=critic_note,
    )

    result = chat_completion(system_prompt, user_prompt, max_tokens=3000)
    if result.get("ok") and result.get("content", "").strip():
        source = "llm_revised" if critic_note else "llm"
        return result["content"].strip(), source

    return _fallback_section(section_title, target), f"fallback_{result.get('error', 'unknown')}"


def _fallback_section(section_title: str, target: str) -> str:
    return (
        f"本节围绕**{target}**的「{section_title}」展开分析。\n\n"
        f"{target}作为重要的产业链研究对象，其{section_title}方面具有显著的行业特点和发展规律。"
        f"从产业链整体视角来看，该环节在价值链中承担关键功能，与上下游之间存在紧密的"
        f"技术经济联系，共同构成了完整的价值创造体系。\n\n"
        f"如需获取 LLM 驱动的深度段落分析，请配置 SILICONFLOW_API_KEY 或 DASHSCOPE_API_KEY，"
        f"然后重启服务。"
    )


# ─────────────────────────────────────────────
# Uploaded file context helpers
# ─────────────────────────────────────────────

def _data_context(structured_summary: dict) -> str:
    overview = structured_summary.get("overview", {})
    if not overview.get("file_count"):
        return ""
    items = structured_summary.get("items", [])
    lines = [f"**已上传结构化数据（{overview['file_count']} 份）：**"]
    for item in items[:3]:
        rows = f"{item['rows']} 行" if item.get("rows") is not None else "行数未知"
        cols = ", ".join(item.get("columns", [])[:8])
        lines.append(f"- `{item['name']}`：{rows}，字段：{cols}")
    return "\n".join(lines)


def _private_context(private_summary: dict) -> str:
    items = private_summary.get("items", [])
    if not items:
        return ""
    lines = [f"**已上传私有材料（{len(items)} 份）：**"]
    for item in items[:4]:
        lines.append(f"- `{item['name']}`：{item.get('preview', '')[:200]}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Markdown assembly
# ─────────────────────────────────────────────

def _assemble_markdown(
    target: str,
    report_type: str,
    version: str,
    outline: list[str],
    sections_md: list[str],
    web_search_data: dict | None,
    llm_used: bool,
) -> str:
    type_meta = get_type_meta(report_type)
    version_meta = get_version_meta(report_type, version)
    now = datetime.now().strftime("%Y-%m-%d")

    if report_type == "company":
        title = f"# {target}分析报告"
    elif report_type == "trade_data":
        title = f"# {target}产业链交易分析报告"
    else:
        title = f"# {target}产业链深度分析报告（{version_meta['label']}）"

    gen_note = " · Planner-Executor-Critic Agent" if llm_used else " · 规则模式（未配置 API Key）"

    if web_search_data and web_search_data.get("ok"):
        web_refs: list[str] = []
        for r in web_search_data.get("results", [])[:8]:
            url = r.get("url", "")
            t = r.get("title", "未命名")
            s = r.get("snippet") or r.get("full_snippet", "")
            if url:
                web_refs.append(f"- **{t}**\n  - 链接：{url}\n  - 摘要：{s[:200]}")
        if web_refs:
            sections_md = list(sections_md) + ["## 附录、联网检索参考来源\n\n" + "\n\n".join(web_refs)]

    return (
        f"{title}\n\n"
        f"**报告类型**：{type_meta['label']} · {version_meta['label']}\n\n"
        f"**分析对象**：{target}\n\n"
        f"**生成日期**：{now}\n\n"
        f"**生成方式**：报告生成智能体 V3{gen_note}\n\n"
        "---\n\n"
        "## 目录\n\n"
        + _toc(outline)
        + "\n\n---\n\n"
        + "\n\n---\n\n".join(sections_md)
        + "\n\n---\n\n*本报告由报告生成智能体 V3 自动生成，仅供参考*"
    )


# ─────────────────────────────────────────────
# Public API — synchronous (backward compatible)
# ─────────────────────────────────────────────

def build_report(
    report_type: str,
    version: str,
    target: str,
    outline: list[str],
    structured_summary: dict,
    private_summary: dict,
    web_search_data: dict | None = None,
    progress_callback=None,
) -> dict:
    """Synchronous Planner-Executor-Critic flow. Returns dict with markdown, sections_meta, etc."""
    result = {}
    for event in build_report_stream(
        report_type=report_type,
        version=version,
        target=target,
        outline=outline,
        structured_summary=structured_summary,
        private_summary=private_summary,
        web_search_data=web_search_data,
        progress_callback=progress_callback,
    ):
        if event.get("type") == "section_start" and progress_callback:
            progress_callback(event["idx"], event["total"], event["title"])
        if event.get("type") == "complete":
            result = event
    return {
        "markdown": result.get("markdown", ""),
        "sections_meta": result.get("sections_meta", []),
        "llm_used": result.get("llm_used", False),
        "elapsed_sec": result.get("elapsed_sec", 0),
    }


# ─────────────────────────────────────────────
# Public API — streaming generator
# ─────────────────────────────────────────────

def build_report_stream(
    report_type: str,
    version: str,
    target: str,
    outline: list[str],
    structured_summary: dict,
    private_summary: dict,
    web_search_data: dict | None = None,
    progress_callback=None,
):
    """
    Generator that yields SSE-ready event dicts.

    Event types:
      stage        — {"type":"stage", "stage": str, "msg": str}
      section_start — {"type":"section_start", "idx": int, "total": int, "title": str}
      section_done  — {"type":"section_done", "idx": int, "title": str, "content": str}
      revision      — {"type":"revision", "idx": int, "title": str}
      complete      — {"type":"complete", "markdown": str, "elapsed_sec": float,
                        "llm_used": bool, "sections_meta": list, "web_search_ok": bool,
                        "filename": str}
    """
    t_start = time.time()

    # ── Phase 0: Planner ──────────────────────────────────────────────────
    yield {"type": "stage", "stage": "planning", "msg": "正在规划报告结构…（Planner）"}

    all_web_results = (web_search_data or {}).get("results", [])
    writing_plan = _plan_report(target, outline, report_type, all_web_results)

    # ── Phase 1: Build per-section web result map ─────────────────────────
    yield {"type": "stage", "stage": "searching", "msg": "正在整理联网检索数据…"}

    by_section: dict[str, list[dict]] = {}
    if web_search_data:
        for sec_data in web_search_data.get("by_section", []):
            by_section[sec_data.get("section", "")] = sec_data.get("results", [])
    global_results: list[dict] = (web_search_data or {}).get("results", [])

    # Uploaded file context
    data_ctx = _data_context(structured_summary)
    private_ctx = _private_context(private_summary)

    # ── Phase 2: Executor (section by section with context) ───────────────
    sections_md: list[str] = []
    sections_data: list[dict] = []   # for Critic
    sections_meta: list[dict] = []
    prior_summaries: list[str] = []  # rolling inter-section context

    for idx, section_title in enumerate(outline, 1):
        yield {"type": "section_start", "idx": idx, "total": len(outline), "title": section_title}

        sec_web = by_section.get(section_title, []) or global_results[:4]
        prior_context = "\n".join(prior_summaries[-3:]) if prior_summaries else ""
        section_plan = writing_plan.get(section_title, "")

        content, source = _generate_section(
            section_title=section_title,
            report_type=report_type,
            target=target,
            web_results=sec_web,
            prior_context=prior_context,
            section_plan=section_plan,
        )

        # Attach uploaded file context note to first section only
        if idx == 1 and (data_ctx or private_ctx):
            extra = "\n\n---\n\n**本报告补充材料说明：**\n\n"
            if data_ctx:
                extra += data_ctx + "\n\n"
            if private_ctx:
                extra += private_ctx
            content = content + extra

        sections_md.append(f"## {_cn(idx)}、{section_title}\n\n{content}")
        sections_data.append({"title": section_title, "content": content})
        sections_meta.append({"title": section_title, "source": source, "index": idx})

        yield {"type": "section_done", "idx": idx, "title": section_title, "content": content}

        # Generate summary for inter-section context (non-blocking on failure)
        summary = _summarize_section(section_title, content)
        prior_summaries.append(f"第{_cn(idx)}节「{section_title}」：{summary}")

    # ── Phase 3: Critic ───────────────────────────────────────────────────
    yield {"type": "stage", "stage": "critic", "msg": "正在审查报告质量…（Critic）"}

    critique = _critique_report(target, report_type, sections_data)

    # ── Phase 4: Revise flagged sections ──────────────────────────────────
    for section_title, critic_note in critique.items():
        # Find the section index
        idx = next(
            (m["index"] for m in sections_meta if m["title"] == section_title),
            None,
        )
        if idx is None:
            continue

        yield {"type": "revision", "idx": idx, "title": section_title}

        sec_web = by_section.get(section_title, []) or global_results[:4]
        revised_content, revised_source = _generate_section(
            section_title=section_title,
            report_type=report_type,
            target=target,
            web_results=sec_web,
            critic_note=critic_note,
        )

        # Patch sections_md and meta in-place
        array_idx = idx - 1
        sections_md[array_idx] = f"## {_cn(idx)}、{section_title}\n\n{revised_content}"
        sections_meta[array_idx]["source"] = "llm_revised"

    # ── Assemble final markdown ───────────────────────────────────────────
    llm_used = any(m["source"] in ("llm", "llm_revised") for m in sections_meta)

    markdown = _assemble_markdown(
        target=target,
        report_type=report_type,
        version=version,
        outline=outline,
        sections_md=sections_md,
        web_search_data=web_search_data,
        llm_used=llm_used,
    )

    from .report_catalog import get_output_filename
    yield {
        "type": "complete",
        "markdown": markdown,
        "sections_meta": sections_meta,
        "llm_used": llm_used,
        "elapsed_sec": round(time.time() - t_start, 1),
        "web_search_ok": bool(web_search_data and web_search_data.get("ok")),
        "filename": get_output_filename(report_type, version, target),
        "critic_revisions": list(critique.keys()),
    }
