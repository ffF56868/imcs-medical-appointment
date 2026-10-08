"""内容审核：输入侧 prompt 注入检测，输出侧医疗合规校验。

策略是**规则优先、模型兜底**。纯模型审核每次多一轮 LLM 调用，
延迟与成本都翻倍；纯规则又覆盖不了语义变体。工程上的合理取舍是：
规则拦截绝大多数明确违规（成本低、延迟近零、可解释、可审计），
只把规则拿不准的边界情况交给小模型二次判断。

医疗输出审核的重点不是「有没有脏话」，而是**有没有越过 AI 的边界**：
系统不能给出确诊结论、不能保证疗效、不能替代医生开具用药方案。
这些是合规红线，一旦越界即使内容医学上正确也必须拦截。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.services.llm.types import LLMScenario

logger = logging.getLogger(__name__)


# ==================== 输入侧：prompt 注入检测 ====================

# 注入尝试的典型模式。中英文都要覆盖——只用英文规则会被中文绕过。
INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|above|prior)", re.I),
    re.compile(r"forget\s+(everything|all\s+(previous|your)\s+(instructions|rules))", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an|in)\s+", re.I),
    re.compile(r"(system|developer)\s*prompt\s*[:：]", re.I),
    re.compile(r"reveal|show|print|output|repeat\s+.{0,20}(system\s*prompt|instructions|rules)", re.I),
    re.compile(r"jailbreak|dan\s*mode|developer\s*mode", re.I),
    # 中文变体
    re.compile(r"(忽略|无视|忘记|丢弃)[^\n]{0,10}(之前|上述|上面|以上|先前)[^\n]{0,10}(指令|指示|规则|提示|设定|约束)"),
    re.compile(r"(你现在是|从现在起你是|扮演|假装你是)[^\n]{0,20}(医生|专家|系统|管理员|AI)"),
    re.compile(r"(输出|显示|告诉我|重复|打印)[^\n]{0,15}(系统提示|提示词|指令|设定|规则)"),
    re.compile(r"(进入|开启|切换到)[^\n]{0,6}(开发者模式|越狱模式|无限制模式)"),
    # 伪造角色标记，试图在用户输入里插入假的 system 消息
    re.compile(r"^\s*(<\|?\s*(system|im_start|im_end)\s*\|?>|\[/?system\])", re.I | re.M),
)

# 输入长度上限。超长输入既可能是注入尝试，也会吃掉大量 token 预算，
# 在审核阶段就拦下来比进到 LLM 再失败更省成本。
MAX_INPUT_CHARACTERS = 8000


# ==================== 输出侧：医疗合规 ====================

# 绝对化诊断与疗效承诺。这些表述越过 AI 边界，必须拦截。
ABSOLUTE_DIAGNOSIS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(你|患者|病人)(肯定|一定|确定|必然)(是|患有|得的是)"),
    re.compile(r"(确诊|诊断为|明确诊断)(你|您|患者|病人)?(为|是|患有)"),
    re.compile(r"(100%|百分之百|完全|彻底|绝对)(治愈|根治|痊愈|消除|恢复)"),
    re.compile(r"(保证|确保|承诺|一定)(能|可以)?(治愈|治好|根治|痊愈|有效)"),
    re.compile(r"(包治|根治|药到病除|永不复发|一次性治愈)"),
    re.compile(r"(无需|不用|不必)(就医|看医生|去医院|就诊|检查)"),
    re.compile(r"(停止|停用|不要再吃|不要再服用)[^\n]{0,15}(处方药|药|药物)"),
)

# 高风险用药建议：AI 不应给出具体处方剂量方案
DOSAGE_PRESCRIPTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "每次 X 片，每日 Y 次，连续 Z 天" 这类完整给药方案
    re.compile(r"(每次|一次)[^\n]{0,10}\d+[^\n]{0,6}(片|粒|毫升|ml|mg|克|g|单位|IU)[^\n]{0,20}(每天|每日|一日)[^\n]{0,6}\d+\s*次"),
    re.compile(r"(口服|静脉(滴注|注射)|肌(肉)?注(射)?)[^\n]{0,15}\d+[^\n]{0,8}(mg|g|ml|毫升|克|单位)"),
)

# 需要伴随就医提示的高风险主题
HIGH_RISK_TOPICS: tuple[str, ...] = (
    "自杀",
    "自残",
    "自伤",
    "轻生",
    "不想活",
    "结束生命",
    " overdose",
    "过量服用",
    "药物过量",
    "中毒",
)

# 输出必须包含的免责说明关键词之一。缺失即视为不合规——
# 医疗 AI 回答不带免责与就医提示，本身就是产品缺陷。
REQUIRED_DISCLAIMER_KEYWORDS: tuple[str, ...] = (
    "医生",
    "就医",
    "就诊",
    "专业",
    "线下",
    "医疗机构",
)


@dataclass(slots=True)
class ModerationIssue:
    """单条审核问题。"""

    category: str  # injection / absolute_diagnosis / dosage_prescription / missing_disclaimer / high_risk / too_long
    severity: str  # block（必须拦截）/ warn（记录但放行）
    matched: str = ""
    message: str = ""


@dataclass(slots=True)
class ModerationResult:
    """审核结果。"""

    passed: bool
    blocked: bool = False
    issues: list[ModerationIssue] = field(default_factory=list)
    # 高风险主题需要附加的提示文案
    appended_notice: str | None = None
    # 命中规则数，落库用于统计审核拦截率
    rule_hit_count: int = 0

    @property
    def categories(self) -> list[str]:
        return [issue.category for issue in self.issues]


# 心理危机场景的固定提示。这类情况优先级高于一切业务回答，
# 不能只做「拦截」而要给出口。
MENTAL_HEALTH_NOTICE = (
    "如果你正在经历强烈的痛苦或有伤害自己的念头，请立即联系信任的人，"
    "或拨打心理援助热线（中国大陆：12356 心理援助热线；北京心理危机研究与干预中心：010-82951332）。"
    "这些服务是免费且保密的。本系统无法提供紧急心理干预。"
)

EMERGENCY_NOTICE = (
    "你描述的情况可能属于医疗急症。请立即拨打 120 或前往最近的急诊科，"
    "不要依赖在线回答延误救治。"
)


def moderate_input(text: str) -> ModerationResult:
    """输入侧审核：注入检测与长度限制。

    只做规则判断，不调模型——输入审核在每个请求的关键路径上，
    加一次 LLM 调用会让首 token 延迟翻倍，得不偿失。
    """
    issues: list[ModerationIssue] = []

    if not isinstance(text, str) or not text.strip():
        return ModerationResult(passed=False, blocked=True, issues=[
            ModerationIssue(category="empty_input", severity="block", message="输入为空")
        ])

    if len(text) > MAX_INPUT_CHARACTERS:
        issues.append(
            ModerationIssue(
                category="too_long",
                severity="block",
                matched=f"{len(text)} characters",
                message=f"输入超过 {MAX_INPUT_CHARACTERS} 字符上限",
            )
        )

    for pattern in INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            issues.append(
                ModerationIssue(
                    category="injection",
                    severity="block",
                    # 只记匹配片段前 80 字符，避免把完整注入 payload 落库
                    matched=match.group(0)[:80],
                    message="检测到疑似 prompt 注入尝试",
                )
            )
            # 命中一条就够定性，不必扫完全部模式
            break

    blocked = any(issue.severity == "block" for issue in issues)
    return ModerationResult(
        passed=not blocked,
        blocked=blocked,
        issues=issues,
        rule_hit_count=len(issues),
    )


def _detect_high_risk_topic(text: str) -> str | None:
    """检测心理危机等高风险主题，返回应附加的提示文案。"""
    lowered = text.lower()
    for topic in HIGH_RISK_TOPICS:
        if topic.strip().lower() in lowered:
            return MENTAL_HEALTH_NOTICE
    return None


def moderate_output(
    text: str,
    *,
    scenario: LLMScenario = LLMScenario.CHAT,
    require_disclaimer: bool = True,
) -> ModerationResult:
    """输出侧审核：医疗合规校验。

    分三种处置：
    - **block**：绝对化诊断、疗效承诺、具体给药方案。这些必须拦下来换成降级话术。
    - **warn**：高风险主题命中。放行但附加危机干预提示——拦下来反而更危险，
      因为用户此时最需要的是出口而不是空白回复。
    - **missing_disclaimer**：缺就医提示。记为 warn，由调用方决定是否补上。

    ``scenario`` 影响严格程度：面向患者的 chat 最严；
    面向医生/药师的专业场景（report_interpret）允许更技术性的表述，
    因为使用者本身具备判断能力，但仍不允许绝对化疗效承诺。
    """
    issues: list[ModerationIssue] = []
    if not isinstance(text, str) or not text.strip():
        return ModerationResult(passed=True, issues=[], rule_hit_count=0)

    appended_notice: str | None = None

    # 1. 高风险主题优先处理（心理危机 / 急症）
    notice = _detect_high_risk_topic(text)
    if notice is not None:
        appended_notice = notice
        issues.append(
            ModerationIssue(
                category="high_risk",
                severity="warn",
                message="命中高风险主题，已附加危机干预提示",
            )
        )

    # 2. 绝对化诊断与疗效承诺：一律拦截，不分场景
    for pattern in ABSOLUTE_DIAGNOSIS_PATTERNS:
        match = pattern.search(text)
        if match:
            issues.append(
                ModerationIssue(
                    category="absolute_diagnosis",
                    severity="block",
                    matched=match.group(0)[:60],
                    message="输出包含绝对化诊断或疗效承诺",
                )
            )

    # 3. 具体给药方案：面向患者时拦截，面向专业人员时仅告警
    dosage_severity = "block" if scenario == LLMScenario.CHAT else "warn"
    for pattern in DOSAGE_PRESCRIPTION_PATTERNS:
        match = pattern.search(text)
        if match:
            issues.append(
                ModerationIssue(
                    category="dosage_prescription",
                    severity=dosage_severity,
                    matched=match.group(0)[:60],
                    message="输出包含具体给药方案",
                )
            )
            break

    # 4. 免责声明缺失检查。报告解读等专业场景不强制，
    #    因为解读结果会由医生复核后才呈现给患者。
    if require_disclaimer and scenario == LLMScenario.CHAT:
        if not any(keyword in text for keyword in REQUIRED_DISCLAIMER_KEYWORDS):
            issues.append(
                ModerationIssue(
                    category="missing_disclaimer",
                    severity="warn",
                    message="输出缺少就医提示",
                )
            )

    blocked = any(issue.severity == "block" for issue in issues)
    return ModerationResult(
        passed=not blocked,
        blocked=blocked,
        issues=issues,
        appended_notice=appended_notice,
        rule_hit_count=len(issues),
    )


class ContentModerator:
    """内容审核器。

    保留类形式而非纯函数，是为了后续能挂「模型兜底」通道：
    规则拿不准时调小模型二次判断。当前阶段只实现规则层，
    接口形态已经为模型层预留。
    """

    def moderate_input(self, text: str) -> ModerationResult:
        return moderate_input(text)

    def moderate_output(
        self,
        text: str,
        *,
        scenario: LLMScenario = LLMScenario.CHAT,
        require_disclaimer: bool = True,
    ) -> ModerationResult:
        return moderate_output(
            text, scenario=scenario, require_disclaimer=require_disclaimer
        )


# 全局单例。审核器无状态，可安全共享。
moderator = ContentModerator()


__all__ = [
    "ModerationIssue",
    "ModerationResult",
    "ContentModerator",
    "moderator",
    "moderate_input",
    "moderate_output",
    "MENTAL_HEALTH_NOTICE",
    "EMERGENCY_NOTICE",
    "MAX_INPUT_CHARACTERS",
]
