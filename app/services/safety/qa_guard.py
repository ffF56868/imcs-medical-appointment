"""健康咨询的安全守卫：急症与心理危机的短路拦截（阶段 7 批次 0）。

**为什么单独建一个模块而不是写在路由里**：拦截规则是医疗安全关键信息，
必须能一眼看全、便于审计和扩充，和分诊的 ``EMERGENCY_KEYWORDS`` 同一个道理。
放在服务层也让它可以被单元测试直接覆盖，不必起 HTTP。

两类守卫，取向不同：

1. **躯体急症**——直接复用分诊的 ``_match_emergency`` / ``EMERGENCY_KEYWORDS``
   （``app/services/triage_workflow.py``）。刻意不另抄一份词表：同一句「突然胸痛」
   在分诊里被判急症、在咨询里却正常走 LLM 生成，是最危险的不一致。

2. **心理危机**——分诊词表**不覆盖**自伤/自杀表达（它面向躯体主诉），
   而健康咨询是开放问答，患者完全可能在这里说出「不想活了」。
   这类输入绝不能交给 LLM 自由生成，必须先给求助通道。词表与话术见下。

两类命中都**不调 LLM**，直接返回固定就医指引：既省一次调用，
更重要的是保证输出可预测——兜底话术经过审核，模型生成不保证。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.triage_workflow import _match_emergency

# 心理危机词表。取向与急症词表一致：宁可误报。
# 「活着没意思」既可能是抑郁表达也可能是一句抱怨，但漏掉一次真实危机的
# 代价不可接受，误报的代价只是多给一次求助通道——安全不对称。
CRISIS_KEYWORDS: tuple[str, ...] = (
    "不想活",
    "不想活着",
    "活不下去",
    "活着没意思",
    "活着没有意义",
    "了结自己",
    "结束生命",
    "结束自己",
    "自杀",
    "自尽",
    "轻生",
    "自我了断",
    "自残",
    "割腕",
    "伤害自己",
    "想死",
    "死了算了",
    "一了百了",
    "跳楼",
    "跳河",
    "烧炭",
    "安眠药自杀",
)

_CRISIS_PATTERN = re.compile("|".join(re.escape(k) for k in CRISIS_KEYWORDS))

# 12356：全国统一心理援助热线，国家卫生健康委协调工业和信息化部设置
# （国卫医政函〔2024〕259 号），2025-05-01 起全国 31 省区市统一使用。
# 12355 是共青团中央的青少年服务台（9:00-19:00），作为补充渠道给出。
PSYCH_HOTLINE = "12356"
YOUTH_HOTLINE = "12355"
EMERGENCY_NUMBER = "120"

# 心理危机话术。刻意不做情绪评判、不说教、不追问细节，
# 只给「此刻能做的三件事」：立即求助的通道、陪伴建议、紧急兜底。
# 话术里不出现「你不应」「你想太多」这类评判性表达。
CRISIS_RESPONSE = (
    "我很担心你现在的状态，谢谢你愿意说出来。\n\n"
    "请先做一件事：拨打 **全国心理援助热线 {psych}**（24 小时免费，"
    "由精神科医护人员与心理咨询师接听），把你现在的感受告诉他们。\n"
    "如果你是未成年人，也可以拨打 **{youth} 青少年服务台**。\n\n"
    "如果此刻身边有人，请让 TA 陪着你，不要独处；"
    "把剪刀、药品等可能造成伤害的物品放到自己不易拿到的地方。\n\n"
    "如果你已经有明确的自伤计划，或感觉自己无法控制，"
    "请立即拨打 **{emergency}** 或前往最近的医院急诊。\n\n"
    "你现在的痛苦是真实的，也是可以缓解的。"
    "这类情况需要专业医生或心理咨询师的当面帮助，"
    "我无法替代他们，但请相信求助是有效的。"
).format(psych=PSYCH_HOTLINE, youth=YOUTH_HOTLINE, emergency=EMERGENCY_NUMBER)

# 躯体急症话术。与分诊急诊分支同一口径：先给行动指令，不做疾病解释。
URGENT_RESPONSE = (
    "你描述的情况可能属于急症，请**立即拨打 {emergency}** 或前往最近医院的急诊科，"
    "不要等待门诊预约，也不要自行服药观察。\n\n"
    "命中的警示表现：{matched}。\n\n"
    "在等待急救时：保持安静平卧、解开领口，如有既往用药（硝酸甘油等）"
    "按医生此前的医嘱使用；身边有人请让其陪同并告知医生发病时间。\n\n"
    "我不能替代急诊评估，此处不提供进一步的家庭处理建议。"
)

# 检索零结果话术。医疗安全取向：不给通用建议、不让模型凭常识发挥，
# 直接说明知识库未收录并引导线下就诊——宁可少答，不可错答。
ZERO_RESULT_RESPONSE = (
    "知识库中暂未收录与这个问题直接相关的权威资料，我不能凭推测给你答案。\n\n"
    "建议：\n"
    "1. 换个说法再问一次（比如把「胸闷」说成「胸口压迫感、喘不上气」）；\n"
    "2. 或前往医院相应科室门诊，由医生结合检查判断；\n"
    "3. 如症状突然加重或出现急重表现，请立即拨打 {emergency}。"
).format(emergency=EMERGENCY_NUMBER)


@dataclass(slots=True)
class GuardVerdict:
    """守卫判定结果。

    ``blocked=True`` 表示短路：调用方应直接把 ``response`` 当回答发出，
    不做检索、不调 LLM。``kind`` 区分急症与心理危机——两者的统计口径不同
    （急症拦截率是分诊质量指标，危机拦截率是心理安全指标），混在一起
    会让两个指标都失真。
    """

    blocked: bool = False
    kind: str | None = None  # "emergency" / "crisis"
    matched: list[str] = field(default_factory=list)
    response: str = ""


def match_crisis(text: str) -> list[str]:
    """返回命中的心理危机关键词（去重保序）。"""
    found: list[str] = []
    for kw in CRISIS_KEYWORDS:
        if kw in text and kw not in found:
            found.append(kw)
    return found


def guard_question(text: str) -> GuardVerdict:
    """对一句咨询做安全守卫判定。

    **心理危机优先于躯体急症**：两者同时命中时（如「吃了药不想活了」），
    先给心理求助通道。理由是躯体急症话术会让用户去急诊，
    而危机状态下独自在路上反而更危险，需要的是先接上人工支持。
    """
    if not text or not text.strip():
        return GuardVerdict()

    crisis = match_crisis(text)
    if crisis:
        return GuardVerdict(
            blocked=True, kind="crisis", matched=crisis, response=CRISIS_RESPONSE
        )

    # 复用分诊的急症判定，保证同一句话在分诊与咨询里结论一致
    matched = _match_emergency(text)
    if matched:
        return GuardVerdict(
            blocked=True,
            kind="emergency",
            matched=matched,
            response=URGENT_RESPONSE.format(
                emergency=EMERGENCY_NUMBER, matched="、".join(matched)
            ),
        )

    return GuardVerdict()


__all__ = [
    "CRISIS_KEYWORDS",
    "CRISIS_RESPONSE",
    "EMERGENCY_NUMBER",
    "GuardVerdict",
    "PSYCH_HOTLINE",
    "URGENT_RESPONSE",
    "YOUTH_HOTLINE",
    "ZERO_RESULT_RESPONSE",
    "guard_question",
    "match_crisis",
]
