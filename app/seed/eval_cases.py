"""评测题库种子：按 IMCS 实际知识库内容构造可命中的评测题。

**为什么不照搬旧系统的 100 题**：旧题库里大量题目的 ``expected_name`` 指向
``布洛芬``/``阿莫西林`` 这类**药物记录**和「常见感冒对症药物_知识库测试资料」
这类 PDF 分片，而 IMCS 的知识库（app/seed/knowledge.py）只有 27 篇文档、
``doc_type`` 仅 ``condition`` 与 ``guideline`` 两种，既没有独立药物记录、
也没有那批 PDF。原样迁移的结果是这些题永远命中不到，Top1/Recall@3 被
不存在的目标拖成 0，指标不再反映检索质量，只反映「题库和数据对不上」。
评测集必须锚定真实存在的资料，这里据 27 篇文档逐篇构造可命中题目。

**字段口径**（与 models/evaluation.py 一致）：
- ``expected_name`` = KnowledgeChunk.title = KnowledgeDocument.title，逐字一致，
  否则命中判定（title 相等）恒假；
- ``expected_type`` ∈ {condition, guideline}，与 doc_type 同域；
- ``alternative_names`` 给简称/带科室前缀的写法，容忍表述差异；
- ``expected_refusal=True`` 的题指向知识库确实没有的内容（处方剂量、挂号政策、
  医保报销、具体医生排班），用于反向度量「检索该空时是否真空」，
  防止系统靠瞎编把缺口掩盖成能力。

幂等：按 question 去重，重复执行只补新增、不重复插入。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.evaluation import EvalCase

logger = logging.getLogger(__name__)


def _case(
    question: str,
    expected_name: str,
    expected_type: str,
    category: str,
    *,
    alternatives: Sequence[str] = (),
    answer_keywords: Sequence[str] = (),
    refusal: bool = False,
) -> dict:
    """一条评测题的字典构造。

    拒答题（refusal=True）刻意把 answer_keywords / citation_names 留空、
    expected_name 用占位符：旧质量层对这类题的判定是
    「检索结果为空 = 正确拒答」，不看关键词也不看引用名，因此这三项必须为空，
    否则占位符会被当成期望引用而永远判错。
    """
    if refusal:
        return {
            "question": question,
            "expected_name": expected_name,
            "expected_type": expected_type,
            "category": category,
            "alternative_names": [],
            "answer_keywords": [],
            "citation_names": [],
            "expected_refusal": True,
        }
    return {
        "question": question,
        "expected_name": expected_name,
        "expected_type": expected_type,
        "category": category,
        "alternative_names": list(alternatives),
        "answer_keywords": list(answer_keywords),
        "citation_names": list(alternatives) or [expected_name],
        "expected_refusal": False,
    }


# 概览文档的 doc_type 是 condition；专病健康教育与睡眠提示是 guideline。
EVAL_CASES: tuple[dict, ...] = (
    # ---------- 专科概览（condition）----------
    _case("鼻塞流涕打喷嚏咳嗽低热这些感冒样症状属于哪个科室的常见问题？",
          "呼吸内科常见病概览", "condition", "专科检索",
          alternatives=["呼吸内科"], answer_keywords=["上呼吸道感染", "呼吸"]),
    _case("突发进行性呼吸困难属于什么级别的情况？",
          "呼吸内科常见病概览", "condition", "紧急警示",
          answer_keywords=["紧急", "呼吸困难"]),
    _case("血压异常活动后胸闷应该看哪个专科概览？",
          "心血管内科常见病概览", "condition", "专科检索",
          alternatives=["心血管内科"], answer_keywords=["心血管"]),
    _case("咯血和嘴唇发紫时需要怎么处理？",
          "呼吸内科常见病概览", "condition", "紧急警示",
          answer_keywords=["紧急", "就医"]),
    _case("腹痛腹泻便血这些消化问题应先看哪份概览？",
          "消化内科常见病概览", "condition", "专科检索",
          alternatives=["消化内科"]),
    _case("头晕肢体麻木头痛属于哪个专科资料？",
          "神经内科常见病概览", "condition", "专科检索",
          alternatives=["神经内科"]),
    _case("口渴增多体重变化怕冷怕热应查哪个专科概览？",
          "内分泌科常见病概览", "condition", "专科检索",
          alternatives=["内分泌科"]),
    _case("右下腹疼痛恶心发热要看哪个外科概览？",
          "普通外科常见病概览", "condition", "专科检索",
          alternatives=["普通外科"]),
    _case("外伤后肿胀畸形不能负重应查询哪个专科？",
          "骨科常见病概览", "condition", "专科检索",
          alternatives=["骨科"]),
    _case("尿频尿急尿痛和下腹不适属于哪个专科资料？",
          "泌尿外科常见病概览", "condition", "专科检索",
          alternatives=["泌尿外科"]),
    _case("月经不规律痛经或异常出血应看哪个专科概览？",
          "妇产科常见病概览", "condition", "专科检索",
          alternatives=["妇产科"]),
    _case("儿童反复喘息发热或腹泻应先查哪份专科资料？",
          "儿科常见病概览", "condition", "专科检索",
          alternatives=["儿科"]),
    _case("视物模糊眼痛眼红应该查哪个专科？",
          "眼科常见病概览", "condition", "专科检索",
          alternatives=["眼科"]),
    _case("耳痛听力下降鼻出血或咽痛属于哪个专科概览？",
          "耳鼻喉科常见病概览", "condition", "专科检索",
          alternatives=["耳鼻喉科"]),
    _case("反复皮疹瘙痒水疱应看哪个专科概览？",
          "皮肤科常见病概览", "condition", "专科检索",
          alternatives=["皮肤科"]),
    _case("持续焦虑情绪低落或睡眠受影响应看哪个专科概览？",
          "精神心理科常见问题概览", "condition", "专科检索",
          alternatives=["精神心理科"]),
    _case("水肿尿量变化和肾功能异常应查看哪个专科概览？",
          "肾内科常见病概览", "condition", "专科检索",
          alternatives=["肾内科"]),
    _case("关节肿痛晨僵和免疫性问题该查哪份资料？",
          "风湿免疫科常见病概览", "condition", "专科检索",
          alternatives=["风湿免疫科"]),
    _case("体检报告出现异常指标时应先查哪份解读资料？",
          "健康体检与常见指标解读概览", "condition", "专科检索",
          alternatives=["健康体检"]),

    # ---------- 急症识别（condition）----------
    _case("突发胸痛昏迷或严重外伤需要查询哪份急症资料？",
          "急诊科常见急症识别概览", "condition", "紧急警示",
          alternatives=["急诊科"], answer_keywords=["急症", "立即"]),
    _case("哪些情况需要立即寻求紧急医疗帮助？",
          "急诊科常见急症识别概览", "condition", "紧急警示"),

    # ---------- 专病健康教育（guideline）----------
    _case("高血压通常有哪些健康教育和就医警示？",
          "高血压：健康教育与就医警示", "guideline", "重点健康教育",
          alternatives=["高血压"]),
    _case("血压高但没有明显症状时该看哪份资料？",
          "高血压：健康教育与就医警示", "guideline", "重点健康教育"),
    _case("2 型糖尿病的健康教育和风险提示是什么？",
          "2 型糖尿病：健康教育与就医警示", "guideline", "重点健康教育",
          alternatives=["2型糖尿病", "糖尿病"]),
    _case("血糖异常口渴和多尿时应查询哪份资料？",
          "2 型糖尿病：健康教育与就医警示", "guideline", "重点健康教育"),
    _case("哮喘反复喘息和夜间咳嗽的健康提示是什么？",
          "哮喘：健康教育与就医警示", "guideline", "重点健康教育",
          alternatives=["哮喘"]),
    _case("运动后胸闷喘鸣时应查询哪份哮喘资料？",
          "哮喘：健康教育与就医警示", "guideline", "重点健康教育"),
    _case("胃食管反流的健康教育与就医警示有哪些？",
          "胃食管反流：健康教育与就医警示", "guideline", "重点健康教育",
          alternatives=["胃食管反流"]),
    _case("吞咽困难或反酸反复加重时应看哪份资料？",
          "胃食管反流：健康教育与就医警示", "guideline", "重点健康教育"),
    _case("尿路感染的健康教育与就医警示是什么？",
          "尿路感染：健康教育与就医警示", "guideline", "重点健康教育",
          alternatives=["尿路感染"]),
    _case("尿频尿急尿痛伴发热时应查询哪份资料？",
          "尿路感染：健康教育与就医警示", "guideline", "重点健康教育"),
    _case("焦虑障碍常见表现和求助信号是什么？",
          "焦虑障碍：健康教育与求助信号", "guideline", "重点健康教育",
          alternatives=["焦虑障碍"]),
    _case("焦虑导致学习工作受影响时应看哪份资料？",
          "焦虑障碍：健康教育与求助信号", "guideline", "重点健康教育"),
    _case("失眠的健康教育和求助信号有哪些？",
          "失眠：健康教育与求助信号", "guideline", "重点健康教育",
          alternatives=["失眠"]),
    _case("入睡困难夜间醒来和白天困倦时应查询哪份资料？",
          "失眠：健康教育与求助信号", "guideline", "重点健康教育"),
    _case("偏头痛的健康教育与就医警示是什么？",
          "偏头痛：健康教育与就医警示", "guideline", "重点健康教育",
          alternatives=["偏头痛"]),
    _case("头痛模式突然改变时应查看哪份资料？",
          "偏头痛：健康教育与就医警示", "guideline", "重点健康教育"),
    _case("睡眠不足时有哪些基础健康建议？",
          "睡眠健康提示", "guideline", "生活方式",
          alternatives=["睡眠"]),

    # ---------- 拒答题（知识库确实不覆盖，检索为空才是正确行为）----------
    _case("阿司匹林肠溶片每天应该吃几毫克？", "（知识库未收录具体处方剂量）",
          "guideline", "拒答校验", refusal=True),
    _case("在这个平台怎么预约核磁共振的挂号？", "（知识库不含挂号操作指引）",
          "guideline", "拒答校验", refusal=True),
    _case("职工医保门诊报销比例是多少？", "（知识库不含医保报销政策）",
          "guideline", "拒答校验", refusal=True),
    _case("王建国医生下周哪天出诊？", "（知识库不含医生排班查询）",
          "guideline", "拒答校验", refusal=True),
)


async def seed_eval_cases(db: AsyncSession) -> dict[str, int]:
    """写入评测题库，按 question 幂等去重。

    遵守 seed_all 的事务约定：**只 flush 不 commit**，由调用方
    （run_seed / 容器初始化）统一提交，避免种子写入被拆成多个事务。
    """
    existing = {
        row[0]
        for row in (await db.execute(select(EvalCase.question))).all()
    }
    created = 0
    for spec in EVAL_CASES:
        if spec["question"] in existing:
            continue
        db.add(EvalCase(**spec))
        created += 1
    await db.flush()
    result = {
        "eval_cases_created": created,
        "eval_cases_skipped": len(EVAL_CASES) - created,
    }
    if created:
        logger.info("评测题库写入 %s 条", created)
    return result
