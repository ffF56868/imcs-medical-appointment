"""Seed 100 repeatable custom retrieval-evaluation cases.

Run inside the API container after a rebuild:
    docker compose exec api python scripts/seed_custom_evaluation_cases.py

The script only adds questions that do not already exist, so it is safe to
run again. It deliberately does not delete a user's existing custom cases.
"""

from __future__ import annotations

import json

from sqlmodel import Session, select

import sys
from pathlib import Path

# Allow `python scripts/seed_custom_evaluation_cases.py` inside the container.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database import engine
from app.legacy.models import RAGEvaluationCase


def case(
    question: str,
    expected_name: str,
    expected_type: str,
    category: str,
) -> dict[str, str]:
    return {
        "question": question,
        "expected_name": expected_name,
        "expected_type": expected_type,
        "category": category,
    }


CUSTOM_EVALUATION_CASES = (
    # 1-18: drug retrieval
    case("布洛芬主要能缓解哪些不适？", "布洛芬", "drug", "药物检索"),
    case("发热和疼痛时，知识库里有布洛芬的什么说明？", "布洛芬", "drug", "药物检索"),
    case("布洛芬使用前要注意什么？", "布洛芬", "drug", "药物检索"),
    case("对乙酰氨基酚通常用于什么情况？", "对乙酰氨基酚", "drug", "药物检索"),
    case("想了解退热止痛药对乙酰氨基酚的作用。", "对乙酰氨基酚", "drug", "药物检索"),
    case("对乙酰氨基酚的使用说明在哪里？", "对乙酰氨基酚", "drug", "药物检索"),
    case("鼻痒、连续打喷嚏时可查哪种抗过敏药？", "氯雷他定", "drug", "药物检索"),
    case("氯雷他定能缓解流清鼻涕吗？", "氯雷他定", "drug", "药物检索"),
    case("氯雷他定的用药注意事项是什么？", "氯雷他定", "drug", "药物检索"),
    case("阿莫西林和普通感冒之间有什么用药提示？", "阿莫西林", "drug", "药物检索"),
    case("阿莫西林属于抗生素吗，资料里如何说明？", "阿莫西林", "drug", "药物检索"),
    case("使用阿莫西林前应该看哪些说明？", "阿莫西林", "drug", "药物检索"),
    case("流感相关的奥司他韦资料是什么？", "奥司他韦", "drug", "药物检索"),
    case("奥司他韦主要用于哪一类感染？", "奥司他韦", "drug", "药物检索"),
    case("奥司他韦的使用说明在哪里查看？", "奥司他韦", "drug", "药物检索"),
    case("急性腹泻时蒙脱石散有什么作用？", "蒙脱石散", "drug", "药物检索"),
    case("蒙脱石散适合查找什么消化道症状资料？", "蒙脱石散", "drug", "药物检索"),
    case("蒙脱石散使用时需要阅读什么说明？", "蒙脱石散", "drug", "药物检索"),
    # 19-33: structured condition retrieval
    case("鼻塞、流涕和打喷嚏可能对应普通感冒的什么资料？", "普通感冒", "condition", "病症检索"),
    case("普通感冒常见症状和处理建议是什么？", "普通感冒", "condition", "病症检索"),
    case("感冒样症状持续不缓解时该查看哪条病症资料？", "普通感冒", "condition", "病症检索"),
    case("高热、肌肉酸痛和乏力更像哪种流感相关资料？", "流行性感冒", "condition", "病症检索"),
    case("流行性感冒的症状与处理建议是什么？", "流行性感冒", "condition", "病症检索"),
    case("发热伴全身酸痛时，知识库中应优先检索哪个病症？", "流行性感冒", "condition", "病症检索"),
    case("鼻痒、喷嚏、清水样鼻涕常见于哪种病症？", "过敏性鼻炎", "condition", "病症检索"),
    case("过敏性鼻炎有哪些常见表现？", "过敏性鼻炎", "condition", "病症检索"),
    case("接触过敏原后反复鼻塞应查什么资料？", "过敏性鼻炎", "condition", "病症检索"),
    case("饭后反酸、烧心、平躺后加重可能是什么问题？", "胃食管反流", "condition", "病症检索"),
    case("胃食管反流的症状与处理建议是什么？", "胃食管反流", "condition", "病症检索"),
    case("夜间咳嗽同时反酸时应该检索哪种病症？", "胃食管反流", "condition", "病症检索"),
    case("反复搏动性头痛伴畏光应查哪种病症资料？", "偏头痛", "condition", "病症检索"),
    case("偏头痛通常有哪些伴随症状？", "偏头痛", "condition", "病症检索"),
    case("恶心、畏光、单侧头部跳痛时检索什么病症？", "偏头痛", "condition", "病症检索"),
    # 34-39: safety and lifestyle documents
    case("哪些症状出现后不应继续等待，需要及时就医？", "需要及时就医的警示信号", "document", "紧急警示"),
    case("持续胸痛、呼吸困难时有什么紧急警示？", "需要及时就医的警示信号", "document", "紧急警示"),
    case("突然意识改变或单侧肢体无力时应查看哪份资料？", "需要及时就医的警示信号", "document", "紧急警示"),
    case("睡眠不足时有哪些基础健康建议？", "睡眠健康提示", "document", "生活方式"),
    case("如何建立更规律的睡眠节律？", "睡眠健康提示", "document", "生活方式"),
    case("白天总是困倦时，睡眠健康资料建议关注什么？", "睡眠健康提示", "document", "生活方式"),
    # 40-71: two questions for each original specialty overview
    case("反复咳嗽、喘息和气促应了解哪个内科资料？", "呼吸内科常见病概览", "document", "专科知识"),
    case("呼吸内科常见病概览覆盖哪些呼吸道风险提示？", "呼吸内科常见病概览", "document", "专科知识"),
    case("血压异常、活动后胸闷应查询哪个专科概览？", "心血管内科常见病概览", "document", "专科知识"),
    case("心血管内科资料中有哪些需要重视的胸部不适？", "心血管内科常见病概览", "document", "专科知识"),
    case("腹痛、腹泻、便血等消化问题应先看哪份概览？", "消化内科常见病概览", "document", "专科知识"),
    case("消化内科常见病概览如何提示报警症状？", "消化内科常见病概览", "document", "专科知识"),
    case("头晕、肢体麻木、头痛等问题属于哪个专科概览？", "神经内科常见病概览", "document", "专科知识"),
    case("神经内科常见病概览中哪些症状需要及时评估？", "神经内科常见病概览", "document", "专科知识"),
    case("口渴增多、体重变化、怕冷怕热应查哪个专科资料？", "内分泌科常见病概览", "document", "专科知识"),
    case("内分泌科常见病概览与血糖、甲状腺问题有关吗？", "内分泌科常见病概览", "document", "专科知识"),
    case("右下腹疼痛、恶心发热时应看哪个外科概览？", "普通外科常见病概览", "document", "专科知识"),
    case("普通外科常见病概览如何提示急腹症风险？", "普通外科常见病概览", "document", "专科知识"),
    case("外伤后肿胀、畸形、不能负重应查询哪个专科？", "骨科常见病概览", "document", "专科知识"),
    case("骨科常见病概览对骨折和软组织损伤有什么提示？", "骨科常见病概览", "document", "专科知识"),
    case("尿频、尿急、尿痛和下腹不适属于哪个专科资料？", "泌尿外科常见病概览", "document", "专科知识"),
    case("泌尿外科常见病概览如何提示尿路问题就医？", "泌尿外科常见病概览", "document", "专科知识"),
    case("月经不规律、痛经或异常出血应看哪个专科概览？", "妇产科常见病概览", "document", "专科知识"),
    case("妇产科常见病概览中哪些症状不适合自行处理？", "妇产科常见病概览", "document", "专科知识"),
    case("儿童反复喘息、发热或腹泻应先查哪份专科资料？", "儿科常见病概览", "document", "专科知识"),
    case("儿科常见病概览对儿童便秘和喘息有什么提示？", "儿科常见病概览", "document", "专科知识"),
    case("视物模糊、眼痛、眼红等症状应该查哪个专科？", "眼科常见病概览", "document", "专科知识"),
    case("眼科常见病概览中有哪些就医警示？", "眼科常见病概览", "document", "专科知识"),
    case("耳痛、听力下降、鼻出血或咽痛属于哪个专科概览？", "耳鼻喉科常见病概览", "document", "专科知识"),
    case("耳鼻喉科常见病概览如何提示持续咽喉不适？", "耳鼻喉科常见病概览", "document", "专科知识"),
    case("牙痛、牙龈出血、口腔溃疡应查询哪个专科资料？", "口腔科常见病概览", "document", "专科知识"),
    case("口腔科常见病概览中反复口腔溃疡要注意什么？", "口腔科常见病概览", "document", "专科知识"),
    case("反复皮疹、瘙痒、水疱应看哪个专科概览？", "皮肤科常见病概览", "document", "专科知识"),
    case("皮肤科常见病概览对带状疱疹有什么提示？", "皮肤科常见病概览", "document", "专科知识"),
    case("突发胸痛、昏迷、严重外伤需要查询哪份急症资料？", "急诊科常见急症识别概览", "document", "专科知识"),
    case("急诊科常见急症识别概览包括哪些立即就医情形？", "急诊科常见急症识别概览", "document", "专科知识"),
    case("想了解中医健康咨询中的禁忌与就医边界，应查哪份资料？", "中医科常见就诊问题概览", "document", "专科知识"),
    case("中医科常见就诊问题概览对中药和急症有什么提醒？", "中医科常见就诊问题概览", "document", "专科知识"),
    # 72-91: expanded specialty and focused health education documents
    case("水肿、尿量变化和肾功能异常应查看哪个专科概览？", "肾内科常见病概览", "document", "扩展专科"),
    case("关节肿痛、晨僵和免疫性问题该查哪份资料？", "风湿免疫科常见病概览", "document", "扩展专科"),
    case("反复出血、瘀斑或贫血相关问题属于哪个专科概览？", "血液科常见病概览", "document", "扩展专科"),
    case("反复发热、传染性疾病风险应查询哪个专科资料？", "感染科常见感染性疾病概览", "document", "扩展专科"),
    case("新出现肿块、异常出血或长期不愈溃疡应看哪份警示资料？", "肿瘤科常见就诊警示概览", "document", "扩展专科"),
    case("持续焦虑、情绪低落或睡眠受影响应看哪个专科概览？", "精神心理科常见问题概览", "document", "扩展专科"),
    case("运动损伤恢复和功能训练应查询哪个医学科资料？", "康复医学科常见康复问题概览", "document", "扩展专科"),
    case("体重管理、饮食结构和营养风险应看哪份资料？", "临床营养与体重管理概览", "document", "扩展专科"),
    case("老年人多病共存和用药安全应查询哪个专科概览？", "老年医学常见健康问题概览", "document", "扩展专科"),
    case("体检报告出现异常指标时应先查哪份解读资料？", "健康体检与常见指标解读概览", "document", "扩展专科"),
    case("高血压通常有哪些健康教育和就医警示？", "高血压：健康教育与就医警示", "document", "重点健康教育"),
    case("血压高但没有明显症状时该看哪份资料？", "高血压：健康教育与就医警示", "document", "重点健康教育"),
    case("2 型糖尿病的健康教育和风险提示是什么？", "2 型糖尿病：健康教育与就医警示", "document", "重点健康教育"),
    case("血糖异常、口渴和多尿时应查询哪份资料？", "2 型糖尿病：健康教育与就医警示", "document", "重点健康教育"),
    case("哮喘反复喘息和夜间咳嗽的健康提示是什么？", "哮喘：健康教育与就医警示", "document", "重点健康教育"),
    case("运动后胸闷、喘鸣时应查询哪份哮喘资料？", "哮喘：健康教育与就医警示", "document", "重点健康教育"),
    case("胃食管反流的健康教育与就医警示有哪些？", "胃食管反流：健康教育与就医警示", "document", "重点健康教育"),
    case("吞咽困难或反酸反复加重时应看哪份胃食管反流资料？", "胃食管反流：健康教育与就医警示", "document", "重点健康教育"),
    case("尿路感染的健康教育与就医警示是什么？", "尿路感染：健康教育与就医警示", "document", "重点健康教育"),
    case("尿频尿急尿痛伴发热时应查询哪份资料？", "尿路感染：健康教育与就医警示", "document", "重点健康教育"),
    # 92-100: mental health, neurology, and uploaded-PDF retrieval
    case("焦虑障碍常见表现和求助信号是什么？", "焦虑障碍：健康教育与求助信号", "document", "重点健康教育"),
    case("焦虑导致学习工作受影响时应看哪份资料？", "焦虑障碍：健康教育与求助信号", "document", "重点健康教育"),
    case("失眠的健康教育和求助信号有哪些？", "失眠：健康教育与求助信号", "document", "重点健康教育"),
    case("入睡困难、夜间醒来和白天困倦时应查询哪份资料？", "失眠：健康教育与求助信号", "document", "重点健康教育"),
    case("偏头痛的健康教育与就医警示是什么？", "偏头痛：健康教育与就医警示", "document", "重点健康教育"),
    case("头痛模式突然改变时应查看哪份偏头痛资料？", "偏头痛：健康教育与就医警示", "document", "重点健康教育"),
    case("常见感冒对症药物 PDF 第 1 页主要介绍什么？", "常见感冒对症药物_知识库测试资料（第1页）", "document", "PDF 检索"),
    case("感冒对症药物 PDF 的第一页有哪些用药资料？", "常见感冒对症药物_知识库测试资料（第1页）", "document", "PDF 检索"),
    case("常见感冒对症药物 PDF 第 2 页主要介绍什么？", "常见感冒对症药物_知识库测试资料（第2页）", "document", "PDF 检索"),
)


def main() -> None:
    if len(CUSTOM_EVALUATION_CASES) != 100:
        raise RuntimeError("Custom evaluation set must contain exactly 100 cases.")

    questions = [item["question"] for item in CUSTOM_EVALUATION_CASES]
    if len(set(questions)) != len(questions):
        raise RuntimeError("Custom evaluation questions must be unique.")

    with Session(engine) as session:
        existing_questions = {
            record.question
            for record in session.exec(select(RAGEvaluationCase)).all()
        }
        created_count = 0
        skipped_count = 0
        for item in CUSTOM_EVALUATION_CASES:
            if item["question"] in existing_questions:
                skipped_count += 1
                continue
            session.add(
                RAGEvaluationCase(
                    **item,
                    alternative_names_json=json.dumps([], ensure_ascii=False),
                    answer_keywords_json=json.dumps([], ensure_ascii=False),
                    citation_names_json=json.dumps([], ensure_ascii=False),
                )
            )
            created_count += 1
        session.commit()

    print(
        "Custom evaluation cases ready: "
        f"created={created_count}, skipped_existing={skipped_count}, total_batch=100"
    )


if __name__ == "__main__":
    main()
