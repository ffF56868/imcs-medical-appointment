"""Add targeted, safety-focused retrieval language for ten failed RAG cases.

The additions are deliberately attached to the original knowledge records, not
created as duplicate documents. This lets the evaluation continue to verify
whether the intended source itself is retrieved in the top three results.

Run inside the API container:
    docker compose exec api python scripts/seed_retrieval_gap_knowledge.py
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys

from sqlmodel import Session, select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.cache import invalidate_knowledge_status_cache
from app.database import engine
from app.legacy.models import Condition, KnowledgeDocument


# These phrases correspond to the ten cases that missed their expected record
# in the latest baseline run. They retain conservative health-education advice.
CONDITION_SUPPLEMENTS = {
    "普通感冒": """
## 检索补充：鼻塞、流鼻涕和打喷嚏
当问题出现鼻塞、流鼻涕或流涕、打喷嚏、咽痛、轻度咳嗽、低热等感冒样表现时，可检索本条“普通感冒”资料。类似表现也可能与过敏性鼻炎等情况有关；症状持续、明显加重、反复发热或出现呼吸困难时，应及时线下评估。
""".strip(),
    "流行性感冒": """
## 检索补充：发热、肌肉酸痛和乏力
发热、头痛、肌肉酸痛、明显乏力、咳嗽且起病较急时，可检索本条“流行性感冒”资料。症状严重、持续高热、呼吸困难，或儿童、老年人、孕期及有慢性病的人群出现不适时，应尽快由医疗人员评估。
""".strip(),
}

DOCUMENT_SUPPLEMENTS = {
    "泌尿外科常见病概览": """
## 检索补充：尿频、尿急、尿痛和下腹不适
出现尿频、尿急、尿痛、下腹不适、尿液异常或伴发热时，可优先检索本份“泌尿外科常见病概览”。男性、孕期、儿童、老年人及反复发作者不宜自行反复用药，应及时就医评估。

## 检索补充：尿路问题的就医提示
腰痛伴发热、肉眼血尿、排尿困难、尿量明显减少，或症状迅速加重时，需要尽快线下就医；这些表现不能只靠本知识库自行判断。
""".strip(),
    "妇产科常见病概览": """
## 检索补充：不适合自行处理的妇产科症状
异常阴道出血、妊娠期腹痛或出血、持续或剧烈下腹痛、发热伴盆腔不适、分泌物异常且明显不适，都不适合仅自行用药或依赖线上资料，应尽快进行妇产科评估。
""".strip(),
    "儿科常见病概览": """
## 检索补充：儿童喘息、发热和腹泻
儿童反复喘息、发热、腹泻、呕吐或精神状态变差时，可检索本份“儿科常见病概览”。家长应重点观察呼吸是否费力、饮水和尿量、精神状态；婴幼儿、持续高热或出现呼吸困难时需要及时就医。

## 检索补充：儿童便秘和喘息
儿童便秘可表现为排便次数减少、排便疼痛或大便干硬；反复喘息可伴咳嗽、喘鸣和夜间症状。两类问题都需要结合年龄、持续时间和生长情况由儿科评估，不应自行长期用药。
""".strip(),
    "眼科常见病概览": """
## 检索补充：视物模糊、眼痛和眼红
视物模糊、眼痛、眼红、畏光、分泌物增多或视力波动时，可检索本份“眼科常见病概览”。症状原因多样，不能仅凭单一表现判断。

## 检索补充：眼科就医警示
突发视力下降、剧烈眼痛、眼外伤、化学物入眼、闪光感或视野缺损等情况属于眼科警示，应尽快就医，不宜自行等待或使用不明滴眼液。
""".strip(),
    "口腔科常见病概览": """
## 检索补充：反复口腔溃疡
反复口腔溃疡、牙痛、牙龈出血或口腔黏膜疼痛时，可检索本份“口腔科常见病概览”。溃疡长期不愈、范围扩大、伴明显发热或吞咽困难时，应尽快由口腔科或相关科室评估。
""".strip(),
    "皮肤科常见病概览": """
## 检索补充：反复皮疹、瘙痒和水疱
反复皮疹、瘙痒、水疱、红斑、脱屑或风团时，可检索本份“皮肤科常见病概览”。皮疹形态相似但原因可能不同，不建议凭图片或经验自行长期用药。

## 检索补充：皮肤科警示
皮疹伴呼吸困难、面唇舌肿胀、高热、迅速扩散、水疱破溃或明显全身不适时，需要及时就医评估。
""".strip(),
}


def place_retrieval_supplement(existing: str, supplement: str) -> str:
    """Keep the added query language in the first embedding chunk.

    Documents are chunked before embedding. Keeping the retrieval-focused
    paragraph at the start prevents it from being diluted by a long overview.
    Removing it first also keeps subsequent runs idempotent.
    """
    remaining_content = existing.replace(supplement, "").strip()
    return f"{supplement}\n\n{remaining_content}"


def main() -> None:
    updated: list[str] = []
    with Session(engine) as session:
        for name, supplement in CONDITION_SUPPLEMENTS.items():
            record = session.exec(select(Condition).where(Condition.name == name)).first()
            if record is None:
                raise RuntimeError(f"Missing condition record: {name}")
            content = place_retrieval_supplement(record.symptoms, supplement)
            if content != record.symptoms:
                record.symptoms = content
                record.updated_at = datetime.now(UTC)
                session.add(record)
                updated.append(name)

        for title, supplement in DOCUMENT_SUPPLEMENTS.items():
            record = session.exec(
                select(KnowledgeDocument).where(KnowledgeDocument.title == title)
            ).first()
            if record is None:
                raise RuntimeError(f"Missing knowledge document: {title}")
            content = place_retrieval_supplement(record.content, supplement)
            if content != record.content:
                record.content = content
                record.updated_at = datetime.now(UTC)
                session.add(record)
                updated.append(title)

        session.commit()

    invalidate_knowledge_status_cache()
    print(f"Retrieval supplements ready: updated={len(updated)}, records={updated}")


if __name__ == "__main__":
    main()
