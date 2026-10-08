import json
from hashlib import sha256

from sqlmodel import Session, delete, select

from app.legacy.models import Condition, Drug, KnowledgeDocument, KnowledgeSnapshot


def serialize_source_record(record: Condition | Drug | KnowledgeDocument) -> dict:
    return record.model_dump(mode="json")


def build_snapshot_payload(session: Session) -> dict[str, list[dict]]:
    return {
        "conditions": [
            serialize_source_record(record)
            for record in session.exec(select(Condition).order_by(Condition.id)).all()
        ],
        "drugs": [
            serialize_source_record(record)
            for record in session.exec(select(Drug).order_by(Drug.id)).all()
        ],
        "documents": [
            serialize_source_record(record)
            for record in session.exec(
                select(KnowledgeDocument).order_by(KnowledgeDocument.id)
            ).all()
        ],
    }


def serialize_snapshot_payload(payload: dict[str, list[dict]]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def get_snapshot_hash(payload: dict[str, list[dict]]) -> str:
    serialized = serialize_snapshot_payload(payload)
    return sha256(serialized.encode("utf-8")).hexdigest()


def get_snapshot_document_count(payload: dict[str, list[dict]]) -> int:
    return sum(len(records) for records in payload.values())


def create_knowledge_snapshot(
    session: Session,
    reason: str,
) -> tuple[KnowledgeSnapshot, bool]:
    payload = build_snapshot_payload(session)
    content_hash = get_snapshot_hash(payload)
    existing_snapshot = session.exec(
        select(KnowledgeSnapshot)
        .where(KnowledgeSnapshot.content_hash == content_hash)
        .order_by(KnowledgeSnapshot.id.desc())
    ).first()
    if existing_snapshot is not None:
        return existing_snapshot, False

    snapshot = KnowledgeSnapshot(
        content_hash=content_hash,
        document_count=get_snapshot_document_count(payload),
        reason=reason,
        payload_json=serialize_snapshot_payload(payload),
    )
    session.add(snapshot)
    session.flush()
    return snapshot, True


def get_current_snapshot_hash(session: Session) -> str:
    return get_snapshot_hash(build_snapshot_payload(session))


def load_snapshot_payload(snapshot: KnowledgeSnapshot) -> dict[str, list[dict]]:
    payload = json.loads(snapshot.payload_json)
    required_keys = {"conditions", "drugs", "documents"}
    if not isinstance(payload, dict) or not required_keys.issubset(payload):
        raise ValueError("知识库版本数据格式无效")
    if not all(isinstance(payload[key], list) for key in required_keys):
        raise ValueError("知识库版本数据格式无效")
    return payload


def restore_snapshot_payload(
    session: Session,
    snapshot: KnowledgeSnapshot,
) -> int:
    payload = load_snapshot_payload(snapshot)

    session.exec(delete(Condition))
    session.exec(delete(Drug))
    session.exec(delete(KnowledgeDocument))
    session.flush()

    session.add_all(
        [Condition.model_validate(record) for record in payload["conditions"]]
    )
    session.add_all([Drug.model_validate(record) for record in payload["drugs"]])
    session.add_all(
        [
            KnowledgeDocument.model_validate(record)
            for record in payload["documents"]
        ]
    )
    session.flush()
    return get_snapshot_document_count(payload)
