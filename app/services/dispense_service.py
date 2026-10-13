"""发药域业务服务（阶段 5）：处方审核、先审后发、发药联动与取药记录查询。

============================ 审核建模选择（方案 B：草稿 → 审核置真） ============================

DispensingRecord 一张表同时承载「审核」与「发药」两个动作，字段有 verified /
verification_notes / dispensed_at / notes。本模块采用**两步走**：

1. ``approve_prescription``：药师对 status=paid 的处方做四查十对。若该处方还没有
   发药记录，就建一条并直接 ``verified=True`` + 写审核意见；若已有草稿记录则把
   verified 翻真。RBAC 上挂的是 ``prescription:approve``。
2. ``dispense_prescription``：实际发药。**前置校验 DispensingRecord.verified is True**
   （先审后发），未审核直接拒。通过后写 pharmacist_id / dispensed_at / notes，
   并把处方 paid→dispensed。RBAC 上挂 ``dispensing:create``。

之所以选「先建即 verified=True」而不是「必须先单独建 verified=False 草稿再审核」：
后者要求多一个纯为状态而存在的建草稿端点，临床上药师的动作就是「审核通过」这一个
语义事件，把它拆成两次点击反而更容易漏掉第二步导致处方卡在无记录态。因此 approve
一步就把审核事实落到 DispensingRecord.verified=True，发药是它的下游闸门。
两种建模都能自洽，本实现选这条更贴近真实工作流的路径，并在测试里锁定「未 approve
就不能 dispense」。

============================ 不扣库存 ============================

系统没有库存表（models 层无 stock/inventory 实体，迁移 0001 亦未建）。发药只推进
处方状态与写发药记录，**不做任何库存扣减**。这不是遗漏而是范围界定：库存属于另一个
子系统，本批刻意不引入伪库存字段以免给未来接 WMS 制造假数据源。

============================ 跨域不依赖 prescription_service ============================

与收费域同源的理由：处方域文件由并行同事负责，本批启动时可能不存在。这里只依赖
冻结的模型层（Prescription / PrescriptionStatus / DispensingRecord）与已登记的
状态机 assert_transition("prescription", ...)，直接操作 ORM 对象推进状态，
不 import 尚不存在的 service。状态机是唯一真相来源，跨模块调用它并不绕过校验。

============================ 唯一性与终态 ============================

- 一张处方只能有一条发药记录（uq_dispensing_prescription）。service 在建记录前先查，
  给出友好 409 而不是让 IntegrityError 冒成 500。
- 处方到达 dispensed 即终态（PRESCRIPTION_TRANSITIONS 里 dispensed 无出边），
  已发药的处方/发药记录不可再改——二次发药会被状态机拒绝。
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    InvalidStateTransitionError,
    ResourceNotFoundError,
)
from app.models.clinical import Visit
from app.models.patient import PatientProfile
from app.models.pharmacy import (
    DispensingRecord,
    Prescription,
    PrescriptionStatus,
)
from app.services.state_machine import assert_transition

logger = logging.getLogger(__name__)


def utcnow_zoned() -> datetime:
    """时区感知的当前时间。单独成函数便于测试冻结。"""
    return datetime.now(UTC)


# ==================== 内部工具 ====================


async def _resolve_profile_id(db: AsyncSession, user_id: uuid.UUID) -> uuid.UUID | None:
    row = await db.execute(
        select(PatientProfile.id).where(PatientProfile.user_id == user_id)
    )
    return row.scalar_one_or_none()


async def _load_prescription(db: AsyncSession, rx_id: uuid.UUID) -> Prescription:
    rx = await db.get(Prescription, rx_id)
    if rx is None:
        raise ResourceNotFoundError("处方不存在")
    return rx


async def _find_record(db: AsyncSession, rx_id: uuid.UUID) -> DispensingRecord | None:
    row = await db.execute(
        select(DispensingRecord).where(DispensingRecord.prescription_id == rx_id)
    )
    return row.scalar_one_or_none()


# ==================== 审核 ============================


async def approve_prescription(
    db: AsyncSession,
    *,
    prescription_id: uuid.UUID,
    pharmacist_user_id: uuid.UUID,
    notes: str,
) -> DispensingRecord:
    """药师审核处方（paid 状态可审），落地到 DispensingRecord.verified=True。

    - 处方必须存在且处于 paid（已缴费待发药）才可审核：draft/confirmed 还没缴费，
      dispensed/cancelled 已过账，都不该被再次审核。非 paid → 409。
    - 若尚无发药记录：新建一条，pharmacist_id=当前药师，verified=True，写审核意见。
    - 若已有记录但未 verified：翻真并更新意见（幂等重审允许）。
    - 若已 verified：视为重复审核，幂等返回当前记录（药师复核不该报错）。
    """
    rx = await _load_prescription(db, prescription_id)
    if rx.status != PrescriptionStatus.PAID.value:
        raise InvalidStateTransitionError(
            "只有已缴费待发药的处方才能审核",
            current=str(rx.status),
            target="approved",
            entity="prescription",
        )

    record = await _find_record(db, rx.id)
    if record is None:
        record = DispensingRecord(
            prescription_id=rx.id,
            pharmacist_id=pharmacist_user_id,
            verified=True,
            verification_notes=notes,
        )
        db.add(record)
    elif not record.verified:
        record.verified = True
        record.pharmacist_id = pharmacist_user_id
        record.verification_notes = notes
    else:
        logger.info("处方 %s 已审核，幂等返回", rx.id)
        return record

    await db.commit()
    await db.refresh(record)
    logger.info(
        "处方审核完成 prescription_id=%s 药师=%s", rx.id, pharmacist_user_id
    )
    return record


# ==================== 发药 ============================


async def dispense_prescription(
    db: AsyncSession,
    *,
    prescription_id: uuid.UUID,
    pharmacist_user_id: uuid.UUID,
    notes: str | None = None,
) -> DispensingRecord:
    """发药：处方 paid→dispensed，写发药时间与备注。

    **先审后发硬闸门**：必须存在 DispensingRecord 且 verified=True，否则拒绝。
    未审核就发药在临床上是重大差错（四查十对没做），这里用业务规则而非仅靠 UI 拦。

    同一处方唯一发药记录（uq_dispensing_prescription）：本函数只在已有 verified
    记录上补 dispensed_at，不会新建第二条，天然满足唯一约束。
    处方已 dispensed（终态）→ assert_transition 抛 409，二次发药被拒。
    """
    rx = await _load_prescription(db, prescription_id)

    record = await _find_record(db, rx.id)
    if record is None or not record.verified:
        # 缺审核记录或未审核，都归为「先审后发」未满足。
        raise InvalidStateTransitionError(
            "处方尚未通过药师审核，不能发药（需先执行审核）",
            current=str(rx.status),
            target=PrescriptionStatus.DISPENSED.value,
            entity="dispensing",
        )

    # 状态机推进：paid→dispensed。非 paid（含已是 dispensed 的二次发药）在此被拦成 409。
    rx.status = assert_transition(
        "prescription", rx.status, PrescriptionStatus.DISPENSED.value
    )
    record.dispensed_at = utcnow_zoned()
    record.notes = notes
    # 发药执行人以实际操作的药师为准（可能与审核人不同）。
    record.pharmacist_id = pharmacist_user_id
    await db.commit()
    await db.refresh(record)
    logger.info(
        "处方已发药 prescription_id=%s 药师=%s", prescription_id, pharmacist_user_id
    )
    return record


# ==================== 读取（取药记录，含行级隔离） ============================


async def list_records(
    db: AsyncSession,
    *,
    requester_user_id: uuid.UUID,
    role_name: str,
    patient_id: uuid.UUID | None = None,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[DispensingRecord], int]:
    """发药记录分页列表。

    - patient：只看自己处方的取药记录（按 patient_id 反查其处方再关联记录）。
    - pharmacist（dispensing:read_all）：全院可见。
    - doctor：dispensing:read，收窄到自己接诊产生的处方对应记录。
    """
    conditions: list[Any] = []

    if role_name == "patient":
        own = await _resolve_profile_id(db, requester_user_id)
        if own is None:
            return [], 0
        conditions.append(
            DispensingRecord.prescription_id.in_(
                select(Prescription.id).where(Prescription.patient_id == own)
            )
        )
    elif role_name == "doctor":
        conditions.append(
            DispensingRecord.prescription_id.in_(
                select(Prescription.id).where(
                    Prescription.visit_id.in_(
                        select(Visit.id).where(Visit.doctor_id == requester_user_id)
                    )
                )
            )
        )
    # pharmacist/admin：不加患者限定。

    stmt = (
        select(DispensingRecord).where(*conditions)
        if conditions
        else select(DispensingRecord)
    )
    total = int(
        (await db.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one()
    )
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    rows = await db.execute(
        stmt.order_by(DispensingRecord.created_at.desc(), DispensingRecord.id)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return list(rows.scalars().all()), total


async def get_record_detail(
    db: AsyncSession,
    *,
    record_id: uuid.UUID,
    requester_user_id: uuid.UUID,
    role_name: str,
) -> DispensingRecord:
    """取药记录详情。越权一律 404（不确认他人记录存在）。"""
    record = await db.get(DispensingRecord, record_id)
    if record is None:
        raise ResourceNotFoundError("取药记录不存在")

    if role_name == "patient":
        own = await _resolve_profile_id(db, requester_user_id)
        if own is None:
            raise ResourceNotFoundError("取药记录不存在")
        rx = await db.get(Prescription, record.prescription_id)
        if rx is None or rx.patient_id != own:
            raise ResourceNotFoundError("取药记录不存在")
    elif role_name == "doctor":
        rx = await db.get(Prescription, record.prescription_id)
        visit = await db.get(Visit, rx.visit_id) if rx else None
        if visit is None or visit.doctor_id != requester_user_id:
            raise ResourceNotFoundError("取药记录不存在")
    # pharmacist/admin：read_all，放行。
    return record


async def serialize_record(
    db: AsyncSession, record: DispensingRecord
) -> dict[str, Any]:
    """ORM → DispensingRecordOut 字典，附处方快照列。

    不读 record.prescription / record.pharmacist 关系（lazy 加载在 AsyncSession
    下对主键取回对象会触发 MissingGreenlet）。处方信息用一次窄 SELECT 补齐。
    """
    rx_row = (
        await db.execute(
            select(
                Prescription.patient_id,
                Prescription.status,
                Prescription.total_amount,
            ).where(Prescription.id == record.prescription_id)
        )
    ).one_or_none()
    patient_id = str(rx_row[0]) if rx_row else None
    rx_status = rx_row[1] if rx_row else None
    total_amount = int(rx_row[2]) if rx_row and rx_row[2] is not None else None
    return {
        "id": str(record.id),
        "prescription_id": str(record.prescription_id),
        "pharmacist_id": str(record.pharmacist_id),
        "verified": bool(record.verified),
        "verification_notes": record.verification_notes,
        "dispensed_at": record.dispensed_at,
        "notes": record.notes,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "patient_id": patient_id,
        "prescription_status": rx_status,
        "total_amount": total_amount,
    }


__all__ = [
    "approve_prescription",
    "dispense_prescription",
    "get_record_detail",
    "list_records",
    "serialize_record",
    "utcnow_zoned",
]
