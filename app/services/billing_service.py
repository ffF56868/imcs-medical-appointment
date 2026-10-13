"""收费域业务服务（阶段 5）：建单、支付联动处方、退费、取消与查询。

============================ 金额口径 ============================

**一律 int「分」**，与 ``Prescription.total_amount``、``ExamRequest.fee_amount``、
``ChargeOrder.total_amount/paid_amount`` 同源（models/billing.py 模块 docstring
已写明「金额一律用分为单位存整数，避免浮点误差」）。本服务不做任何 /100 换算，
也不引入 Decimal——源单据本身就是 int 分，汇总只是整数相加，没有精度损失面。
自定义项（type=other）的金额由客户端以 int 分传入并在 schema 层限正整数上界。

============================ 跨域联动刻意不依赖 prescription_service ============================

支付要把关联处方从 confirmed→paid，退费要检查处方是否已发药。这两处都**直接操作
Prescription ORM 对象 + assert_transition("prescription", ...)**，而不是 import
尚不存在的 prescription_service：

- 处方域的 service/api/schema 由并行同事负责，本批启动时该模块可能还不存在，
  import 它会让收费域在合并前就无法通过 py_compile 与收集。
- 状态机的唯一真相来源是 ``state_machine.assert_transition`` + 模型里的
  ``PRESCRIPTION_TRANSITIONS``，跨模块直接调它并不绕过任何校验——反而比"猜一个
  同事的函数签名"更稳。
- 因此这里只依赖 **已冻结的模型层**（Prescription / PrescriptionStatus）与
  **已登记的状态机**，二者都在禁改清单里，接口稳定。

============================ 支付的事务语义：整单同生同灭 ============================

pay_charge_order 一次写两类数据：收费单 pending→paid + 每张关联处方 confirmed→paid。
它们必须**同一个 commit 落地**。任一处方当前不是 confirmed（比如已被作废、或还没
被医生开具确认），assert_transition 抛错 → 整个 session rollback → 收费单停在 pending，
绝不出现「钱收了但处方没进待发药队列」这种半态。这就是「已开具待发药」的缴费语义：
缴费这个动作把处方推进到 paid，是它的业务后果，不是一个可选副作用。

============================ 退费边界：模型的既有限制，不改状态机 ============================

PRESCRIPTION_TRANSITIONS 里 ``paid`` 的后继只有 ``dispensed``——**没有回退路径**。
这带来两条硬边界，本服务如实实现为拒绝，而不是试图绕开或修改状态机：

1. 关联处方已 ``dispensed`` → 拒绝退费。药已经发出，退款会造成「钱退了药没退」的
   账实不符；真实流程要走药品退回 + 人工冲正，不在演示范围。
2. 关联处方处于 ``paid``（尚未发药）→ **同样拒绝**，提示「请先由医生作废处方/走人工流程」。
   因为收费单一旦 paid，其处方就跟着到了 paid，而处方 paid 无法回退到 confirmed/cancelled，
   所以「退一张含处方项的已缴收费单」在当前状态机下**结构上不可达**。这是模型既有的限制，
   本批把它作为已知边界固化进测试，不去动 PRESCRIPTION_TRANSITIONS（禁改清单）。

结论：**只有 items 全为 exam/other 的收费单能被正常退费**。含处方项的收费单一旦支付即进入
不可逆区，退费必然被上面两条之一拦下。refund_charge_order 会先做这道前置判定再落库。
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
from app.models.billing import ChargeOrder, ChargeOrderStatus
from app.models.clinical import ExamRequest, Visit
from app.models.patient import PatientProfile
from app.models.pharmacy import Prescription, PrescriptionStatus
from app.services.state_machine import allowed_targets, assert_transition

logger = logging.getLogger(__name__)

# ==================== 常量 ====================

# 「占用中」的收费单状态：非 cancelled/refunded。同一处方只能被一张这类单子关联。
# pending 也算占用——否则两张待缴单能同时挂同一处方，任意一张支付都会重复扣费。
OCCUPYING_STATUSES: tuple[str, ...] = (
    ChargeOrderStatus.PENDING.value,
    ChargeOrderStatus.PAID.value,
)

# 退费时禁止放行的处方状态 → 拒绝理由文案。
# dispensed：药已发出；paid：处方状态机无回退边，退了收不回（见模块 docstring）。
_NON_REFUNDABLE_PRESCRIPTION: dict[str, str] = {
    PrescriptionStatus.DISPENSED.value: "关联处方已发药，不能退费（需走药品退回与人工冲正流程）",
    PrescriptionStatus.PAID.value: "关联处方已缴费待发药，请先由医生作废处方或走人工流程后再退费",
}


class DuplicatePrescriptionChargeError(InvalidStateTransitionError):
    """同一处方已被另一张未作废/未退费的收费单关联。

    为什么不新增到 exceptions.py（那里在禁改清单）：这是一个只在收费域出现的
    具体冲突，复用 ``InvalidStateTransitionError`` 的 409 + state_transition_invalid
    语义足够——它本质就是「当前状态下这张单不该再关联这个处方」。
    在本模块内定义子类，既不污染全局异常表，又能给出精确的业务文案。
    """


def utcnow_zoned() -> datetime:
    """时区感知的当前时间。单独成函数便于测试冻结时间。"""
    return datetime.now(UTC)


# ==================== receipt_no 生成 ============================


def _make_receipt_no(seq_hint: int | None = None) -> str:
    """生成小票号：CO + YYYYMMDD + 6 位序号后缀。

    **非全局唯一权威，仅演示**：真正的票据号应由数据库序列或专门的号码池分配，
    保证跨进程、跨日不重号。这里用「日期 + 秒级时间戳低位 + 传入序号」拼出一个
    形态可信、可读的号，够前端展示与面试演示，但在高并发或多实例下可能撞号。
    表上 receipt_no 有 unique=True 兜底，撞号会由 IntegrityError 暴露而非静默双票，
    生产落地必须换成 DB 序列。
    """
    now = utcnow_zoned()
    date_part = now.strftime("%Y%m%d")
    # 序号优先用调用方给的（便于测试可预测），否则退化到微秒低位。
    seq = seq_hint if seq_hint is not None else now.microsecond
    return f"CO{date_part}{seq % 1_000_000:06d}"


# ==================== 内部工具 ====================


async def _resolve_profile_id(db: AsyncSession, user_id: uuid.UUID) -> uuid.UUID | None:
    """账号 id → 患者档案 id。查不到返回 None，由调用方决定报错口径。"""
    row = await db.execute(
        select(PatientProfile.id).where(PatientProfile.user_id == user_id)
    )
    return row.scalar_one_or_none()


async def _load_order(db: AsyncSession, order_id: uuid.UUID) -> ChargeOrder:
    order = await db.get(ChargeOrder, order_id)
    if order is None:
        raise ResourceNotFoundError("收费单不存在")
    return order


def _extract_prescription_ids(items: list[dict[str, Any]]) -> list[uuid.UUID]:
    """从规范化后的 items 里取出全部处方 id（保持顺序、去重）。

    **必须转回 UUID**：items 落 JSONB 后 prescription_id 是字符串，
    直接把字符串喂给 ``db.get(Prescription, ...)``，SQLAlchemy 的 UUID
    主键绑定会在 ``value.hex`` 上崩掉（sqlite/aiosqlite 与 PG 同口径报错），
    支付/退费的处方联动就是这么被一个类型问题打断的。
    """
    seen: set[uuid.UUID] = set()
    out: list[uuid.UUID] = []
    for item in items:
        if item.get("type") == "prescription":
            pid = item.get("prescription_id")
            if pid:
                pid = _to_uuid(pid)
                if pid not in seen:
                    seen.add(pid)
                    out.append(pid)
    return out


async def _assert_prescriptions_not_double_charged(
    db: AsyncSession, prescription_ids: list[uuid.UUID]
) -> None:
    """查重：任一处方若已被另一张 pending/paid 收费单关联，硬拒。

    JSONB items 的按内容过滤在 SQLite 测试环境不可用（PG 才有 @> 运算符），
    因此这里改用「取候选单的 id + items，Python 侧解析 prescription_id 求交」的两段式：
    SQL 只做粗筛（限定 status IN OCCUPYING），精确判重在应用层完成。
    代价是本域处方关联量小（一页几十条），可接受；生产若要高频可按
    prescription_id 建一张关联表，但那属于模型改动，不在本批范围。
    """
    if not prescription_ids:
        return
    wanted = {pid.hex for pid in prescription_ids}
    rows = await db.execute(
        select(ChargeOrder.id, ChargeOrder.items).where(
            ChargeOrder.status.in_(OCCUPYING_STATUSES)
        )
    )
    for _oid, raw_items in rows.all():
        for item in raw_items or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "prescription":
                continue
            pid = item.get("prescription_id")
            if not pid:
                continue
            # **两边都必须规范化**：存量 items 里存的是带连字符的 str(uuid)，
            # 而 wanted 用的是 hex（无连字符）。拿 str 直接查 hex 集合永远不命中，
            # 判重会静默失效——这里吃过一次亏，用 _to_uuid().hex 统一口径。
            try:
                norm = _to_uuid(pid).hex
            except (ValueError, TypeError, AttributeError):
                continue
            if norm in wanted:
                raise DuplicatePrescriptionChargeError(
                    "同一处方只能被一张有效收费单关联，该处方已在其他待缴/已缴收费单中"
                )


# ==================== 建单 ============================


async def create_charge_order(
    db: AsyncSession,
    *,
    doctor_user_id: uuid.UUID,
    visit_id: uuid.UUID,
    items_in: list[Any],
) -> ChargeOrder:
    """医生为某次就诊创建收费单（status=pending）。

    - visit 必须存在且归属当前医生（行级隔离：越权 404，不确认他人就诊存在）。
    - patient_id 从 visit 带出，绝不由客户端指定。
    - 逐项求值金额：prescription 取 Prescription.total_amount，exam 取
      ExamRequest.fee_amount，other 用客户端传入的分值。规范化的 items
      （带服务端求出的 amount）落 JSONB，total_amount = 各项金额之和。
    - 处方查重：任一处方已被别的 pending/paid 单关联则整体拒绝。
    - 处方必须属于本次 visit（防串单）；检查项同理。
    """
    visit = await db.get(Visit, visit_id)
    if visit is None or visit.doctor_id != doctor_user_id:
        raise ResourceNotFoundError("就诊记录不存在")

    normalized: list[dict[str, Any]] = []
    total = 0

    # 先收集所有 prescription_id 做一次批量查重，避免逐条往返。
    # 必须转成 UUID：JSONB 读回后是字符串，直接喂给按内容比对的查重逻辑会失效。
    rx_ids = [
        _to_uuid(it.prescription_id)
        for it in items_in
        if getattr(it, "type", None) == "prescription"
    ]
    await _assert_prescriptions_not_double_charged(db, rx_ids)

    for item in items_in:
        itype = getattr(item, "type", None)
        if itype == "prescription":
            rx_id = _to_uuid(item.prescription_id)
            rx = await db.get(Prescription, rx_id)
            if rx is None:
                raise ResourceNotFoundError("关联的处方不存在")
            if rx.visit_id != visit.id:
                # 处方不属于这次就诊却挂进来，会把 A 就诊的药费算到 B 就诊头上。
                raise InvalidStateTransitionError(
                    "处方不属于该就诊，不能合并收费",
                    current=str(rx.status),
                    entity="charge_order",
                )
            amount = int(rx.total_amount or 0)
            normalized.append(
                {
                    "type": "prescription",
                    "prescription_id": str(rx_id),
                    "name": f"处方 {str(rx_id)[:8]}",
                    "amount": amount,
                }
            )
            total += amount
        elif itype == "exam":
            ex_id = _to_uuid(item.exam_request_id)
            ex = await db.get(ExamRequest, ex_id)
            if ex is None:
                raise ResourceNotFoundError("关联的检查申请单不存在")
            if ex.visit_id != visit.id:
                raise InvalidStateTransitionError(
                    "检查申请单不属于该就诊，不能合并收费",
                    current=str(ex.status),
                    entity="charge_order",
                )
            amount = int(ex.fee_amount or 0)
            normalized.append(
                {
                    "type": "exam",
                    "exam_request_id": str(ex_id),
                    "name": ex.exam_name,
                    "amount": amount,
                }
            )
            total += amount
        elif itype == "other":
            amount = int(item.amount)
            normalized.append(
                {"type": "other", "name": item.name, "amount": amount}
            )
            total += amount
        else:
            raise InvalidStateTransitionError(
                f"未知的费用项类型：{itype}", entity="charge_order"
            )

    order = ChargeOrder(
        visit_id=visit.id,
        patient_id=visit.patient_id,
        items=normalized,
        total_amount=total,
        paid_amount=0,
        status=ChargeOrderStatus.PENDING.value,
    )
    db.add(order)
    await db.commit()
    await db.refresh(order)
    logger.info(
        "收费单已创建 id=%s visit_id=%s 项数=%d 应收=%d 操作员=%s",
        order.id,
        visit.id,
        len(normalized),
        total,
        doctor_user_id,
    )
    return order


def _to_uuid(value: object) -> uuid.UUID:
    """items 里的 id 已是合法 UUID 字符串（schema 层校验过），此处转回 UUID。"""
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


# ==================== 支付 ============================


async def pay_charge_order(
    db: AsyncSession,
    *,
    order_id: uuid.UUID,
    cashier_user_id: uuid.UUID,
    channel: str = "mock",
) -> ChargeOrder:
    """收费员确认收款：pending→paid，并联动关联处方 confirmed→paid。

    **整单同生同灭**：先把每张处方推进到 paid（非 confirmed 则 assert_transition
    抛错），再推收费单，最后一次性 commit。任一步失败 → rollback，收费单回到 pending，
    不留「钱收了处方没动」的脏写。这正是「已开具待发药」的缴费语义：缴费把处方
    推到 paid，是这一动作的业务后果而非可选副作用。

    paid_amount 恒置为应收全额（本项目 mock 支付不支持部分缴费）。
    幂等：已是 paid 的单再次支付直接返回当前态（收费员双击不该报错，也不该重复联动）。
    """
    order = await _load_order(db, order_id)
    if order.status == ChargeOrderStatus.PAID.value:
        logger.info("收费单 %s 已是 paid，幂等返回", order.id)
        return order

    try:
        # 1) 联动处方：confirmed→paid。放在收费单之前，让处方状态成为主闸门。
        for rx_id in _extract_prescription_ids(order.items or []):
            rx = await db.get(Prescription, rx_id)
            if rx is None:
                raise ResourceNotFoundError("关联的处方不存在")
            rx.status = assert_transition(
                "prescription",
                rx.status,
                PrescriptionStatus.PAID.value,
            )

        # 2) 推进收费单本身。
        order.status = assert_transition(
            "charge_order", order.status, ChargeOrderStatus.PAID.value
        )
        order.paid_amount = order.total_amount
        order.payment_channel = channel
        order.cashier_id = cashier_user_id
        order.paid_at = utcnow_zoned()
        order.receipt_no = _make_receipt_no()

        await db.commit()
    except Exception:
        # 处方联动失败（如某张处方非 confirmed）时，回滚整个事务，
        # 收费单与已改写的处方内存态一起撤销，回到 pending。
        await db.rollback()
        raise

    await db.refresh(order)
    logger.info(
        "收费单已支付 id=%s receipt=%s 实收=%d 渠道=%s 收费员=%s",
        order.id,
        order.receipt_no,
        order.paid_amount,
        channel,
        cashier_user_id,
    )
    return order


# ==================== 退费 ============================


async def refund_charge_order(
    db: AsyncSession,
    *,
    order_id: uuid.UUID,
    cashier_user_id: uuid.UUID,
    reason: str,
) -> ChargeOrder:
    """收费员退费：paid→refunded。

    **退费边界（模型既有限制，见模块 docstring）**：先扫描关联处方状态，
    命中 dispensed / paid 之一即拒绝——前者药已发出账实不符，后者处方状态机
    无回退边，退了收不回。只有 items 全为 exam/other 的收费单能顺利退费。
    判定在任何写库之前完成，拒绝时不落半个字段。

    成功后写 refunded_at / refund_reason，paid_amount 归零（模型注释：退费后为 0）。
    """
    order = await _load_order(db, order_id)

    # 起点必须是 paid：refunded/cancelled/pending 都不允许直接退。
    # 对已 refunded 给幂等友好处理前先做状态合法性判断，避免误清字段。
    if order.status == ChargeOrderStatus.REFUNDED.value:
        logger.info("收费单 %s 已退费，幂等返回", order.id)
        return order

    # 前置业务闸门：处方联动检查（在任何写入之前）。
    for rx_id in _extract_prescription_ids(order.items or []):
        rx = await db.get(Prescription, rx_id)
        if rx is None:
            raise ResourceNotFoundError("关联的处方不存在")
        block_msg = _NON_REFUNDABLE_PRESCRIPTION.get(rx.status)
        if block_msg is not None:
            raise InvalidStateTransitionError(
                block_msg,
                current=str(rx.status),
                target=ChargeOrderStatus.REFUNDED.value,
                entity="prescription",
            )

    # 状态机迁移（pending→refunded 非法，会被 assert_transition 拦成 409）。
    order.status = assert_transition(
        "charge_order", order.status, ChargeOrderStatus.REFUNDED.value
    )
    order.refunded_at = utcnow_zoned()
    order.refund_reason = reason
    order.paid_amount = 0
    await db.commit()
    await db.refresh(order)
    logger.info(
        "收费单已退费 id=%s 原因=%s 收费员=%s", order.id, reason, cashier_user_id
    )
    return order


# ==================== 取消 ============================


async def cancel_charge_order(
    db: AsyncSession,
    *,
    order_id: uuid.UUID,
    operator_user_id: uuid.UUID,
) -> ChargeOrder:
    """作废收费单：pending→cancelled。已支付的单不能作废（矩阵里 paid 只能→refunded）。"""
    order = await _load_order(db, order_id)
    if order.status == ChargeOrderStatus.CANCELLED.value:
        logger.info("收费单 %s 已作废，幂等返回", order.id)
        return order
    order.status = assert_transition(
        "charge_order", order.status, ChargeOrderStatus.CANCELLED.value
    )
    await db.commit()
    await db.refresh(order)
    logger.info("收费单已作废 id=%s 操作员=%s", order.id, operator_user_id)
    return order


# ==================== 读取（列表 + 详情，含行级隔离） ============================


async def list_orders(
    db: AsyncSession,
    *,
    requester_user_id: uuid.UUID,
    role_name: str,
    patient_id: uuid.UUID | None = None,
    status_filter: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[ChargeOrder], int]:
    """收费单分页列表，按角色强制行级可见性。

    - patient：只看自己档案下的单（越权的 patient_id 被忽略并按自己档案过滤，
      返回空集而非 403——不向外确认他人记录是否存在，与 exam/consult 同口径）。
    - doctor：看自己名下就诊关联的单（按 visits.doctor_id 反查，不收窄到单患者）。
    - cashier/pharmacist（read_all）：全院可见，可按 patient_id/status 收窄。
    """
    conditions: list[Any] = []

    if role_name == "patient":
        own = await _resolve_profile_id(db, requester_user_id)
        if own is None:
            return [], 0
        conditions.append(ChargeOrder.patient_id == own)
    elif role_name == "doctor":
        # 医生读的是「自己接诊产生的收费单」：join visits 限定 doctor_id。
        # 不给 read_all，因此不能用全院视图。
        conditions.append(
            ChargeOrder.visit_id.in_(
                select(Visit.id).where(Visit.doctor_id == requester_user_id)
            )
        )
        if patient_id is not None:
            conditions.append(ChargeOrder.patient_id == patient_id)
    else:
        # cashier / pharmacist / admin：全域视图，可按患者与状态收窄。
        if patient_id is not None:
            conditions.append(ChargeOrder.patient_id == patient_id)

    if status_filter is not None:
        conditions.append(ChargeOrder.status == status_filter)

    stmt = select(ChargeOrder).where(*conditions) if conditions else select(ChargeOrder)
    total = int(
        (await db.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one()
    )
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    rows = await db.execute(
        stmt.order_by(ChargeOrder.created_at.desc(), ChargeOrder.id)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return list(rows.scalars().all()), total


async def get_order_detail(
    db: AsyncSession,
    *,
    order_id: uuid.UUID,
    requester_user_id: uuid.UUID,
    role_name: str,
) -> ChargeOrder:
    """收费单详情。越权一律 404（不确认他人记录存在）。"""
    order = await _load_order(db, order_id)

    if role_name == "patient":
        own = await _resolve_profile_id(db, requester_user_id)
        if own is None or order.patient_id != own:
            raise ResourceNotFoundError("收费单不存在")
    elif role_name == "doctor":
        visit = await db.get(Visit, order.visit_id)
        if visit is None or visit.doctor_id != requester_user_id:
            raise ResourceNotFoundError("收费单不存在")
    # cashier / pharmacist / admin：持有 read_all，放行全量读取。
    return order


def serialize_order(order: ChargeOrder) -> dict[str, Any]:
    """ORM → 可直接喂给 ChargeOrderOut 的字典（序列化统一在 service 出口）。

    不读 order.cashier 关系（lazy=selectin 在 AsyncSession 下对 db.get 取回的对象
    首次访问会触发隐式同步 IO → MissingGreenlet → 500，见 consult_service 同款坑）。
    cashier_id 直接用列值即可。
    """
    items_out: list[dict[str, Any]] = []
    for item in order.items or []:
        if not isinstance(item, dict):
            continue
        items_out.append(
            {
                "type": item.get("type"),
                "prescription_id": item.get("prescription_id"),
                "exam_request_id": item.get("exam_request_id"),
                "name": item.get("name"),
                "amount": int(item.get("amount") or 0),
            }
        )
    return {
        "id": str(order.id),
        "visit_id": str(order.visit_id),
        "patient_id": str(order.patient_id),
        "status": order.status,
        "items": items_out,
        "total_amount": int(order.total_amount or 0),
        "paid_amount": int(order.paid_amount or 0),
        "payment_channel": order.payment_channel,
        "cashier_id": str(order.cashier_id) if order.cashier_id else None,
        "paid_at": order.paid_at,
        "refunded_at": order.refunded_at,
        "refund_reason": order.refund_reason,
        "receipt_no": order.receipt_no,
        "created_at": order.created_at,
        "updated_at": order.updated_at,
        "allowed_transitions": allowed_targets("charge_order", order.status),
    }


__all__ = [
    "DuplicatePrescriptionChargeError",
    "OCCUPYING_STATUSES",
    "cancel_charge_order",
    "create_charge_order",
    "get_order_detail",
    "list_orders",
    "pay_charge_order",
    "refund_charge_order",
    "serialize_order",
    "utcnow_zoned",
]
