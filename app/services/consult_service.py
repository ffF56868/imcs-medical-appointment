"""接诊、病历、检查开单与结案的业务逻辑（阶段 4 医生侧）。

链路位置：``appointment(checked_in) → visit(in_progress) → 病历(documented)
→ 检查申请(exam_request) → visit(closed) + appointment(completed)``。
签到（booked→checked_in）已在 appointment 侧实现，本模块**不重复实现**，
只把 checked_in 当作开始接诊的硬前置。

============================ 为什么「同一预约只有一个活跃 visit」 ============================
``visits.appointment_id`` 带 ``unique=True``（models/clinical.py），数据库层面
本来就一对一。但唯一约束只能告诉我们「撞了」，给不出「撞在哪条」，
而且 SQLite 测试环境根本不复刻 PG 的部分索引谓词。所以这里用一条显式查询
做第一层防呆：**判据是 ``status != 'closed'``，而不是「该 appointment 是否已有 visit」**。

这个口径差别不是文字游戏：现实中患者被要求「明天带着检查结果回来复读」是常态，
若按「存在过 visit 就拒绝」实现，医生第二天就无法再打开这次挂号；
而按「无活跃 visit」实现，同一次挂号可以合法地关闭后重开，
数据上留下两段完整病历而不是把两次问诊塞进一段。代价是一次挂号可能对应
多条 visit 历史，因此所有读取都必须显式区分「活跃」与「全部」。

============================ 状态机在本批的两处收紧 ============================

1. **documented 之后仍可回改病历**。矩阵里 documented 只有 closed 一条出边，
   看起来像「定稿即锁死」，但转换矩阵管的是**状态**迁移，不管**字段**写入。
   放开字段回改的理由是真实的临床时序：AI 辅助开药前医生要能补体格检查和
   诊断细节，锁死会让医生为了改一个字而去找管理员。写入口的闸门是
   「visit 未 closed」，而不是「visit 处于 in_progress」。
2. **closed 必须先经过 documented**。矩阵其实允许 in_progress→closed 直跳，
   本模块刻意不走那条边——没有诊断结论的结案等于凭空造出一条无法质控的病历，
   医疗纠纷时医院拿不出任何诊疗依据。收紧方式与 ``cancel_appointment``
   同一套做法：在调用点加白名单前置判定，**不改全局转换矩阵**，
   矩阵仍是合法性唯一真相来源，两层都过才写库。

============================ 跨实体事务：病历结案与预约完成同生同灭 ============================

沿用阶段 3「号源扣减与预约记录同生同灭」的写法：``close_visit`` 里
visit.closed 与 appointment.completed 必须**同一个 commit** 落地，任何一步失败整体回滚。
半途失败的后果是对称的——病历关了但预约还挂在 in_progress，医生工作台
永远显示这个人没看完；预约完成了但病历没关，这条 visit 会成为无人认领的
活跃记录并挡住后续重开接诊。两个方向都是真实的数据事故，所以不做补偿、
不做最终一致，直接同事务。

============================ 与其它批次的边界 ============================

- **不 import 检验侧的任何文件**（样本接收、结果录入与智能解读由并行同事负责）。
  本模块只写入 ``ExamRequest(status='ordered')`` 这一初始态，之后的 received /
  processing / reported / reviewed 全部由检验侧推进。危急路径通过
  ``urgency='stat'`` 的 info 日志留痕，供同事侧与运维检索，不建立任何函数级耦合。
- **不 import legacy 侧任何模块**（旧版路由包、旧数据库会话工厂、旧模型层）。
- **不碰收费**：``is_charged=False``、``fee_amount`` 取本模块演示价目常量，
  生成 charge_order 与实收是阶段 5 的事。
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    DuplicateAppointmentError,
    InvalidStateTransitionError,
    ResourceNotFoundError,
)
from app.models.auth import RoleName, User
from app.models.base import utcnow
from app.models.clinical import (
    Appointment,
    AppointmentStatus,
    ExamRequest,
    ExamRequestStatus,
    Urgency,
    Visit,
    VisitStatus,
)
from app.models.patient import PatientProfile
from app.schemas.consult import MedicalRecordSaveRequest, ExamOrderRequest
from app.services.state_machine import allowed_targets, assert_transition, validate_status_value

logger = logging.getLogger(__name__)

# 「今天」必须按东八区算，理由与 appointment_service.today_local() 完全一致：
# 容器默认 UTC，datetime.now(UTC).date() 在北京时间 00:00–08:00 之间会返回「昨天」，
# 于是当天待诊队列整个查空，演示时表现为「医生工作台今天没人」。
_LOCAL_TZ = ZoneInfo("Asia/Shanghai")


def today_local() -> date:
    """东八区口径的今天。单独成函数便于测试冻结时间。"""
    return datetime.now(_LOCAL_TZ).date()


# ==================== 演示价目常量 ====================

# **来源说明：这是演示价目，不是物价办核准的真实收费标准。**
# 数值参照常见公立医院检验项目价格量级（人民币元）放大 100 倍存为「分」，
# 目的是让接口返回的金额字段有可信的量级感，便于前端展示与阶段 5 收费联调。
# 单位一律「分」（Integer），不用 float：模型层注释已写明「避免浮点误差」，
# 一旦在这里除以 100，精度决策就被推给了二进制浮点。
# 表外项目（exam_type 不在键里）费用记 0 并打日志，交由收费侧核价——
# 比在医生开单时硬拦更符合临床实际，因为检查项目清单远不止这八项。
EXAM_FEE_TABLE_CENTS: dict[str, int] = {
    "blood_routine": 2500,  # 血常规 ¥25.00
    "urine_routine": 1500,  # 尿常规 ¥15.00
    "liver_function": 6000,  # 肝功能全套 ¥60.00
    "renal_function": 6000,  # 肾功能全套 ¥60.00
    "blood_glucose": 500,  # 血糖 ¥5.00
    "blood_lipid": 800,  # 血脂四项 ¥8.00
    "ecg": 3000,  # 心电图 ¥30.00
    "ultrasound_abdomen": 12000,  # 腹部 B 超 ¥120.00
}

# 项目中文名，仅在价目命中日志里用；开单入参仍要求医生显式填 exam_name，
# 因为同名项目在不同医院叫法不同，服务端猜名字容易和收费票据对不上。
EXAM_FEE_TABLE_NAMES: dict[str, str] = {
    "blood_routine": "血常规",
    "urine_routine": "尿常规",
    "liver_function": "肝功能全套",
    "renal_function": "肾功能全套",
    "blood_glucose": "血糖",
    "blood_lipid": "血脂四项",
    "ecg": "心电图",
    "ultrasound_abdomen": "腹部B超",
}

# 未命中价目表时的默认费用。0 表示「待核价」而不是「免费」，
# 语义由 is_charged=False 配合表达，阶段 5 收费侧会重新定价。
DEFAULT_EXAM_FEE_CENTS = 0

# 病历可开检查的状态白名单。closed 不在其中：结案后再开单属于
# 「另起一次就诊」，必须走新的 visit，否则检查报告会挂到一段已归档病历下，
# 质控抽样时会当成漏诊处理。
OPENABLE_VISIT_STATUSES: tuple[str, ...] = (
    VisitStatus.IN_PROGRESS.value,
    VisitStatus.DOCUMENTED.value,
)

# 「活跃 visit」判据：未到终态 closed 的一切就诊。见模块 docstring。
ACTIVE_VISIT_STATUSES: tuple[str, ...] = (
    VisitStatus.IN_PROGRESS.value,
    VisitStatus.DOCUMENTED.value,
)

# 医生工作台待诊队列的预约状态集合。
# 含 in_progress 是因为「接诊中但还没写完」的人同样还在等待区，
# 医生刷新页面时必须能一眼看到自己做到一半的那几个。
PENDING_APPOINTMENT_STATUSES: tuple[str, ...] = (
    AppointmentStatus.CHECKED_IN.value,
    AppointmentStatus.IN_PROGRESS.value,
)

# 结束就诊时预约必须处于的状态。矩阵里 checked_in→completed 并不合法，
# 因此这里不需要额外收紧：assert_transition 本身就会拦住「没接诊就结案」。
APPOINTMENT_CLOSE_SOURCE = AppointmentStatus.IN_PROGRESS.value

# 开始接诊响应里附带的历史病历条数上限。5 条足够覆盖「最近几次因何就诊」，
# 全量返回会让打开接诊面板这一次点击的响应体随患者病史线性膨胀。
PRIOR_VISITS_LIMIT = 5


# ==================== 内部工具 ====================


def _stringify(value: object | None) -> str | None:
    """UUID → str，None 保持 None（响应契约与后端方言无关）。"""
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


async def _resolve_profile_id(db: AsyncSession, user_id: uuid.UUID) -> uuid.UUID | None:
    """账号 id → 患者档案 id。查不到返回 None，由调用方决定报错口径。

    ``visits.patient_id`` 的外键指向 ``patient_profiles.id`` 而非 ``users.id``
    （models/clinical.py），这与 appointment 侧同源。医生接诊时的患者身份
    **从预约带出来**，绝不由医生传入，见 ``start_consultation``。
    """
    row = await db.execute(
        select(PatientProfile.id).where(PatientProfile.user_id == user_id)
    )
    return row.scalar_one_or_none()


async def _load_visit(db: AsyncSession, visit_id: uuid.UUID) -> Visit:
    """取 visit，不存在 404。"""
    visit = await db.get(Visit, visit_id)
    if visit is None:
        raise ResourceNotFoundError("就诊记录不存在")
    return visit


async def _require_own_visit(db: AsyncSession, visit_id: uuid.UUID, doctor_id: uuid.UUID) -> Visit:
    """行级隔离（医生）：只能操作自己名下的 visit，越权一律 404。

    404 而不是 403：403 等于承认「这条病历存在，只是不属于你」，
    攻击者可以据此枚举全院就诊 ID；404 与「不存在」不可区分。
    与 appointment / knowledge / triage 侧同一口径。
    """
    visit = await _load_visit(db, visit_id)
    if visit.doctor_id != doctor_id:
        raise ResourceNotFoundError("就诊记录不存在")
    return visit


async def _active_visit_id_map(
    db: AsyncSession, appointment_ids: list[uuid.UUID]
) -> dict[uuid.UUID, tuple[uuid.UUID, str]]:
    """批量取「这些预约各自的活跃 visit」，返回 appointment_id -> (visit_id, status)。

    用一次 IN 查询而不是逐预约查：工作台一页可能有几十个人，
    N+1 查询在医生早上开机的高峰时段会把接口拖成秒级，
    而这里只需要两张列的窄结果集，IN 一次拿完最省。
    """
    if not appointment_ids:
        return {}
    rows = await db.execute(
        select(Visit.appointment_id, Visit.id, Visit.status).where(
            Visit.appointment_id.in_(appointment_ids),
            Visit.status.in_(ACTIVE_VISIT_STATUSES),
        )
    )
    return {row[0]: (row[1], row[2]) for row in rows.all()}


async def _doctor_name(db: AsyncSession, doctor_id: uuid.UUID) -> str:
    """单列 SELECT 取医生姓名。

    **不读 visit.doctor 关系**：它是 ``lazy="selectin"``，只在查询时预取；
    对经 ``db.get()`` 主键取回的对象首次访问关系属性会触发隐式同步 IO，
    AsyncSession 下抛 MissingGreenlet，把一次正常读取变成 500
    （appointment_service.serialize_appointment 里同样的坑已经踩过一次）。
    """
    row = await db.execute(select(User.real_name).where(User.id == doctor_id))
    return str(row.scalar_one_or_none() or "")


async def serialize_visit(db: AsyncSession, visit: Visit) -> dict[str, object]:
    """Visit ORM → 可直接喂给 ``VisitOut`` 的字典（序列化统一在 service 出口）。"""
    exam_count = int(
        (
            await db.execute(
                select(func.count())
                .select_from(ExamRequest)
                .where(ExamRequest.visit_id == visit.id)
            )
        ).scalar_one()
    )
    return {
        "id": _stringify(visit.id),
        "appointment_id": _stringify(visit.appointment_id),
        "patient_id": _stringify(visit.patient_id),
        "doctor_id": _stringify(visit.doctor_id),
        "doctor_name": await _doctor_name(db, visit.doctor_id),
        "chief_complaint": visit.chief_complaint,
        "present_illness": visit.present_illness,
        "past_history": visit.past_history,
        "physical_exam": visit.physical_exam,
        "diagnosis": visit.diagnosis,
        "icd_codes": list(visit.icd_codes or []),
        "treatment_advice": visit.treatment_advice,
        "status": visit.status,
        "started_at": visit.started_at,
        "ended_at": visit.ended_at,
        "created_at": visit.created_at,
        "updated_at": visit.updated_at,
        "exam_request_count": exam_count,
        # 业务收紧后的可选迁移，而非裸矩阵投影（见 _visit_allowed_transitions）
        "allowed_transitions": _visit_allowed_transitions(visit.status),
    }


def _visit_allowed_transitions(status: str) -> list[str]:
    """返回前端可安全渲染的迁移列表 = 矩阵迁移 ∩ 服务层实际放行者。

    不能用裸 allowed_targets：VISIT_TRANSITIONS 允许 in_progress→closed 直跳，
    但 close_visit 刻意收紧为「必须先 documented」（空病历不得归档，见其
    docstring）。照抄矩阵会让前端渲染出一个点了必得 409 的「结束就诊」按钮
    ——可操作性的承诺必须由后端同一口径给出，而不是让前端去猜哪条边有闸门。
    """
    targets = allowed_targets("visit", status)
    if status == VisitStatus.IN_PROGRESS.value:
        return [t for t in targets if t != VisitStatus.CLOSED.value]
    return targets


async def serialize_exam_request(db: AsyncSession, request: ExamRequest) -> dict[str, object]:
    """ExamRequest ORM → ``ExamRequestOut`` 字典。"""
    del db  # 当前字段全在单行内，无需关联查询；保留形参以便日后扩展
    return {
        "id": _stringify(request.id),
        "visit_id": _stringify(request.visit_id),
        "patient_id": _stringify(request.patient_id),
        "doctor_id": _stringify(request.doctor_id),
        "exam_type": request.exam_type,
        "exam_name": request.exam_name,
        "body_part": request.body_part,
        "clinical_info": request.clinical_info,
        "urgency": request.urgency,
        "status": request.status,
        "is_charged": request.is_charged,
        "fee_amount": request.fee_amount,
        "created_at": request.created_at,
        "allowed_transitions": allowed_targets("exam_request", request.status),
    }


# ==================== 开始接诊 ====================


async def start_consultation(
    db: AsyncSession,
    appointment_id: uuid.UUID,
    doctor_id: uuid.UUID,
) -> Visit:
    """开始接诊：预约 checked_in→in_progress，并新建一条 in_progress 病历。

    **两步同事务，同生同灭**（写法承袭阶段 3 的号源扣减与预约创建）：
    预约改了状态但病历没建 → 医生点开工作台发现「这人正在接诊却没有病历」，
    既不能书写也不能结案，只能人工改库；
    病历建了但预约没改 → 患者侧仍显示「已到院待诊」，叫号系统会把他再次排进去。
    两种半态都无法由系统自愈，因此这里只在最后 ``commit()`` 一次，
    中途任何异常都由 FastAPI 会话关闭自然回滚，不留部分写入。

    患者身份来自预约而非入参：``appointment.patient_id`` 已经是
    patient_profiles.id，医生没有任何途径指定「我在看谁」，
    归属越权在数据结构层面就不成立。

    主诉从预约带出：挂号时患者填的 chief_complaint 直接成为病历必填项，
    既省去医生重录，也保证「患者自述」这一原始口径不被转述污染。
    预约上没有主诉时用占位文本兜底而不是让 visit 落一个 NULL——
    该列 nullable=False，真撞非空约束会变成 500，而给出可读占位符
    能让医生立刻在病历里看到「需要补充主诉」。
    """
    appointment = await db.get(Appointment, appointment_id)
    if appointment is None:
        raise ResourceNotFoundError("预约不存在")

    # 行级隔离：医生只能接诊分给自己的预约。放在状态校验之前，
    # 否则别的医生能靠错误信息反推出这张单子处于什么状态。
    if appointment.doctor_id != doctor_id:
        raise ResourceNotFoundError("预约不存在")

    # 活跃 visit 判据（见模块 docstring）：closed 的历史病历不拦重开。
    active = (
        await db.execute(
            select(Visit.id, Visit.status).where(
                Visit.appointment_id == appointment_id,
                Visit.status.in_(ACTIVE_VISIT_STATUSES),
            )
        )
    ).first()
    if active is not None:
        raise DuplicateAppointmentError(
            "该预约已在接诊中，请勿重复开始",
            detail={"visit_id": str(active[0]), "visit_status": active[1]},
        )

    # 状态机：非法前置（booked / completed / cancelled / no_show）在此自然 409，
    # 错误里自带 current/target/entity，前端能直接翻译成人话。
    appointment.status = assert_transition(
        "appointment",
        appointment.status,
        AppointmentStatus.IN_PROGRESS.value,
    )

    initial_status = validate_status_value("visit", VisitStatus.IN_PROGRESS.value)

    visit = Visit(
        appointment_id=appointment.id,
        patient_id=appointment.patient_id,
        doctor_id=appointment.doctor_id,
        chief_complaint=appointment.chief_complaint or "患者未填写主诉，请补充",
        status=initial_status,
        started_at=utcnow(),
    )
    db.add(visit)

    await db.flush()
    await db.commit()
    await db.refresh(visit)

    logger.info(
        "开始接诊 visit=%s appointment=%s doctor=%s patient=%s",
        visit.id,
        appointment.id,
        doctor_id,
        appointment.patient_id,
    )
    return visit


# ==================== 病历书写 ====================


async def save_medical_record(
    db: AsyncSession,
    visit_id: uuid.UUID,
    doctor_id: uuid.UUID,
    payload: MedicalRecordSaveRequest,
) -> tuple[Visit, bool]:
    """保存病历字段，首次写入时把 visit 从 in_progress 推到 documented。

    返回 ``(visit, transitioned)``：transitioned 如实告知本次是否发生了流转，
    前端据此提示「病历已定稿」而不必自己比对状态串。

    **documented 之后仍然允许改写**（模块 docstring 第 1 条收紧点）：
    闸门是「未 closed」而非「处于 in_progress」。理由是 AI 辅助开药前医生要能
    补齐体检所见和诊断细节，锁死字段会逼人绕道改库。

    部分更新语义：payload 里传了才覆盖，None 保持原值。这让渐进成文成立，
    代价是**无法把某栏改回空**——清空一栏属于极少发生的更正操作，
    真要撤回应当整段重写而不是指望 None 当删除标记（把 None 同时用作
    「未提供」和「置空」是两个语义挤一个值，迟早出事）。

    状态流转只在「此前尚未定稿」时发生一次。若每次保存都调 assert_transition，
    第二次保存会撞上 documented→documented 这条不存在的边而报 409，
    于是医生修改一个字就被拒——所以这里判断的是「当前是否已是目标态」，
    已是则跳过迁移，而不是绕过状态机直接赋值。
    """
    visit = await _require_own_visit(db, visit_id, doctor_id)

    if visit.status == VisitStatus.CLOSED.value:
        # closed 是终态：病历归档后任何人不能再改内容。这里给出比
        # assert_transition 更具体的提示（用户此刻想做的是「改病历」而不是
        # 「改状态」），但仍不调用状态机去伪造一次迁移。
        raise InvalidStateTransitionError(
            "就诊已结束，病历已归档，不可再修改",
            current=visit.status,
            target=VisitStatus.DOCUMENTED.value,
            entity="visit",
        )

    updates = payload.model_dump(exclude_none=True)
    for field, value in updates.items():
        setattr(visit, field, value)

    transitioned = False
    if visit.status == VisitStatus.IN_PROGRESS.value:
        visit.status = assert_transition(
            "visit", visit.status, VisitStatus.DOCUMENTED.value
        )
        transitioned = True

    await db.commit()
    await db.refresh(visit)

    logger.info(
        "病历已保存 visit=%s doctor=%s fields=%d transitioned=%s",
        visit.id,
        doctor_id,
        len(updates),
        transitioned,
    )
    return visit, transitioned


# ==================== 检查开单 ====================


async def order_exam(
    db: AsyncSession,
    visit_id: uuid.UUID,
    doctor_id: uuid.UUID,
    payload: ExamOrderRequest,
) -> ExamRequest:
    """开一张检查申请单，初始态 ordered，费用由演示价目表给出。

    只有 in_progress / documented 的 visit 可以开单（OPENABLE_VISIT_STATUSES）。
    closed 拒开的原因写在常量注释里：结案后补检应另起一次就诊，
    不能让报告挂在已归档病历下。

    这里**不使用 assert_transition**：新建一张单子不是一次状态迁移
    （没有 from 状态），正确工具是 ``validate_status_value``——
    它拦住「状态值本身写错」这类会让后续所有流转判断失效的静默错误。

    ``urgency='stat'`` 额外记一条 info 日志：这是给并行开发中的检验侧看的
    危急路径标记（同事据此知道哪些单子需要插队处理），也是运维排查
    「危急值有没有漏处理」的检索锚点。刻意**不做**函数级调用或共享状态，
    两侧只通过 exam_requests 表的 urgency 列这一个事实通信。

    fee_amount 取演示价目常量；表外类型落到 0 并打日志，交给阶段 5 收费侧核价。
    """
    visit = await _require_own_visit(db, visit_id, doctor_id)

    if visit.status not in OPENABLE_VISIT_STATUSES:
        raise InvalidStateTransitionError(
            f"当前就诊状态「{visit.status}」不允许开具检查单",
            current=visit.status,
            target=ExamRequestStatus.ORDERED.value,
            entity="visit",
        )

    fee = EXAM_FEE_TABLE_CENTS.get(payload.exam_type, DEFAULT_EXAM_FEE_CENTS)
    if payload.exam_type not in EXAM_FEE_TABLE_CENTS:
        logger.info(
            "检查项目 %s 未在演示价目表中登记，费用暂记 0 待收费侧核价",
            payload.exam_type,
        )

    request = ExamRequest(
        visit_id=visit.id,
        patient_id=visit.patient_id,
        doctor_id=visit.doctor_id,
        exam_type=payload.exam_type,
        exam_name=payload.exam_name,
        body_part=payload.body_part,
        clinical_info=payload.clinical_info,
        urgency=payload.urgency,
        status=validate_status_value("exam_request", ExamRequestStatus.ORDERED.value),
        is_charged=False,
        fee_amount=fee,
    )
    db.add(request)
    await db.commit()
    await db.refresh(request)

    if payload.urgency == Urgency.STAT.value:
        logger.info(
            "危急检查单已开出 request=%s visit=%s patient=%s 项目=%s(%s)，需检验科插队处理",
            request.id,
            visit.id,
            visit.patient_id,
            payload.exam_type,
            payload.exam_name,
        )

    logger.info(
        "检查单已开立 request=%s visit=%s type=%s fee=%d urgency=%s",
        request.id,
        visit.id,
        payload.exam_type,
        fee,
        payload.urgency,
    )
    return request


# ==================== 结束就诊 ====================


async def close_visit(
    db: AsyncSession, visit_id: uuid.UUID, doctor_id: uuid.UUID
) -> tuple[Visit, Appointment]:
    """结束就诊：visit documented→closed，并连带 appointment in_progress→completed。

    **强制先写病历才能结案**（模块 docstring 第 2 条收紧点）：转换矩阵允许
    in_progress→closed 直跳，本函数刻意不放行。没有诊断结论的结案会留下一条
    无法质控的病历，医疗纠纷时医院拿不出任何诊疗依据；宁可让医生多点一次保存，
    也不能让空病历进入归档。in_progress 直接结案会得到带 current_status 的 409。

    **同事务同生同灭**（学习阶段 3「号源扣减与预约同生同灭」的写法）：
    两个实体的四列变更（visit.status/ended_at、appointment.status）
    要么一起提交要么一起回滚，任何一步失败整体 rollback。半途成功的后果
    在模块 docstring 里已对称列出，两种半态都不能自愈，因此这里
    明确写出 except 分支的 rollback —— 失败路径必须显式，
    绝不能依赖「FastAPI 关会话时顺手回滚」这种隐式行为。

    预约侧的迁移不需要额外收紧：completed 只能从 in_progress 到达，
    而能走到这一步的预约必然已被 start_consultation 改成 in_progress，
    矩阵自身就是闸门。

    返回 (visit, appointment) 让路由如实回执两侧状态，跨实体事务的可观测性
    不该只留给客户端再发两次 GET 去猜。
    """
    visit = await _require_own_visit(db, visit_id, doctor_id)

    if visit.status == VisitStatus.IN_PROGRESS.value:
        raise InvalidStateTransitionError(
            "请先完成病历书写再结束就诊",
            current=visit.status,
            target=VisitStatus.CLOSED.value,
            entity="visit",
        )

    visit.status = assert_transition("visit", visit.status, VisitStatus.CLOSED.value)
    visit.ended_at = utcnow()

    appointment = await db.get(Appointment, visit.appointment_id)
    if appointment is None:
        # 预约行不见了属于数据完整性问题（外键 ondelete=CASCADE 本该连带删掉
        # visit）。此时不能只关病历却假装成功——那会留下一条没有归宿的 closed 病历。
        raise ResourceNotFoundError("关联的预约记录不存在，无法结束就诊")

    appointment.status = assert_transition(
        "appointment",
        appointment.status,
        AppointmentStatus.COMPLETED.value,
    )

    try:
        await db.flush()
        await db.commit()
    except Exception:
        # 显式回滚：保证不会出现「病历关了但预约还在接诊中」这类半态。
        await db.rollback()
        raise

    await db.refresh(visit)
    await db.refresh(appointment)

    logger.info(
        "就诊已结束 visit=%s appointment=%s doctor=%s",
        visit.id,
        appointment.id,
        doctor_id,
    )
    return visit, appointment


# ==================== 医生工作台 ====================


async def list_pending_appointments(
    db: AsyncSession,
    doctor_id: uuid.UUID,
    day: date | None = None,
) -> tuple[list[dict[str, object]], date]:
    """医生今日待诊队列：指定日期（缺省今天）仍处于 checked_in / in_progress 的预约。

    返回 ``[{appointment_id, patient_id, time_slot, ..., visit_id, visit_status}]``
    与生效日期。活跃 visit 用一次 IN 查询批量补齐（见 ``_active_visit_id_map``），
    避免逐个预约回表。

    按 ``time_slot`` 升序而不是按创建时间：门诊的实际顺序是「上午早段优先」，
    而 text 型的 "08:00-12:00" 恰好字典序等于时间序（种子数据的时段串都补零），
    这样医生看到的队列与叫号顺序一致。

    日期参数刻意可选而不是必填：医生点开工作台想知道的就是「今天我还有几个人」，
    逼前端算日期只会多一处会算错的地方；要查历史某天则显式传 ``date``。
    """
    effective_day = day if day is not None else today_local()

    appointments = list(
        (
            await db.execute(
                select(Appointment)
                .where(
                    Appointment.doctor_id == doctor_id,
                    Appointment.appointment_date == effective_day,
                    Appointment.status.in_(PENDING_APPOINTMENT_STATUSES),
                )
                .order_by(Appointment.time_slot.asc(), Appointment.id.asc())
            )
        )
        .scalars()
        .all()
    )

    active_map = await _active_visit_id_map(db, [item.id for item in appointments])

    items: list[dict[str, object]] = []
    for item in appointments:
        visit_pair = active_map.get(item.id)
        items.append(
            {
                "appointment_id": _stringify(item.id),
                "patient_id": _stringify(item.patient_id),
                "time_slot": item.time_slot,
                "appointment_date": item.appointment_date,
                "status": item.status,
                "chief_complaint": item.chief_complaint,
                "triage_summary": item.triage_summary,
                "checked_in_at": item.checked_in_at,
                "visit_id": _stringify(visit_pair[0]) if visit_pair else None,
                "visit_status": visit_pair[1] if visit_pair else None,
            }
        )
    return items, effective_day


# ==================== 读取与视图 ====================


async def get_visit_detail(
    db: AsyncSession,
    visit_id: uuid.UUID,
    *,
    requester_user_id: uuid.UUID,
    requester_role: str,
) -> Visit:
    """按角色取病历详情：医生限本人，患者限本人，管理员按矩阵意图拒绝。

    **admin 在这里同样 404**：权限矩阵刻意不给管理员 visit/medical_record 的读写
    （rbac_matrix.ADMIN 只有系统管理权限，理由是不能篡改诊疗记录，否则审计失去意义）。
    如果本函数照某些接口的习惯「admin 一律放行」，就等于用一个函数后门
    把矩阵里那条重要的设计决策抹掉了，因此这里不放行。
    ``require_permission`` 对 admin 的依赖层放行只意味着「允许调用这个端点」，
    行级可见范围仍由本函数说了算。
    """
    visit = await _load_visit(db, visit_id)

    if requester_role == RoleName.DOCTOR.value:
        if visit.doctor_id != requester_user_id:
            raise ResourceNotFoundError("就诊记录不存在")
        return visit

    if requester_role == RoleName.PATIENT.value:
        profile_id = await _resolve_profile_id(db, requester_user_id)
        if profile_id is None or visit.patient_id != profile_id:
            raise ResourceNotFoundError("就诊记录不存在")
        return visit

    raise ResourceNotFoundError("就诊记录不存在")


async def list_my_visits(
    db: AsyncSession,
    *,
    doctor_id: uuid.UUID,
    status_filter: str | None = None,
    patient_id: uuid.UUID | None = None,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[Visit], int]:
    """医生分页列出自己的就诊记录（可按状态、按患者筛）。

    doctor_id 恒作为过滤条件，不接受「查看全部医生病历」的入参——
    跨医生的病历检索属于病案管理范畴，矩阵里没有这个动作，
    本批次也不该开这个口子。

    count 用子查询包一层而非 ``func.count()`` 直接聚合，与
    ``knowledge_service.list_documents`` 同一做法：条件里含 OR 时
    某些版本生成的 count 会漏 WHERE，导致 total 恒等于全表行数。
    """
    conditions = [Visit.doctor_id == doctor_id]
    if status_filter:
        conditions.append(Visit.status == status_filter)
    if patient_id is not None:
        conditions.append(Visit.patient_id == patient_id)

    stmt = select(Visit).where(*conditions)
    total = int((await db.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one())

    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    items = list(
        (
            await db.execute(
                stmt.order_by(Visit.started_at.desc(), Visit.id)
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
        )
        .scalars()
        .all()
    )
    return items, total


async def list_patient_visits(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    include_closed_only: bool = False,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[Visit], int]:
    """患者视角：只看自己档案下的病历。

    患者身份必须经 ``user_id → patient_profiles.id`` 换算（外键指向档案而非账号）。
    查不到档案时报 404 而不是 500：这类账号通常是管理员手工建的患者号，
    让外键在查询里炸成 500 会把「你还没建档」这件业务事实伪装成服务器故障。

    ``include_closed_only`` 用于患者端「历史病历」页（只看已结束的），
    默认 False 时连正在进行的也一并展示——患者有权知道自己现在在看诊。
    """
    profile_id = await _resolve_profile_id(db, user_id)
    if profile_id is None:
        raise ResourceNotFoundError("当前账号尚未建立患者健康档案，无法查看就诊记录")

    conditions = [Visit.patient_id == profile_id]
    if include_closed_only:
        conditions.append(Visit.status == VisitStatus.CLOSED.value)

    stmt = select(Visit).where(*conditions)
    total = int((await db.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one())

    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    items = list(
        (
            await db.execute(
                stmt.order_by(Visit.started_at.desc(), Visit.id)
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
        )
        .scalars()
        .all()
    )
    return items, total


async def list_prior_visits(
    db: AsyncSession, patient_profile_id: uuid.UUID, *, limit: int = PRIOR_VISITS_LIMIT
) -> list[Visit]:
    """同一患者的既往病历摘要（开始接诊时带给医生看）。

    排除 closed 之外的记录？不排除——医生恰恰也要看到「上一次没看完」的那次，
    因此这里按开始时间倒序取最近若干条，由医生自己判断相关性。
    """
    rows = (
        await db.execute(
            select(Visit)
            .where(Visit.patient_id == patient_profile_id)
            .order_by(Visit.started_at.desc(), Visit.id)
            .limit(max(1, min(limit, 50)))
        )
    ).scalars().all()
    return list(rows)


async def list_visit_exam_requests(
    db: AsyncSession, visit_id: uuid.UUID, requester_role: str, requester_user_id: uuid.UUID
) -> list[ExamRequest]:
    """列出某次就诊的检查单。

    权限映射用 exam_request:read（医生、患者持有）。lab_tech 持有的是 read_all、
    cashier 同理，因此这两个角色走到这里会在依赖层就 403，本函数不必再判；
    留一道角色兜底是为了防止日后有人给新角色开了 read 却忘了行级隔离。
    """
    visit = await _load_visit(db, visit_id)

    if requester_role == RoleName.DOCTOR.value:
        if visit.doctor_id != requester_user_id:
            raise ResourceNotFoundError("就诊记录不存在")
    elif requester_role == RoleName.PATIENT.value:
        profile_id = await _resolve_profile_id(db, requester_user_id)
        if profile_id is None or visit.patient_id != profile_id:
            raise ResourceNotFoundError("就诊记录不存在")
    else:
        raise ResourceNotFoundError("就诊记录不存在")

    rows = (
        await db.execute(
            select(ExamRequest)
            .where(ExamRequest.visit_id == visit_id)
            .order_by(ExamRequest.created_at.asc(), ExamRequest.id)
        )
    ).scalars().all()
    return list(rows)


__all__ = [
    "ACTIVE_VISIT_STATUSES",
    "DEFAULT_EXAM_FEE_CENTS",
    "EXAM_FEE_TABLE_CENTS",
    "EXAM_FEE_TABLE_NAMES",
    "OPENABLE_VISIT_STATUSES",
    "PENDING_APPOINTMENT_STATUSES",
    "PRIOR_VISITS_LIMIT",
    "close_visit",
    "get_visit_detail",
    "list_my_visits",
    "list_patient_visits",
    "list_pending_appointments",
    "list_prior_visits",
    "list_visit_exam_requests",
    "order_exam",
    "save_medical_record",
    "serialize_exam_request",
    "serialize_visit",
    "start_consultation",
    "today_local",
]
