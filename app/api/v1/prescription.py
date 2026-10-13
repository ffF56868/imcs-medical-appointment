"""处方路由：草稿、确认开具、作废与多角色读取（阶段 5 处方侧）。

路由层保持薄：解析请求 → 调 ``prescription_service`` → 组装响应。
本模块出现的业务判断只有「路径参数是不是合法 UUID」（协议层）。

============================ 权限映射（rbac_matrix 为准，未改动矩阵） ============================

+---------------------------------------------+--------------------------------------+------------------+
| 端点                                        | require_permission                   | 实际可访问角色     |
+---------------------------------------------+--------------------------------------+------------------+
| POST /prescriptions                         | prescription:create                  | 医生              |
| PUT  /prescriptions/{id}                    | prescription:update                  | 医生              |
| POST /prescriptions/{id}/confirm            | prescription:update                  | 医生              |
| PATCH /prescriptions/{id}/cancel            | prescription:update                  | 医生              |
| GET  /prescriptions                         | prescription:read 或 read_all        | 全部四类          |
| GET  /prescriptions/{id}                    | prescription:read 或 read_all        | 全部四类          |
| GET  /prescriptions/by-visit/{visit_id}     | prescription:read                    | 医生/患者         |
+---------------------------------------------+--------------------------------------+------------------+

**路径命名说明**：就诊维度的读端点注册为 ``/prescriptions/by-visit/{visit_id}``
而不是更自然的 ``/prescriptions/visits/{visit_id}``，原因见该端点 docstring ——
后者会被 ``GET /{prescription_id}`` 抢先匹配。

**药师刻意没有写入口**：矩阵注释原话「不给 prescription:update——药师不能改医生开的
处方，有问题应驳回让医生重开，这是处方权的边界」。因此本模块不出现任何
挂在 pharmacist 上的写端点，药师的审核动作属发药域（dispensing:*）。

**为什么 confirm / cancel 复用 update 而不是新增动作**：矩阵的动作清单里没有
confirm / cancel 这类细粒度动作，而两者本质都是「把 prescription 这条记录往前推一格」，
与 visit 侧把 close 落在 visit:update 上是同一个授权位。为此新增动作要同时改
RESOURCES/ACTIONS/ROLE_PERMISSIONS/种子四处，收益只是命名更精确，不值得在本批次动权限矩阵。

**读端点必须用 read/read_all 双门**：矩阵里患者/医生持 ``prescription:read``，
pharmacist/cashier **只有 read_all**。若详情接口按单数 read 把关，药师核对待发药处方
会直接 403；反之若只挂 read_all，患者就看不到自己的处方。与 exam.py 的
``_gate_any`` 是同一处理，两个端点各自独立成路径而不是在一个端点里判角色名 ——
「靠角色名硬编码」正是 rbac_matrix 模块 docstring 要避免的事。

**跨行可见性不在这里控制**：权限矩阵只回答「能不能做这个动作」，
「能看到哪几行」由 service 按角色强制（医生限自己 doctor_id、患者限自己档案），
越权一律 404 而非 403，不泄露「这张处方存在」。
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    CurrentUser,
    DbSession,
    _load_role_permissions,
    get_role_name,
    require_permission,
)
from app.core.exceptions import PermissionDeniedError, ResourceNotFoundError
from app.models.auth import User
from app.models.pharmacy import Prescription
from app.schemas.prescription import (
    PrescriptionCancelRequest,
    PrescriptionCancelResponse,
    PrescriptionConfirmRequest,
    PrescriptionConfirmResponse,
    PrescriptionCreateRequest,
    PrescriptionListResponse,
    PrescriptionOut,
    PrescriptionUpdateRequest,
)
from app.services import prescription_service as svc

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prescriptions", tags=["处方"])

# 集中定义鉴权闸门：同一个动作在不同端点上权限是否一致，只看这一处就能确认。
# ``require_permission`` 每次调用都返回一个新闭包，而 FastAPI 的「同一请求内相同
# 依赖只解析一次」按依赖对象身份判定，因此工厂结果先提成模块级私有变量再挂进
# Annotated（与 consult.py / exam.py 同口径）。签名里写别名本身没问题——
# 出问题的是把别名再当成 ``Depends(...)`` 的参数二次包裹，那会让 FastAPI
# 把一个已含依赖的 Annotated 当作待注入可调用对象解析，端点凭空多出必填参数。
_PermissionDeps: dict[tuple[str, str], Any] = {}


def _gate(resource: str, action: str):
    """按 (资源, 动作) 缓存 require_permission 闭包，保证同请求内只解析一次。"""
    key = (resource, action)
    if key not in _PermissionDeps:
        _PermissionDeps[key] = require_permission(resource, action)
    return _PermissionDeps[key]


def _gate_any(resource: str, *actions: str):
    """组合权限门：持有任一动作即放行，全都没有才 403。

    处方读端点必须用它：患者/医生持 read，pharmacist/cashier 持 read_all，
    单一动作位必然漏掉一半人（详见模块 docstring 的权限表）。
    """

    async def checker(current_user: CurrentUser, db: DbSession) -> User:
        role_name = get_role_name(current_user)
        if role_name == "admin":
            return current_user
        permissions = await _load_role_permissions(db, current_user.role_id)
        if any((resource, action) in permissions for action in actions):
            return current_user
        raise PermissionDeniedError(f"缺少权限：{resource}:{'/'.join(actions)} 任一")

    return checker


CanCreatePrescription = Annotated[User, Depends(_gate("prescription", "create"))]
CanUpdatePrescription = Annotated[User, Depends(_gate("prescription", "update"))]
CanReadPrescription = Annotated[User, Depends(_gate("prescription", "read"))]
CanReadAnyPrescription = Annotated[
    User, Depends(_gate_any("prescription", "read", "read_all"))
]


def _parse_uuid(value: str, what: str) -> uuid.UUID:
    """路径/查询参数转 UUID，非法值报 404（与 appointment/consult/exam 同一口径）。

    对客户端而言「ID 格式不对」和「这条记录不存在」是同一种失败——它拿不到资源，
    统一成 404 让前端错误分支只处理一种情况。
    """
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError) as exc:
        raise ResourceNotFoundError(f"{what}不存在") from exc


# ==================== 序列化 ====================


async def to_prescription_out(db: AsyncSession, rx: Prescription) -> PrescriptionOut:
    """ORM → 响应模型（序列化逻辑在 service，路由只做装配）。"""
    return PrescriptionOut(**await svc.serialize_prescription(db, rx))


# ==================== 草稿 ====================


@router.post(
    "",
    response_model=PrescriptionOut,
    status_code=status.HTTP_201_CREATED,
    summary="创建处方草稿",
)
async def create_prescription(
    payload: PrescriptionCreateRequest,
    db: DbSession,
    current_user: CanCreatePrescription,
) -> PrescriptionOut:
    """医生为名下的一次有效就诊开一张 draft 处方。

    前置：visit 必须属于当前医生且未 closed。closed 之后开药会得到 409——
    处方挂在已归档病历下会被质控当成漏诊处理，正确做法是另起一次就诊。
    越权访问别人的 visit 得到 404（不泄露「这条病历存在」）。

    **草稿不做安全校验**：草稿是医生的工作台便签，此刻拦住只会打断思路；
    合理用药校验发生在「确认开具」这一有法律效力的动作上。

    明细至少一条、同一处方内 drug_code 唯一（协议层与服务层双重拦截），
    违反得到 422 而不是撞数据库唯一约束冒成的 500。

    金额由服务端按明细算出，不接受客户端传入 total_amount：
    否则收费侧拿到的就是一个可以被前端随意改写的数字。
    """
    rx = await svc.create_draft(db, current_user.id, payload)
    logger.info("处方草稿创建 user_id=%s rx_id=%s", current_user.id, rx.id)
    return await to_prescription_out(db, rx)


@router.put(
    "/{prescription_id}",
    response_model=PrescriptionOut,
    summary="修改处方草稿（整体替换明细）",
)
async def update_prescription(
    prescription_id: str,
    payload: PrescriptionUpdateRequest,
    db: DbSession,
    current_user: CanUpdatePrescription,
) -> PrescriptionOut:
    """改草稿。仅 draft 可改，非 draft 返回 409。

    用 PUT 且**要求全量提交 items**：语义是「把这张草稿的明细换成这一份」，
    幂等且可安全重试。不提供逐行增删的三个端点——草稿阶段的改药在临床上就是
    「重开一张单子」，逐行 PATCH 会让前端自己维护行 id 集合，
    也让「明细至少一条」这条不变量在多次调用之间无法保证。

    药师走到这里会在依赖层 403：矩阵刻意不给 pharmacist prescription:update。
    """
    rx = await svc.update_draft(
        db, _parse_uuid(prescription_id, "处方"), current_user.id, payload
    )
    return await to_prescription_out(db, rx)


# ==================== 确认开具 ====================


@router.post(
    "/{prescription_id}/confirm",
    response_model=PrescriptionConfirmResponse,
    summary="确认开具（执行合理用药校验）",
)
async def confirm_prescription(
    prescription_id: str,
    db: DbSession,
    current_user: CanUpdatePrescription,
    payload: PrescriptionConfirmRequest | None = None,
) -> PrescriptionConfirmResponse:
    """draft → confirmed，跑三条纯规则并把结果写入 safety_checks。

    **校验规则**（演示级规则引擎，见 service 模块 docstring）：
    - 过敏史冲突（药名/编码命中患者档案过敏原）→ block
    - 重复用药（同通用名不同厂牌/剂型）→ block
    - 单日剂量超限（频次 × 单次剂量 vs 规则上限）→ warning，不阻断

    **医师处方权优先**：存在阻断项时默认 422 拒绝
    （code 为 ``prescription_allergy_conflict`` / ``prescription_drug_interaction``，
    detail 带 findings 与 ``override_required=true``）。医生若已在临床上复核过风险
    （过敏史记录有误、脱敏治疗期间仍需用药等），可携带非空
    ``safety_override_reason`` 强行开具，理由落库供药师审核与事后追责。
    被压过的风险项仍然完整留在 safety_checks 里——override 改变的是能否开具，
    不是记录是否存在。

    响应 200 而非 201：这次调用推进了已有记录的状态，没有新建资源。
    已是 confirmed 再次确认得到 409，不做静默幂等——重复开具意味着重复进入收费流程。
    """
    rx, overridden = await svc.confirm_prescription(
        db, _parse_uuid(prescription_id, "处方"), current_user.id, payload
    )
    logger.info(
        "处方确认开具 user_id=%s rx_id=%s overridden=%s", current_user.id, rx.id, overridden
    )
    return PrescriptionConfirmResponse(
        prescription=await to_prescription_out(db, rx), overridden=overridden
    )


# ==================== 作废 ====================


@router.patch(
    "/{prescription_id}/cancel",
    response_model=PrescriptionCancelResponse,
    summary="作废处方（draft/confirmed 可作废）",
)
async def cancel_prescription(
    prescription_id: str,
    db: DbSession,
    current_user: CanUpdatePrescription,
    payload: PrescriptionCancelRequest | None = None,
) -> PrescriptionCancelResponse:
    """作废一张还没缴费的处方。

    **paid 不可取消**：转换矩阵里 paid 的后继只有 dispensed。缴费后的撤药属于退费流程，
    要冲正收费单并可能追回已发出的药，那是收费域的动作；在这里开口子会留下一张
    「已收钱却被作废」的处方。非法起点由 assert_transition 自然 409。

    用 PATCH 且返回响应体而不是 204：作废后前端要立刻确认状态与时间戳都落地了。
    空请求体合法（不带原因的作废同样成立，例如医生发现自己开错了病人）。
    """
    rx = await svc.cancel_prescription(
        db,
        _parse_uuid(prescription_id, "处方"),
        current_user.id,
        payload.reason if payload else None,
    )
    return PrescriptionCancelResponse(prescription=await to_prescription_out(db, rx))


# ==================== 读取 ====================


@router.get(
    "",
    response_model=PrescriptionListResponse,
    summary="处方列表（按角色隔离：医生看自己开的，药师/收费员看全部，患者看自己的）",
)
async def list_prescriptions(
    db: DbSession,
    current_user: CanReadAnyPrescription,
    status_filter: Annotated[
        str | None,
        Query(alias="status", description="draft/confirmed/paid/dispensed/cancelled"),
    ] = None,
    visit_id: Annotated[str | None, Query(description="按就诊过滤")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> PrescriptionListResponse:
    """一个端点服务四种视角，行级过滤在 service 里按角色强制。

    不拆 /my-prescriptions 与 /all-prescriptions：URL 表达的是「谁的数据」，
    而这个信息已经在 token 里；拆两个路径只会让前端先判角色再决定调哪个接口，
    反而多一处会写错的地方。这与 consult.py 把医生/患者的病历列表合成一个端点同源。

    患者传别人的 visit_id 也查不出东西：service 恒注入 patient_id 条件，
    ``visit_id`` 对它只做收窄而不是覆盖。
    """
    rows, total = await svc.list_prescriptions(
        db,
        requester_user_id=current_user.id,
        requester_role=get_role_name(current_user),
        status_filter=status_filter,
        visit_id=_parse_uuid(visit_id, "就诊记录") if visit_id else None,
        page=page,
        page_size=page_size,
    )
    return PrescriptionListResponse(
        items=[await to_prescription_out(db, rx) for rx in rows],
        total=total,
        page=max(1, page),
        page_size=max(1, min(page_size, 100)),
    )


@router.get(
    "/{prescription_id}",
    response_model=PrescriptionOut,
    summary="处方详情（含安全校验结果）",
)
async def get_prescription_detail(
    prescription_id: str,
    db: DbSession,
    current_user: CanReadAnyPrescription,
) -> PrescriptionOut:
    """取单张处方。越权访问返回 404 而非 403（不泄露「这张处方存在」）。

    ``safety_checks`` / ``blocked_count`` / ``warning_count`` /
    ``safety_override_reason`` 一并返回：药师审核时要看得见医生忽略了什么风险、
    有没有写过 override 理由。管理员同样 404（矩阵刻意不给 admin 临床记录读写）。
    """
    rx = await svc.get_prescription_detail(
        db,
        _parse_uuid(prescription_id, "处方"),
        requester_user_id=current_user.id,
        requester_role=get_role_name(current_user),
    )
    return await to_prescription_out(db, rx)


@router.get(
    "/by-visit/{visit_id}",
    response_model=list[PrescriptionOut],
    summary="某次就诊下的全部处方",
)
async def list_visit_prescriptions(
    visit_id: str,
    db: DbSession,
    current_user: CanReadPrescription,
) -> list[PrescriptionOut]:
    """病历页展开用：列出这一次就诊开的所有处方（可能不止一张）。

    挂 ``prescription:read``（医生、患者持有）而不是 read_all：
    这是「某一次就诊」的视图，药师的全域待办走 ``GET /prescriptions``。

    **路径为什么是 /by-visit 而不是 /visits/{visit_id}**：本路由同时存在
    ``GET /{prescription_id}``，而 FastAPI 按注册顺序匹配——``/visits/<uuid>``
    的第一段会被当成 prescription_id 吃掉，请求永远进不到这个端点。
    带字面量前缀的 by-visit 与单段参数路径不可能重叠。

    归属校验复用 service 的角色分流（而不是逐张调详情做 N+1 校验）：
    先取「该角色 + 该 visit」的总数，为 0 就 404 —— 越权者与「这次就诊确实没开药」
    在这里不可区分，正是我们想要的效果；行级隔离逻辑只有一份。
    """
    parsed = _parse_uuid(visit_id, "就诊记录")
    rows, total = await svc.list_prescriptions(
        db,
        requester_user_id=current_user.id,
        requester_role=get_role_name(current_user),
        visit_id=parsed,
        page=1,
        # 上限取 schema 里的 MAX_ITEMS 同量级：一个 visit 下的处方数远小于此，
        # 传 100（service 允许的最大页长）足够把可见行一次取完。
        page_size=100,
    )
    if total == 0 and not rows:
        raise ResourceNotFoundError("就诊记录不存在")
    return [await to_prescription_out(db, rx) for rx in rows]


__all__ = ["router"]
