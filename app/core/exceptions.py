"""IMCS 统一异常与错误码。

业务异常携带机器可读的 ``code``，前端据此区分提示方式：
「号源已满」应引导换时段，「状态流转非法」是程序性错误，
两者的用户提示完全不同，不能都笼统返回 400。
"""

from __future__ import annotations

from fastapi import status
from fastapi.responses import JSONResponse


class ErrorCode:
    """业务错误码常量。命名规则：域_具体错误。"""

    # 认证与权限
    AUTH_INVALID_CREDENTIALS = "auth_invalid_credentials"
    AUTH_TOKEN_EXPIRED = "auth_token_expired"
    AUTH_TOKEN_INVALID = "auth_token_invalid"
    AUTH_ACCOUNT_LOCKED = "auth_account_locked"
    AUTH_FORBIDDEN = "auth_forbidden"

    # 挂号与号源
    APPOINTMENT_SLOT_FULL = "appointment_slot_full"
    APPOINTMENT_SLOT_CONFLICT = "appointment_slot_conflict"
    APPOINTMENT_DUPLICATE = "appointment_duplicate"
    SCHEDULE_NOT_OPEN = "schedule_not_open"

    # 业务状态机
    STATE_TRANSITION_INVALID = "state_transition_invalid"

    # 处方与用药安全
    PRESCRIPTION_DRUG_INTERACTION = "prescription_drug_interaction"
    PRESCRIPTION_ALLERGY_CONFLICT = "prescription_allergy_conflict"

    # AI 与 LLM
    LLM_ALL_PROVIDERS_FAILED = "llm_all_providers_failed"
    LLM_TOKEN_BUDGET_EXCEEDED = "llm_token_budget_exceeded"
    LLM_CONTENT_BLOCKED = "llm_content_blocked"
    LLM_STRUCTURED_OUTPUT_INVALID = "llm_structured_output_invalid"
    AI_TASK_NOT_FOUND = "ai_task_not_found"

    # 异步任务派发
    TASK_ALREADY_RUNNING = "task_already_running"
    TASK_QUEUE_UNAVAILABLE = "task_queue_unavailable"

    # 检索
    RETRIEVAL_NO_RESULT = "retrieval_no_result"

    # 通用
    RESOURCE_NOT_FOUND = "resource_not_found"


class IMCSError(Exception):
    """所有业务异常的基类。"""

    code: str = "internal_error"
    status_code: int = status.HTTP_400_BAD_REQUEST
    default_message: str = "请求处理失败"

    def __init__(self, message: str | None = None, *, detail: dict | None = None) -> None:
        self.message = message or self.default_message
        self.detail = detail or {}
        super().__init__(self.message)

    def to_response(self) -> JSONResponse:
        """转换为可直接作为 ASGI 响应的 ``JSONResponse``。

        错误码放在 ``detail.code``，前端据此做分支处理（如 401 跳登录页）。

        **为什么返回 JSONResponse 而不是 HTTPException**：Starlette 的
        ``@app.exception_handler`` 会把处理器的返回值当作 ASGI 应用直接调用
        （``await response(scope, receive, send)``）。HTTPException 是普通异常
        对象而非 Response，返回它会在响应阶段抛 TypeError，被外层兜底成
        500——结果是所有业务异常（401/403/409/429）统统退化成
        「服务器内部错误」，错误码也一并丢失。
        """
        return JSONResponse(
            status_code=self.status_code,
            content={
                "detail": {
                    "code": self.code,
                    "message": self.message,
                    **self.detail,
                }
            },
        )


class AuthenticationError(IMCSError):
    code = ErrorCode.AUTH_INVALID_CREDENTIALS
    status_code = status.HTTP_401_UNAUTHORIZED
    default_message = "账号或密码不正确"


class TokenExpiredError(IMCSError):
    code = ErrorCode.AUTH_TOKEN_EXPIRED
    status_code = status.HTTP_401_UNAUTHORIZED
    default_message = "登录已过期，请重新登录"


class TokenInvalidError(IMCSError):
    code = ErrorCode.AUTH_TOKEN_INVALID
    status_code = status.HTTP_401_UNAUTHORIZED
    default_message = "登录凭证无效"


class AccountLockedError(IMCSError):
    code = ErrorCode.AUTH_ACCOUNT_LOCKED
    status_code = status.HTTP_423_LOCKED
    default_message = "账号已被临时锁定，请稍后再试"


class PermissionDeniedError(IMCSError):
    code = ErrorCode.AUTH_FORBIDDEN
    status_code = status.HTTP_403_FORBIDDEN
    default_message = "当前角色无权执行该操作"


class ResourceNotFoundError(IMCSError):
    code = ErrorCode.RESOURCE_NOT_FOUND
    status_code = status.HTTP_404_NOT_FOUND
    default_message = "请求的资源不存在"


class SlotFullError(IMCSError):
    """号源已满。前端应引导患者改选其他时段，而非报错。"""

    code = ErrorCode.APPOINTMENT_SLOT_FULL
    status_code = status.HTTP_409_CONFLICT
    default_message = "该时段号源已挂满，请选择其他时段"


class SlotConflictError(IMCSError):
    """乐观锁版本冲突。通常由并发抢号引起，可提示重试。"""

    code = ErrorCode.APPOINTMENT_SLOT_CONFLICT
    status_code = status.HTTP_409_CONFLICT
    default_message = "号源状态已变更，请刷新后重试"


class DuplicateAppointmentError(IMCSError):
    code = ErrorCode.APPOINTMENT_DUPLICATE
    status_code = status.HTTP_409_CONFLICT
    default_message = "你在该时段已有预约"


class InvalidStateTransitionError(IMCSError):
    """状态机非法迁移，例如把已发药的处方改回草稿。"""

    code = ErrorCode.STATE_TRANSITION_INVALID
    status_code = status.HTTP_409_CONFLICT
    default_message = "当前状态不允许该操作"

    def __init__(
        self,
        message: str | None = None,
        *,
        current: str | None = None,
        target: str | None = None,
        entity: str | None = None,
    ) -> None:
        detail = {}
        if current is not None:
            detail["current_status"] = current
        if target is not None:
            detail["target_status"] = target
        if entity is not None:
            detail["entity"] = entity
        super().__init__(message, detail=detail)


class AllergyConflictError(IMCSError):
    """处方药物与患者过敏史冲突，属于用药安全拦截。"""

    code = ErrorCode.PRESCRIPTION_ALLERGY_CONFLICT
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    default_message = "处方药物与患者过敏史冲突"


class DrugInteractionError(IMCSError):
    code = ErrorCode.PRESCRIPTION_DRUG_INTERACTION
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    default_message = "处方存在药物相互作用风险"


class LLMError(IMCSError):
    code = ErrorCode.LLM_ALL_PROVIDERS_FAILED
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_message = "AI 服务暂时不可用"


class TokenBudgetExceededError(IMCSError):
    code = ErrorCode.LLM_TOKEN_BUDGET_EXCEEDED
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    default_message = "AI 调用额度已用完"


class ContentBlockedError(IMCSError):
    code = ErrorCode.LLM_CONTENT_BLOCKED
    status_code = status.HTTP_451_UNAVAILABLE_FOR_LEGAL_REASONS
    default_message = "内容未通过安全审核"


class StructuredOutputError(IMCSError):
    """LLM 返回的 JSON 两次都无法通过 Pydantic 校验。"""

    code = ErrorCode.LLM_STRUCTURED_OUTPUT_INVALID
    status_code = status.HTTP_502_BAD_GATEWAY
    default_message = "AI 返回格式异常"


class DuplicateTaskError(IMCSError):
    """同类异步任务已在执行中，拒绝重复派发。

    用 409 而不是 400：客户端的请求本身合法，冲突在于服务端已有一个
    同型任务占位。知识库重建会「删旧切块再建新切块」，两个重建并发执行
    会交错删除，留下半套向量——所以这里必须硬拒而不是排队等第二个。
    ``task_id`` 带回已有任务 ID，前端可直接跳去查看它的进度而不必自己再查一次。
    """

    code = ErrorCode.TASK_ALREADY_RUNNING
    status_code = status.HTTP_409_CONFLICT
    default_message = "已有同类任务正在执行，请等待完成后再试"

    def __init__(
        self,
        message: str | None = None,
        *,
        task_type: str | None = None,
        task_id: str | None = None,
    ) -> None:
        detail: dict[str, object] = {}
        if task_type is not None:
            detail["task_type"] = task_type
        if task_id is not None:
            detail["task_id"] = task_id
        super().__init__(message, detail=detail or None)


class TaskQueueUnavailableError(IMCSError):
    """Celery broker 不可用，任务未能派发。

    用 503 而非 500：这是依赖暂时不可用，重试可能成功，不该记成程序错误。
    业务数据此时已经落库（文档写进去了，只是没索引），因此不能吞掉异常假装成功——
    用户必须知道「资料存下了但还检索不到」。
    """

    code = ErrorCode.TASK_QUEUE_UNAVAILABLE
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_message = "任务队列暂时不可用，请稍后重试"


def register_exception_handlers(app) -> None:
    """把业务异常统一转换为带错误码的 JSON 响应。

    必须返回 Response 实例而非 HTTPException：见 ``IMCSError.to_response``
    的说明。此前返回 HTTPException 会让所有业务异常在响应阶段二次抛错，
    被兜底处理器吞成 500，前端拿不到 code 也就无法区分「密码错误」
    与「账号被锁定」。
    """

    @app.exception_handler(IMCSError)
    async def _handle_imcs_error(request, exc: IMCSError) -> JSONResponse:  # noqa: ARG001
        return exc.to_response()
