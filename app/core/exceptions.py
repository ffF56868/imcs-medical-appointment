"""IMCS 统一异常与错误码。

业务异常携带机器可读的 ``code``，前端据此区分提示方式：
「号源已满」应引导换时段，「状态流转非法」是程序性错误，
两者的用户提示完全不同，不能都笼统返回 400。
"""

from __future__ import annotations

from fastapi import HTTPException, status


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

    def to_http_exception(self) -> HTTPException:
        """转换为 FastAPI HTTPException，错误码放在 detail.code 供前端读取。"""
        return HTTPException(
            status_code=self.status_code,
            detail={
                "code": self.code,
                "message": self.message,
                **self.detail,
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


def register_exception_handlers(app) -> None:
    """把业务异常统一转换为带错误码的 HTTP 响应。"""

    @app.exception_handler(IMCSError)
    async def _handle_imcs_error(request, exc: IMCSError):  # noqa: ARG001
        return exc.to_http_exception()
