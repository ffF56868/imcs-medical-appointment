"""认证相关的 Pydantic v2 协议模型。

设计原则：**本模块不实现任何校验规则**，账号归一化与密码强度一律调用
``app.core.security`` 里的 ``normalize_account`` / ``validate_password``。
旧项目里同一套手机号正则在 schema 和 router 各写了一遍，改一次要同步两处，
很容易出现「注册时放过、登录时拒绝」的分裂行为，这里从源头避免。

Pydantic v2 用 ``field_validator(mode="after")`` 而非 v1 的 ``@validator``；
``mode="before"`` 用于把 None 归一化成空串，避免每个字段都写 None 分支。
"""

from __future__ import annotations

import re
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.security import (
    PASSWORD_MAX_LENGTH,
    PASSWORD_MIN_LENGTH,
    normalize_account,
    validate_password,
)

# 用户名：长度 3-50，字母数字下划线连字符，必须以字母或数字开头。
# 只约束字符集，不含语义，因此留在 schema 层；账号（手机/邮箱）的正则才属于 security。
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,49}$")


class RegisterRequest(BaseModel):
    """自助注册请求。

    ``account`` 是手机号或邮箱，落库时归一化为 ``phone`` / ``email`` 列；
    ``username`` 是登录名，与 account 分开的原因是患者常用手机号注册、
    但希望用一个不暴露号码的昵称登录。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    username: str = Field(min_length=3, max_length=50, description="登录用户名")
    real_name: str = Field(min_length=1, max_length=50, description="真实姓名，就诊必须实名")
    account: str = Field(description="手机号或邮箱")
    password: str = Field(
        min_length=PASSWORD_MIN_LENGTH,
        max_length=PASSWORD_MAX_LENGTH,
        description="8-32 位，需同时含字母与数字",
    )
    gender: str | None = Field(default=None, description="M / F，可选，写入健康档案")
    birth_date: date | None = Field(default=None, description="出生日期，可选")

    @field_validator("username")
    @classmethod
    def _validate_username(cls, value: str) -> str:
        if not USERNAME_PATTERN.fullmatch(value):
            raise ValueError("用户名只能包含字母、数字、下划线和连字符，且以字母或数字开头")
        # 统一小写：username 列有唯一约束，大小写不同的两名用户
        # 在按手机号/邮箱登录时会造成歧义查询
        return value.lower()

    @field_validator("password")
    @classmethod
    def _validate_password(cls, value: str) -> str:
        return validate_password(value)

    @field_validator("account")
    @classmethod
    def _validate_account(cls, value: str) -> str:
        return normalize_account(value)

    @field_validator("gender")
    @classmethod
    def _validate_gender(cls, value: str | None) -> str | None:
        # patient_profiles.gender 上有 CHECK 约束，提前在协议层拦下来
        # 才能返回可读的 422，而不是让数据库抛 IntegrityError 变成 500
        if value is None:
            return None
        upper = value.upper()
        if upper not in {"M", "F"}:
            raise ValueError("性别只能是 M 或 F")
        return upper

    @field_validator("birth_date")
    @classmethod
    def _validate_birth_date(cls, value: date | None) -> date | None:
        if value is not None and value > date.today():
            raise ValueError("出生日期不能晚于今天")
        return value


class LoginRequest(BaseModel):
    """登录请求。

    ``account`` 三选一：手机号、邮箱或用户名。归一化失败（既不是合法手机号
    也不是合法邮箱）时按用户名处理，而不是直接报 422 —— 患者记不清自己
    当初用哪个昵称注册是常态，协议层没有立场拒绝一次合法的尝试。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    account: str = Field(min_length=1, max_length=200, description="手机号 / 邮箱 / 用户名")
    password: str = Field(min_length=1, max_length=128)

    @field_validator("account")
    @classmethod
    def _normalize_account(cls, value: str) -> str:
        try:
            return normalize_account(value)
        except ValueError:
            # 不是手机/邮箱就按用户名处理：小写化，与注册时的入库格式一致
            return value.strip().lower()


class AuthQuery(BaseModel):
    """查询参数形态的账号标识，供后续「按账号查用户」类接口复用。

    与 ``LoginRequest.account`` 的区别是这里**要求**归一化成手机/邮箱，
    因为管理员检索患者时给昵称没有意义。
    """

    account: str

    @field_validator("account")
    @classmethod
    def _validate_account(cls, value: str) -> str:
        return normalize_account(value)


class RefreshRequest(BaseModel):
    """刷新令牌请求。"""

    refresh_token: str = Field(min_length=1, description="登录时下发的刷新令牌")


class TokenPair(BaseModel):
    """令牌对。``expires_in`` 给前端做「提前 N 秒静默刷新」用。"""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="访问令牌有效期秒数")


class UserOut(BaseModel):
    """用户公开信息。

    **绝不包含 password_hash**。字段声明式列出而非 ``from_attributes`` 全量导出，
    是为了让「新增敏感列时忘记排除」变成显式错误——全量导出会静默把新列带给前端。
    """

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="用户 UUID 字符串形式")
    username: str
    real_name: str
    phone: str | None = None
    email: str | None = None
    role_id: int
    role_name: str = Field(description="角色名，来自 JWT claims 或回表 JOIN")
    department_id: str | None = None
    title: str | None = None
    specialty: str | None = None
    is_active: bool
    created_at: datetime | None = None


class LogoutRequest(BaseModel):
    """登出请求体（可整体省略）。

    ``refresh_token`` 可选：不传则只吊销当前访问令牌。前端在「退出登录」按钮里
    应始终带上，否则刷新令牌仍有效，别人拿到就能续期出新会话。
    """

    refresh_token: str | None = Field(default=None, description="一并作废的刷新令牌")


class AuthResult(TokenPair):
    """注册 / 登录响应：令牌 + 用户信息，前端一次拿全，省一次 /me 往返。"""

    user: UserOut


__all__ = [
    "RegisterRequest",
    "LoginRequest",
    "RefreshRequest",
    "LogoutRequest",
    "TokenPair",
    "AuthResult",
    "AuthQuery",
    "UserOut",
]
