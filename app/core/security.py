"""认证与密码工具。

密码哈希沿用旧项目已验证的 scrypt 方案（``app/security.py``）：
scrypt 是内存硬哈希函数，抗 GPU 暴力破解优于 bcrypt 的同类参数配置，
且是 Python 标准库自带、无额外依赖。

Token 从旧项目的「随机字符串 + SHA256 存库」换成 JWT：
- 无状态校验，不必每次请求查数据库，横向扩容时不依赖会话表
- payload 携带 role_id，权限判断可在解码后立即完成
- 代价是无法服务端主动失效单个 token，因此保留短有效期 + Redis 黑名单兜底
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import jwt

from app.core.config import settings
from app.core.exceptions import TokenExpiredError, TokenInvalidError


# ==================== 账号与密码校验 ====================

PHONE_PATTERN = re.compile(r"(?:\+?86)?1[3-9]\d{9}")
EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 32


def normalize_account(value: str) -> str:
    """把中国手机号或邮箱归一化为单一存储格式。

    手机号统一存 ``+86`` 前缀，避免同一号码因输入格式不同产生多个账号。
    """
    if not isinstance(value, str):
        raise ValueError("账号必须是有效的中国手机号或邮箱")

    compact_value = re.sub(r"[\s-]", "", value.strip())
    if PHONE_PATTERN.fullmatch(compact_value):
        if compact_value.startswith("+86"):
            return compact_value
        if compact_value.startswith("86") and len(compact_value) == 13:
            return f"+86{compact_value[2:]}"
        return f"+86{compact_value}"

    if len(value.strip()) <= 200 and EMAIL_PATTERN.fullmatch(value.strip()):
        return value.strip().lower()

    raise ValueError("账号必须是有效的中国手机号或邮箱")


def validate_password(value: str) -> str:
    """密码强度校验：8-32 位且同时包含字母和数字。"""
    if not isinstance(value, str):
        raise ValueError("密码必须是英文字母和数字的组合")
    if not PASSWORD_MIN_LENGTH <= len(value) <= PASSWORD_MAX_LENGTH:
        raise ValueError("密码长度必须是 8-32 位")
    if not re.search(r"[A-Za-z]", value):
        raise ValueError("密码至少需要包含一个英文字母")
    if not re.search(r"\d", value):
        raise ValueError("密码至少需要包含一个数字")
    return value


def hash_password(password: str) -> str:
    """scrypt 加盐哈希。输出格式 ``scrypt$n$r$p$salt$hash`` 便于将来升级参数。"""
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
    )
    encoded_salt = base64.urlsafe_b64encode(salt).decode("ascii")
    encoded_hash = base64.urlsafe_b64encode(digest).decode("ascii")
    return f"scrypt$16384$8$1${encoded_salt}${encoded_hash}"


def verify_password(password: str, stored_hash: str) -> bool:
    """校验密码。用 ``hmac.compare_digest`` 做定长比较，避免时序侧信道。"""
    try:
        algorithm, n, r, p, encoded_salt, encoded_hash = stored_hash.split("$")
        if algorithm != "scrypt":
            return False
        salt = base64.urlsafe_b64decode(encoded_salt.encode("ascii"))
        expected = base64.urlsafe_b64decode(encoded_hash.encode("ascii"))
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(n),
            r=int(r),
            p=int(p),
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError, KeyError):
        return False


# ==================== JWT ====================

TOKEN_TYPE_ACCESS = "access"
TOKEN_TYPE_REFRESH = "refresh"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def create_access_token(
    user_id: UUID,
    role_name: str,
    *,
    role_id: int | None = None,
    username: str | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """签发访问令牌。

    payload 里带 ``role``，这样权限依赖不必每次查库就能判断角色，
    只在需要精确权限矩阵时才回表查 ``role_permissions``。
    """
    now = _utcnow()
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "role": role_name,
        "type": TOKEN_TYPE_ACCESS,
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_expire_minutes),
        "jti": secrets.token_urlsafe(16),
    }
    if role_id is not None:
        payload["role_id"] = role_id
    if username is not None:
        payload["username"] = username
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(user_id: UUID) -> str:
    """签发刷新令牌。有效期更长，仅用于换发访问令牌。"""
    now = _utcnow()
    payload = {
        "sub": str(user_id),
        "type": TOKEN_TYPE_REFRESH,
        "iat": now,
        "exp": now + timedelta(days=settings.refresh_token_expire_days),
        "jti": secrets.token_urlsafe(16),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str, *, expected_type: str = TOKEN_TYPE_ACCESS) -> dict[str, Any]:
    """解码并校验 JWT。

    过期与非法分别抛不同异常，前端据此区分「请重新登录」和「凭证无效」。
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenExpiredError() from exc
    except jwt.InvalidTokenError as exc:
        raise TokenInvalidError() from exc

    if payload.get("type") != expected_type:
        raise TokenInvalidError("令牌类型不匹配")
    return payload


def get_token_expiry_seconds() -> int:
    """访问令牌有效期秒数，供前端提前刷新使用。"""
    return settings.access_token_expire_minutes * 60
