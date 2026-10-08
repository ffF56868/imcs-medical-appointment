import hashlib
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.database import get_session
from app.cache import LOGIN_RATE_LIMIT_PER_MINUTE, consume_fixed_window_limit
from app.legacy.models import LoginAttempt, SecurityAuditLog, User, UserSession
from app.legacy.schemas import (
    AuthLoginRequest,
    AuthRegisterRequest,
    AuthResponse,
    SecurityAuditLogListResponse,
    SecurityAuditLogRead,
    UserRead,
)
from app.security import (
    create_access_token,
    hash_access_token,
    hash_password,
    is_expired,
    LOGIN_LOCKOUT_MINUTES,
    MAX_ACTIVE_SESSIONS,
    MAX_LOGIN_FAILURES,
    verify_password,
)


router = APIRouter(prefix="/auth", tags=["auth"])
bearer_scheme = HTTPBearer(auto_error=False)


def to_user_read(user: User) -> UserRead:
    return UserRead.model_validate(user)


def create_auth_response(session: Session, user: User) -> AuthResponse:
    now = datetime.now(UTC)
    active_sessions: list[UserSession] = []
    for existing_session in session.exec(
        select(UserSession)
        .where(UserSession.user_id == user.id)
        .order_by(UserSession.created_at)
    ).all():
        if existing_session.revoked_at is not None or is_expired(
            existing_session.expires_at
        ):
            if existing_session.revoked_at is None:
                existing_session.revoked_at = now
                session.add(existing_session)
            continue
        active_sessions.append(existing_session)

    # Keep forgotten or stolen tokens from remaining usable indefinitely.
    sessions_to_revoke = max(0, len(active_sessions) - MAX_ACTIVE_SESSIONS + 1)
    for old_session in active_sessions[:sessions_to_revoke]:
        old_session.revoked_at = now
        session.add(old_session)

    access_token, expires_at = create_access_token()
    session.add(
        UserSession(
            user_id=user.id,
            token_hash=hash_access_token(access_token),
            expires_at=expires_at,
        )
    )
    session.commit()
    return AuthResponse(
        access_token=access_token,
        token_type="bearer",
        expires_at=expires_at,
        user=to_user_read(user),
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    session: Session = Depends(get_session),
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="请先登录",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_session = session.exec(
        select(UserSession).where(
            UserSession.token_hash == hash_access_token(credentials.credentials)
        )
    ).first()
    if (
        user_session is None
        or user_session.revoked_at is not None
        or is_expired(user_session.expires_at)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录已失效，请重新登录",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = session.get(User, user_session.user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="账号不可用",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_admin(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> User:
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限",
        )
    request.state.admin_user = current_user
    return current_user


@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    payload: AuthRegisterRequest,
    session: Session = Depends(get_session),
):
    if session.exec(select(User).where(User.account == payload.account)).first():
        raise HTTPException(status_code=409, detail="该账号已经注册")

    user = User(
        account=payload.account,
        password_hash=hash_password(payload.password),
    )
    session.add(user)
    try:
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(status_code=409, detail="该账号已经注册") from error
    session.refresh(user)
    return create_auth_response(session, user)


def _request_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _login_rate_limit_key(account: str, source_ip: str) -> str:
    identifier = f"{account}|{source_ip}".encode("utf-8")
    return f"medical-health:rate:login:{hashlib.sha256(identifier).hexdigest()}"


def _get_login_attempt(
    session: Session,
    account: str,
    source_ip: str,
) -> LoginAttempt | None:
    return session.exec(
        select(LoginAttempt).where(
            LoginAttempt.account == account,
            LoginAttempt.source_ip == source_ip,
        )
    ).first()


def _record_failed_login(
    session: Session,
    account: str,
    source_ip: str,
    attempt: LoginAttempt | None,
) -> bool:
    now = datetime.now(UTC)
    attempt = attempt or LoginAttempt(account=account, source_ip=source_ip)
    attempt.failed_count += 1
    attempt.last_failed_at = now
    is_now_locked = attempt.failed_count >= MAX_LOGIN_FAILURES
    if is_now_locked:
        attempt.locked_until = now + timedelta(minutes=LOGIN_LOCKOUT_MINUTES)
    session.add(attempt)
    session.commit()
    return is_now_locked


@router.post("/login", response_model=AuthResponse)
def login(
    payload: AuthLoginRequest,
    request: Request,
    session: Session = Depends(get_session),
):
    source_ip = _request_ip(request)
    if not consume_fixed_window_limit(
        _login_rate_limit_key(payload.account, source_ip),
        LOGIN_RATE_LIMIT_PER_MINUTE,
    ):
        raise HTTPException(
            status_code=429,
            detail="登录请求过于频繁，请稍后再试",
        )

    user = session.exec(select(User).where(User.account == payload.account)).first()
    if user is None:
        raise HTTPException(status_code=401, detail="账号或密码错误")
    if not user.is_active:
        raise HTTPException(status_code=401, detail="账号不可用")

    attempt = _get_login_attempt(session, user.account, source_ip)
    if attempt and attempt.locked_until and not is_expired(attempt.locked_until):
        raise HTTPException(
            status_code=429,
            detail=f"登录失败次数过多，请 {LOGIN_LOCKOUT_MINUTES} 分钟后再试",
        )

    if attempt and attempt.locked_until and is_expired(attempt.locked_until):
        attempt.failed_count = 0
        attempt.locked_until = None

    if not verify_password(payload.password, user.password_hash):
        locked = _record_failed_login(session, user.account, source_ip, attempt)
        if locked:
            raise HTTPException(
                status_code=429,
                detail=f"登录失败次数过多，请 {LOGIN_LOCKOUT_MINUTES} 分钟后再试",
            )
        raise HTTPException(status_code=401, detail="账号或密码错误")

    if attempt:
        attempt.failed_count = 0
        attempt.locked_until = None
        attempt.last_failed_at = None
        session.add(attempt)
    return create_auth_response(session, user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    session: Session = Depends(get_session),
):
    if credentials is None or credentials.scheme.lower() != "bearer":
        return None
    user_session = session.exec(
        select(UserSession).where(
            UserSession.token_hash == hash_access_token(credentials.credentials)
        )
    ).first()
    if user_session is not None and user_session.revoked_at is None:
        from datetime import UTC, datetime

        user_session.revoked_at = datetime.now(UTC)
        session.add(user_session)
        session.commit()
    return None


@router.get("/me", response_model=UserRead)
def me(current_user: User = Depends(get_current_user)):
    return to_user_read(current_user)


@router.get(
    "/audit-logs",
    response_model=SecurityAuditLogListResponse,
)
def audit_logs(
    _admin: User = Depends(require_admin),
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(get_session),
):
    logs = session.exec(
        select(SecurityAuditLog)
        .order_by(SecurityAuditLog.created_at.desc())
        .limit(limit)
    ).all()
    return SecurityAuditLogListResponse(
        total_count=len(logs),
        logs=[SecurityAuditLogRead.model_validate(log) for log in logs],
    )
