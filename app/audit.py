from fastapi import Request
from sqlmodel import Session

from app.database import engine
from app.legacy.models import SecurityAuditLog, User


def record_admin_request(
    user: User,
    request: Request,
    status_code: int,
) -> None:
    """Persist an admin mutation without exposing audit failures to users."""
    try:
        with Session(engine) as session:
            session.add(
                SecurityAuditLog(
                    user_id=user.id,
                    account=user.account,
                    method=request.method,
                    path=request.url.path,
                    status_code=status_code,
                )
            )
            session.commit()
    except Exception:
        return
