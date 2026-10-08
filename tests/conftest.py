from collections.abc import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.database import get_session
from app.legacy.models import User
from app.routers import (
    ask,
    auth,
    conditions,
    conversations,
    documents,
    drugs,
    evaluation,
    feedback,
    knowledge,
    monitoring,
    memory,
)


@pytest.fixture
def test_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def client(test_engine) -> Generator[TestClient, None, None]:
    test_app = FastAPI()
    test_app.include_router(conditions.router)
    test_app.include_router(drugs.router)
    test_app.include_router(ask.router)
    test_app.include_router(knowledge.router)
    test_app.include_router(conversations.router)
    test_app.include_router(documents.router)
    test_app.include_router(feedback.router)
    test_app.include_router(evaluation.router)
    test_app.include_router(monitoring.router)
    test_app.include_router(memory.router)
    test_app.include_router(auth.router)

    def override_get_session():
        with Session(test_engine) as session:
            yield session

    test_app.dependency_overrides[get_session] = override_get_session
    test_app.dependency_overrides[auth.get_current_user] = lambda: User(
        id=1,
        account="test-admin@example.com",
        is_admin=True,
    )
    with TestClient(test_app) as test_client:
        yield test_client


@pytest.fixture
def auth_client(test_engine) -> Generator[TestClient, None, None]:
    test_app = FastAPI()
    test_app.include_router(conditions.router)
    test_app.include_router(drugs.router)
    test_app.include_router(ask.router)
    test_app.include_router(knowledge.router)
    test_app.include_router(conversations.router)
    test_app.include_router(documents.router)
    test_app.include_router(feedback.router)
    test_app.include_router(evaluation.router)
    test_app.include_router(monitoring.router)
    test_app.include_router(memory.router)
    test_app.include_router(auth.router)

    def override_get_session():
        with Session(test_engine) as session:
            yield session

    test_app.dependency_overrides[get_session] = override_get_session
    with TestClient(test_app) as test_client:
        yield test_client
