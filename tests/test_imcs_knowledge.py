"""知识库管理接口与向量化服务单元测试（文件型 SQLite，不依赖 PostgreSQL）。

**能跑的部分**：schemas 校验、文档 CRUD 服务函数、权限拒绝、search 端点降级、
rebuild 派发（打桩 dispatch）、任务状态机流转、content_hash 幂等判定。

**不能在这里跑的部分及原因**：

1. **真实向量化（vectorize_document）**——它必须调 embedding API。测试绝不打真实
   网络，因此本文件用假的 embed_texts 打桩覆盖「切块 → tsv → 向量 → 状态」这条
   编排链路，向量数值是伪造的，只验证写入结构与统计数字对不对。
2. **tsv 的原生 SQL**——``to_tsvector('simple', :tok)`` 是 PostgreSQL 专有函数，
   SQLite 里根本不存在。已把它收敛到 ``knowledge_service.populate_chunk_tsv``
   单个函数，并在其中按方言分支：非 PG 走普通文本 UPDATE（SQLite 上 tsv 列本就
   退化成 TEXT），因此测试能跑到「填了值」这一步，但**不验证 tsvector 语义**。
   PG 侧的真实行为由容器集成冒烟确认。
3. **advisory lock / FOR UPDATE**——SQLite 无这两个特性。
   ``acquire_dispatch_lock`` 在非 PG 上捕获异常后按「无锁」放行，
   重复派发防护在测试里只剩查重那一道闸，所以本文件的并发用例只验证查重逻辑，
   不声称验证了跨进程互斥。
4. **Celery 任务 body**——需要 broker + worker + 真库，属于容器集成范畴。
   这里只静态断言两个任务的注册名、队列与超时参数没被改坏。

Redis 沿用 auth 测试的双补丁策略（auth_service + core.cache 两处引用都要换）。
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.imcs_main import app
from app.models.auth import Permission, Role, RolePermission, User
from app.models.base import Base, utcnow
from app.models.knowledge import KnowledgeChunk, KnowledgeDocument, VectorEmbedding
from app.schemas.knowledge import (
    DocumentCreateRequest,
    DocumentUpdateRequest,
    SearchRequest,
)
from app.services import auth_service, knowledge_service as ks
from tests.imcs_sqlite import (
    build_session_factory,
    error_code_of,
    seed_role_permissions,
    seed_roles,
)

def _real(obj):
    """从可能被包属性遮蔽的对象上取回真实模块。

    app/tasks/__init__.py 的 ``from app.tasks.celery_app import celery_app``
    让 ``import app.tasks.celery_app as m`` 拿到的 m 也可能是 Celery 实例
    （``import a.b as x`` 优先 getattr(a, 'b')，被遮蔽时得到实例而非模块），
    monkeypatch 打在实例上必抛 AttributeError 或静默失效。
    sys.modules 里存的才是真模块对象——knowledge_service 在函数内
    ``from app.tasks.celery_app import ...`` 执行的正是对该模块的属性查找，
    patch 模块属性才能命中。
    """
    import sys

    mod = sys.modules.get("app.tasks.celery_app")
    if mod is None:
        raise RuntimeError("app.tasks.celery_app 尚未导入")
    return mod


PASSWORD = "Imcs2026demo"


# ==================== 建表 ====================

KNOWLEDGE_TABLES: tuple[str, ...] = (
    "roles",
    "permissions",
    "role_permissions",
    "departments",
    "users",
    "knowledge_documents",
    "knowledge_chunks",
    "vector_embeddings",
    "ai_tasks",
)


def _compile_pg_uuid() -> None:
    """PG UUID 的 SQLite DDL 编译器由 tests.imcs_sqlite 在 import 时注册。

    这里显式导入一次，确保即便本文件单独运行也已挂载该补丁。
    """
    import tests.imcs_sqlite  # noqa: F401


@pytest.fixture()
def engine(tmp_path):
    """文件型 SQLite：同步建表 + 异步被测会话共用同一份数据。

    为什么不用内存库：``sqlite://`` 内存库生命周期绑定单条连接，
    同步引擎建完表后异步引擎拿到的是另一个空库，首个查询即 no such table。
    """
    _compile_pg_uuid()
    db_path = tmp_path / "imcs_knowledge.db"
    test_engine = sa.create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    tables = [Base.metadata.tables[name] for name in KNOWLEDGE_TABLES]
    with test_engine.begin() as conn:
        for table in tables:
            table.create(conn, checkfirst=True)
        seed_roles(conn)
        # require_permission 判定依赖 role_permissions JOIN permissions，
        # 空权限表会让非 admin 一律 403（doctor 读用例实测翻车）。
        # 直接灌 rbac_matrix 的真实矩阵，测试验证的就是实际授权边界。
        seed_role_permissions(conn)
    yield test_engine
    test_engine.dispose()


@pytest.fixture()
def sync_session(engine) -> Iterator[sa.orm.Session]:
    """worker 侧同步会话，直接测 knowledge_service 的同步函数。"""
    factory = sa.orm.sessionmaker(bind=engine, class_=sa.orm.Session, expire_on_commit=False)
    session = factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
async def session_factory(engine):
    factory = build_session_factory(engine)
    yield factory
    await factory.kw["bind"].dispose()


@pytest.fixture()
async def db(session_factory) -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        yield session


# ==================== 假 Redis（沿用 auth 测试的双补丁）====================


class FakeRedis:
    """最小内存 Redis，只需覆盖 require_permission 链路上可能触碰的命令。"""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, *, nx: bool = False, ex: int | None = None) -> bool:
        self.store[key] = value
        return True

    async def delete(self, *keys: str) -> int:
        return sum(1 for key in keys if self.store.pop(key, None) is not None)

    async def exists(self, key: str) -> int:
        return 1 if key in self.store else 0

    async def incrby(self, key: str, amount: int) -> int:
        current = int(self.store.get(key, "0")) + amount
        self.store[key] = str(current)
        return current

    async def ttl(self, key: str) -> int:
        return -2

    async def expire(self, key: str, seconds: int) -> bool:
        return False

    def pipeline(self, *, transaction: bool = True) -> "_Pipeline":
        return _Pipeline(self)


class _Pipeline:
    def __init__(self, owner: FakeRedis) -> None:
        self._owner = owner
        self._commands: list[tuple[str, tuple]] = []

    def incrby(self, key: str, amount: int) -> "_Pipeline":
        self._commands.append(("incrby", (key, amount)))
        return self

    def ttl(self, key: str) -> "_Pipeline":
        self._commands.append(("ttl", (key,)))
        return self

    async def execute(self) -> list:
        results = [await getattr(self._owner, name)(*args) for name, args in self._commands]
        self._commands.clear()
        return results


@pytest.fixture()
def fake_redis() -> Iterator[FakeRedis]:
    from app.core import cache as core_cache

    client = FakeRedis()
    originals = (auth_service.cache_client, core_cache.cache_client)
    auth_service.cache_client = client
    core_cache.cache_client = client
    try:
        yield client
    finally:
        auth_service.cache_client, core_cache.cache_client = originals


# ==================== 用户与鉴权夹具 ====================


ROLE_USER_IDS: dict[str, int] = {
    "patient": 1,
    "doctor": 2,
    "lab_tech": 3,
    "cashier": 4,
    "pharmacist": 5,
    "admin": 6,
}


def _make_user(engine, role_name: str, username: str) -> uuid.UUID:
    """直插一条用户行并返回其 id，同时补齐该角色的 knowledge 权限。

    绕过注册接口是因为注册固定发 patient 角色，而本文件要覆盖
    「有读权限的医生」「无任何知识库权限的收费员」等多种组合。
    """
    user_id = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO users (id, username, password_hash, real_name, role_id, is_active)"
                " VALUES (:id, :username, :pw, :name, :role_id, 1)"
            ),
            {
                "id": user_id.hex,
                "username": username,
                "pw": "x",
                "name": username,
                "role_id": ROLE_USER_IDS[role_name],
            },
        )
    return user_id


def _grant(engine, role_id: int, resource: str, action: str) -> None:
    """给角色授权一条 (resource, action)。

    非 admin 角色走 require_permission 时必须回表查 permissions，
    所以权限拒绝用例的前提是「库里确实没有这条授权」，而不是靠默认值猜。
    """
    with engine.begin() as conn:
        existing = conn.execute(
            sa.text("SELECT id FROM permissions WHERE resource=:r AND action=:a"),
            {"r": resource, "a": action},
        ).fetchone()
        if existing is None:
            conn.execute(
                sa.text("INSERT INTO permissions (resource, action) VALUES (:r, :a)"),
                {"r": resource, "a": action},
            )
            existing = conn.execute(
                sa.text("SELECT id FROM permissions WHERE resource=:r AND action=:a"),
                {"r": resource, "a": action},
            ).fetchone()
        perm_id = existing[0]
        linked = conn.execute(
            sa.text("SELECT 1 FROM role_permissions WHERE role_id=:rid AND permission_id=:pid"),
            {"rid": role_id, "pid": perm_id},
        ).fetchone()
        if linked is None:
            conn.execute(
                sa.text("INSERT INTO role_permissions (role_id, permission_id) VALUES (:rid, :pid)"),
                {"rid": role_id, "pid": perm_id},
            )


def as_user(user_id: uuid.UUID, username: str, role_name: str, role_id: int):
    """构造一个与 deps.get_current_user 同形态的轻量 User（含 _role_name）。"""
    user = User(
        id=user_id,
        username=username,
        real_name=username,
        password_hash="",
        role_id=role_id,
        is_active=True,
    )
    user._role_name = role_name
    return user


@pytest.fixture()
def doctor_id(engine) -> uuid.UUID:
    """医生账号：矩阵里有 knowledge:read，没有 create/delete/rebuild。"""
    return _make_user(engine, "doctor", "dr_zhou")


@pytest.fixture()
def admin_id(engine) -> uuid.UUID:
    return _make_user(engine, "admin", "root_admin")


@pytest.fixture()
def cashier_id(engine) -> uuid.UUID:
    """收费员：矩阵里完全没有 knowledge 相关授权。"""
    return _make_user(engine, "cashier", "cashier_wu")


@pytest.fixture()
async def client(engine, fake_redis, session_factory, request):
    """ASGI 客户端。

    同时覆盖两个依赖：
    - ``get_db`` → 测试库会话工厂
    - ``get_current_user`` → 由用例通过 ``request.param`` 指定的身份；
      不给身份时保持未覆盖，用于验证「不带 token 也能被拒」的路径。

    为什么要覆盖 get_current_user 而不签发真 JWT：本文件关注的是知识库业务分支，
    令牌链路已由 test_imcs_auth.py 完整覆盖；重签一遍只会把两个模块的失败耦合起来。
    权限判定本身（require_permission 回表查 permissions）仍走真实代码路径。
    """
    import app.imcs_main as imcs_main_module

    async def override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    async def fake_check_redis() -> bool:
        return True

    app.dependency_overrides[get_db] = override_get_db
    identity = getattr(request, "param", None)
    if identity is not None:
        user_id, username, role_name, role_id = identity

        async def override_current_user():
            return as_user(user_id, username, role_name, role_id)

        app.dependency_overrides[get_current_user] = override_current_user

    originals = (imcs_main_module.async_session_factory, imcs_main_module.check_redis)
    imcs_main_module.async_session_factory = session_factory
    imcs_main_module.check_redis = fake_check_redis

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client

    app.dependency_overrides.clear()
    imcs_main_module.async_session_factory, imcs_main_module.check_redis = originals


# ==================== 数据助手 ====================

DOC_CONTENT = (
    "# 高血压随访要点\n\n"
    "血压测量应在安静状态下进行，连续多日偏高才考虑诊断。\n"
    "出现持续胸痛或明显气短时应当立即就医。"
)


async def seed_doc(db: AsyncSession, *, title: str = "高血压随访要点", content: str = DOC_CONTENT, **kw) -> KnowledgeDocument:
    """经 service 创建一篇文档，返回 ORM 对象。"""
    return await ks.create_document(
        db,
        title=title,
        content=content,
        doc_type=kw.pop("doc_type", "guideline"),
        source=kw.pop("source", "项目内置资料"),
        source_tier=kw.pop("source_tier", "unverified"),
        **kw,
    )


def fake_vectors(count: int, dim: int | None = None) -> list[list[float]]:
    """伪造向量，维度对齐 settings.embedding_dim 以通过校验。"""
    from app.core.config import settings

    size = dim or settings.embedding_dim
    return [[float(i + 1) / 100.0] * size for i in range(count)]


# ==================== schemas 校验（纯内存）====================


class TestSchemas:
    def test_create_request_valid(self) -> None:
        req = DocumentCreateRequest(
            title="指南A", content="正文", doc_type="guideline", source="WHO"
        )
        assert req.source_tier == "unverified", "source_tier 应有默认值"
        assert req.department_id is None

    def test_create_request_rejects_unknown_doc_type(self) -> None:
        with pytest.raises(ValueError):
            DocumentCreateRequest(
                title="指南A", content="正文", doc_type="novel_type", source="WHO"
            )

    def test_create_request_rejects_unknown_source_tier(self) -> None:
        with pytest.raises(ValueError):
            DocumentCreateRequest(
                title="指南A",
                content="正文",
                doc_type="drug",
                source="WHO",
                source_tier="definitely_wrong",
            )

    def test_create_request_rejects_bad_department_id(self) -> None:
        with pytest.raises(ValueError):
            DocumentCreateRequest(
                title="指南A",
                content="正文",
                doc_type="drug",
                department_id="not-a-uuid",
            )

    def test_create_request_accepts_uuid_department_id(self) -> None:
        did = str(uuid.uuid4())
        req = DocumentCreateRequest(
            title="指南A", content="正文", doc_type="drug", department_id=did
        )
        assert req.department_id == did

    def test_create_request_rejects_empty_content(self) -> None:
        with pytest.raises(ValueError):
            DocumentCreateRequest(title="A", content="", doc_type="drug")

    def test_update_request_partial_semantics(self) -> None:
        """未设置的字段不得出现在 model_dump(exclude_unset=True) 里。

        路由层正是靠这个把 PATCH 的空字段过滤掉；若 Pydantic 版本行为变化
        导致 None 混进 changes，会把标题更新成 NULL。
        """
        req = DocumentUpdateRequest(title="新标题")
        changes = req.model_dump(exclude_unset=True)
        assert changes == {"title": "新标题"}

    def test_search_request_requires_query(self) -> None:
        with pytest.raises(ValueError):
            SearchRequest(query="")

    def test_literals_match_service_constants(self) -> None:
        """schema 的字面量集合与 service 常量一致（import 期已断言，此处再钉一次）。"""
        from typing import get_args

        from app.schemas.knowledge import DocTypeLiteral, SourceTierLiteral

        assert set(get_args(DocTypeLiteral)) == set(ks.DOC_TYPES)
        assert set(get_args(SourceTierLiteral)) == set(ks.SOURCE_TIERS)


# ==================== content_hash 口径 ====================


class TestContentHash:
    def test_hash_matches_seed_algorithm(self) -> None:
        """与种子脚本的 _content_hash 必须同算法同输入，否则每轮都被误判为变更。"""
        import hashlib

        from app.seed.knowledge import _content_hash

        sample = "布洛芬用于退热，请遵医嘱使用。"
        assert ks.content_sha256(sample) == _content_hash(sample)
        assert ks.content_sha256(sample) == hashlib.sha256(
            sample.encode("utf-8")
        ).hexdigest()

    def test_hash_is_deterministic_and_sensitive(self) -> None:
        a = ks.content_sha256("同样内容")
        b = ks.content_sha256("同样内容")
        c = ks.content_sha256("同样内容 ")
        assert a == b
        assert a != c, "尾随空格也应改变指纹，否则增量判断会漏掉真实变更"


# ==================== 文档 CRUD（异步 service）====================


class TestDocumentCrud:
    async def test_create_sets_pending_and_hash(self, db: AsyncSession) -> None:
        doc = await seed_doc(db)
        assert doc.index_status == ks.INDEX_STATUS_PENDING
        assert doc.chunk_count == 0
        assert doc.content_hash == ks.content_sha256(DOC_CONTENT)
        assert doc.is_reviewed is False

    async def test_get_document_missing_raises_404(self, db: AsyncSession) -> None:
        from app.core.exceptions import ResourceNotFoundError

        with pytest.raises(ResourceNotFoundError):
            await ks.get_document(db, uuid.uuid4())

    async def test_list_pagination_and_total(self, db: AsyncSession) -> None:
        for i in range(3):
            await seed_doc(db, title=f"文档{i}", content=f"内容{i}")

        items, total = await ks.list_documents(db, page=1, page_size=2)
        assert total == 3, "total 必须是过滤后的匹配数，不是当前页条数"
        assert len(items) == 2

        second, total2 = await ks.list_documents(db, page=2, page_size=2)
        assert total2 == 3
        assert len(second) == 1

    async def test_list_keyword_filters_title_only_columns(self, db: AsyncSession) -> None:
        await seed_doc(db, title="糖尿病饮食", source="WHO")
        await seed_doc(db, title="高血压测量", source="NIDDK")

        items, total = await ks.list_documents(db, keyword="糖尿病")
        assert total == 1
        assert items[0].title == "糖尿病饮食"

        by_source, _ = await ks.list_documents(db, keyword="NIDDK")
        assert len(by_source) == 1

    async def test_list_excludes_inactive_by_default(self, db: AsyncSession) -> None:
        doc = await seed_doc(db, title="下线资料")
        doc.is_active = False
        await db.commit()

        _, active_total = await ks.list_documents(db)
        assert active_total == 0
        _, all_total = await ks.list_documents(db, include_inactive=True)
        assert all_total == 1

    async def test_list_filter_by_doc_type_and_status(self, db: AsyncSession) -> None:
        await seed_doc(db, title="指南甲", doc_type="guideline")
        await seed_doc(db, title="药品乙", doc_type="drug")

        drugs, n = await ks.list_documents(db, doc_type="drug")
        assert n == 1 and drugs[0].doc_type == "drug"

        pending, pn = await ks.list_documents(db, index_status=ks.INDEX_STATUS_PENDING)
        assert pn == 2
        assert len(pending) == 2

    async def test_update_content_marks_stale(self, db: AsyncSession) -> None:
        """改正文必须标 stale，否则检索会继续返回旧切块——医疗场景的临床风险。"""
        doc = await seed_doc(db)
        doc.index_status = ks.INDEX_STATUS_INDEXED
        await db.commit()

        updated = await ks.update_document(db, doc.id, content="全新内容段落")
        assert updated.index_status == ks.INDEX_STATUS_STALE
        assert updated.content_hash == ks.content_sha256("全新内容段落")

    async def test_update_metadata_only_also_marks_stale(self, db: AsyncSession) -> None:
        """标题/来源分级变了也要重建：切块上的反规范化字段会跟着过期。"""
        doc = await seed_doc(db)
        doc.index_status = ks.INDEX_STATUS_INDEXED
        await db.commit()

        updated = await ks.update_document(db, doc.id, title="改名后的标题")
        assert updated.title == "改名后的标题"
        assert updated.index_status == ks.INDEX_STATUS_STALE

    async def test_update_without_content_change_keeps_status(self, db: AsyncSession) -> None:
        doc = await seed_doc(db)
        doc.index_status = ks.INDEX_STATUS_INDEXED
        await db.commit()

        updated = await ks.update_document(db, doc.id, is_reviewed=True)
        assert updated.is_reviewed is True
        assert updated.index_status == ks.INDEX_STATUS_INDEXED, "仅审核标记不应触发重建"

    async def test_delete_removes_chunks_and_vectors(self, db: AsyncSession, engine) -> None:
        """显式清理切块与向量，不依赖数据库 FK CASCADE。

        SQLite 默认不开 PRAGMA foreign_keys，靠级联的话测试环境会留下孤儿向量，
        检索就会命中指向已删文档的引用。
        """
        doc = await seed_doc(db)
        chunk = KnowledgeChunk(
            document_id=doc.id,
            chunk_index=0,
            content="切块内容",
            content_hash=ks.content_sha256("切块内容"),
            token_count=4,
            title=doc.title,
            doc_type=doc.doc_type,
            source=doc.source,
            source_tier=doc.source_tier,
        )
        db.add(chunk)
        await db.flush()
        db.add(
            VectorEmbedding(
                chunk_id=chunk.id, embedding=[0.1] * 8, model_version="fake-model"
            )
        )
        await db.commit()
        chunk_id = chunk.id

        await ks.delete_document(db, doc.id)

        with engine.connect() as conn:
            assert conn.execute(sa.text("SELECT COUNT(*) FROM knowledge_documents")).scalar() == 0
            assert conn.execute(sa.text("SELECT COUNT(*) FROM knowledge_chunks")).scalar() == 0
            assert conn.execute(sa.text("SELECT COUNT(*) FROM vector_embeddings")).scalar() == 0
        assert chunk_id is not None


# ==================== worker 侧同步函数 ====================


class TestWorkerSide:
    def test_collect_pending_picks_pending_stale_failed(self, sync_session) -> None:
        for status_value in ("pending", "stale", "failed"):
            sync_session.add(
                KnowledgeDocument(
                    id=uuid.uuid4(),
                    title=f"文档-{status_value}",
                    doc_type="guideline",
                    content=f"内容-{status_value}",
                    content_hash=ks.content_sha256(f"内容-{status_value}"),
                    source="测试",
                    source_tier="general",
                    index_status=status_value,
                    chunk_count=0,
                    is_active=True,
                    created_at=utcnow(),
                    updated_at=utcnow(),
                )
            )
        sync_session.add(
            KnowledgeDocument(
                id=uuid.uuid4(),
                title="已索引且未变",
                doc_type="guideline",
                content="稳定内容",
                content_hash=ks.content_sha256("稳定内容"),
                source="测试",
                source_tier="general",
                index_status="indexed",
                chunk_count=1,
                is_active=True,
                created_at=utcnow(),
                updated_at=utcnow(),
            )
        )
        sync_session.commit()

        pending = ks.collect_pending_documents(sync_session)
        titles = {d.title for d in pending}
        assert titles == {"文档-pending", "文档-stale", "文档-failed"}

    def test_collect_detects_hash_drift(self, sync_session) -> None:
        """indexed 但内容被旁路改过的文档必须被捞起，否则永久漏索引。"""
        doc = KnowledgeDocument(
            id=uuid.uuid4(),
            title="被手工改过",
            doc_type="condition",
            content="新内容已被直接写库改掉",
            content_hash=ks.content_sha256("旧的内容"),
            source="测试",
            source_tier="general",
            index_status="indexed",
            chunk_count=1,
            is_active=True,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        sync_session.add(doc)
        sync_session.commit()

        assert [d.id for d in ks.collect_pending_documents(sync_session)] == [doc.id]

    def test_collect_skips_inactive(self, sync_session) -> None:
        sync_session.add(
            KnowledgeDocument(
                id=uuid.uuid4(),
                title="已下线",
                doc_type="condition",
                content="下线内容",
                content_hash=ks.content_sha256("下线内容"),
                source="测试",
                source_tier="general",
                index_status="pending",
                chunk_count=0,
                is_active=False,
                created_at=utcnow(),
                updated_at=utcnow(),
            )
        )
        sync_session.commit()
        assert ks.collect_pending_documents(sync_session) == []

    def test_vectorize_idempotent_skip(self, sync_session, monkeypatch) -> None:
        """hash 未变且已 indexed 时跳过，且不碰 embedding。"""
        called: list[int] = []
        monkeypatch.setattr(
            ks,
            "embed_texts",
            lambda texts, **kw: called.append(len(texts)) or fake_vectors(len(texts)),
        )
        doc = self._make_doc(sync_session, content="不变的内容", index_status="indexed")
        stats = ks.vectorize_document(sync_session, doc)
        sync_session.commit()

        assert stats.skipped is True
        assert called == [], "跳过分支持调用 embedding，否则重跑任务就是白烧配额"

    def test_vectorize_creates_chunks_tsv_and_vectors(
        self, sync_session, monkeypatch
    ) -> None:
        """完整编排：切块数、反规范化字段、tsv 填充、向量维度与模型版本。"""
        calls: list[int] = []

        def fake_embed(texts, **kw):
            calls.append(len(texts))
            return fake_vectors(len(texts))

        monkeypatch.setattr(ks, "embed_texts", fake_embed)

        long_content = "这是一段足够长的中文测试文本，用来触发分块逻辑。" * 60
        doc = self._make_doc(
            sync_session,
            content=long_content,
            index_status="pending",
            stored_hash="wrong-on-purpose",
        )
        stats = ks.vectorize_document(sync_session, doc)
        sync_session.commit()

        assert stats.created_chunks > 1, "长文本应切成多块"
        assert stats.created_vectors == stats.created_chunks
        assert calls == [stats.created_chunks], "embedding 应每篇只批量调用一次"

        chunks = (
            sync_session.query(KnowledgeChunk)
            .filter(KnowledgeChunk.document_id == doc.id)
            .order_by(KnowledgeChunk.chunk_index)
            .all()
        )
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        for chunk in chunks:
            assert chunk.title == doc.title
            assert chunk.doc_type == doc.doc_type
            assert chunk.source == doc.source
            assert chunk.source_tier == doc.source_tier
            assert chunk.page_number is None, "纯文本来源不该伪造页码"
            assert chunk.content_hash == ks.content_sha256(chunk.content)
            assert chunk.tsv, "tsv 必须被填上分词结果（SQLite 下退化为 TEXT）"

        vectors = (
            sync_session.query(VectorEmbedding)
            .filter(VectorEmbedding.chunk_id.in_([c.id for c in chunks]))
            .all()
        )
        from app.core.config import settings

        assert len(vectors) == len(chunks)
        assert all(len(v.embedding) == settings.embedding_dim for v in vectors)
        assert all(v.model_version == settings.embedding_model for v in vectors)

        sync_session.refresh(doc)
        assert doc.index_status == ks.INDEX_STATUS_INDEXED
        assert doc.chunk_count == len(chunks)
        assert doc.content_hash == ks.content_sha256(long_content)
        assert doc.indexed_at is not None

    def test_vectorize_replaces_old_chunks_on_rerun(
        self, sync_session, monkeypatch
    ) -> None:
        """重跑不残留旧切块/旧向量（幂等的另一半）。"""
        monkeypatch.setattr(ks, "embed_texts", lambda texts, **kw: fake_vectors(len(texts)))
        doc = self._make_doc(sync_session, content="第一版内容", index_status="pending")
        ks.vectorize_document(sync_session, doc)
        sync_session.commit()
        first_ids = {
            c.id
            for c in sync_session.query(KnowledgeChunk).filter(
                KnowledgeChunk.document_id == doc.id
            ).all()
        }

        doc.content = "第二版完全不同的内容，长度也做了变化以便重新切块。"
        doc.index_status = ks.INDEX_STATUS_STALE
        sync_session.commit()

        stats = ks.vectorize_document(sync_session, doc)
        sync_session.commit()

        assert stats.deleted_chunks == len(first_ids)
        current = (
            sync_session.query(KnowledgeChunk)
            .filter(KnowledgeChunk.document_id == doc.id)
            .all()
        )
        assert not ({c.id for c in current} & first_ids), "旧切块应已被替换"
        # 向量总数应与切块总数一致，说明没有孤儿向量残留
        assert sync_session.query(VectorEmbedding).count() == len(current)

    def test_vectorize_empty_content_marks_indexed_zero_chunks(
        self, sync_session, monkeypatch
    ) -> None:
        """清洗后为空要标 indexed 而非 pending，否则会每轮空转。"""
        monkeypatch.setattr(ks, "embed_texts", lambda texts, **kw: fake_vectors(len(texts)))
        doc = self._make_doc(sync_session, content="   \n\n  ", index_status="pending")
        stats = ks.vectorize_document(sync_session, doc)
        sync_session.commit()

        sync_session.refresh(doc)
        assert doc.index_status == ks.INDEX_STATUS_INDEXED
        assert doc.chunk_count == 0
        assert stats.skipped is True

    def test_vectorize_failure_marks_failed(self, sync_session, monkeypatch) -> None:
        """embedding 抛错时文档标 failed 并外抛，让上层重试兜底。"""

        def boom(texts, **kw):
            raise RuntimeError("embedding 服务不可用")

        monkeypatch.setattr(ks, "embed_texts", boom)
        doc = self._make_doc(sync_session, content="需要向量化的内容", index_status="pending")

        with pytest.raises(RuntimeError):
            ks.vectorize_document(sync_session, doc)
        sync_session.rollback()
        sync_session.refresh(doc)
        assert doc.index_status == ks.INDEX_STATUS_FAILED

        # failed 会被下一轮重新捞起，一次抖动不会把文档永久卡在失败态
        assert doc.id in {d.id for d in ks.collect_pending_documents(sync_session)}

    def test_embed_texts_validates_dimension(self, monkeypatch) -> None:
        """维度不符必须报错，不能让脏向量落库。"""
        from app.core.config import settings
        import app.tasks.celery_app as celery_module

        monkeypatch.setattr(_real(celery_module), "run_async", lambda coro: coro.close() or None)

        wrong = [[0.1] * (settings.embedding_dim - 1)]

        async def fake_call():
            return wrong

        class FakeLLM:
            async def embed(self, texts, **kw):
                return [], None

        # 直接测校验分支：注入一个返回错误维度的 run_async
        import app.services.llm.service as llm_service_module

        monkeypatch.setattr(
            llm_service_module,
            "get_llm_service",
            lambda: FakeLLM(),
        )
        monkeypatch.setattr(
            _real(celery_module),
            "run_async",
            lambda coro: self._drive_with(coro, wrong),
        )
        with pytest.raises(ValueError):
            ks.embed_texts(["任意文本"])

    @staticmethod
    def _drive_with(coro, result):
        """关闭协程避免 RuntimeWarning，然后返回预设结果。"""
        coro.close()
        return result

    def test_embed_texts_count_mismatch(self, monkeypatch) -> None:
        import app.tasks.celery_app as celery_module

        monkeypatch.setattr(
            _real(celery_module), "run_async", lambda coro: self._drive_with(coro, [[0.1]])
        )
        with pytest.raises(ValueError, match="数量"):
            ks.embed_texts(["一", "二"])

    def test_embed_texts_empty_input_short_circuits(self) -> None:
        assert ks.embed_texts([]) == []

    def test_remove_orphaned(self, sync_session, monkeypatch) -> None:
        """删除「文档已不存在」的切块与向量。"""
        monkeypatch.setattr(ks, "embed_texts", lambda texts, **kw: fake_vectors(len(texts)))
        doc = self._make_doc(sync_session, content="即将消失的文档内容", index_status="pending")
        ks.vectorize_document(sync_session, doc)
        sync_session.commit()

        # 绕过 ORM 关系直接删文档行，模拟迁移脚本/手工 SQL 造成的孤儿
        sync_session.execute(
            sa.text("DELETE FROM knowledge_documents WHERE id = :id"),
            {"id": doc.id.hex},
        )
        sync_session.commit()

        removed = ks.remove_orphaned(sync_session)
        sync_session.commit()

        assert removed >= 1
        assert sync_session.query(KnowledgeChunk).count() == 0
        assert sync_session.query(VectorEmbedding).count() == 0

    def test_mark_document_stale(self, sync_session) -> None:
        doc = self._make_doc(sync_session, content="内容", index_status="indexed")
        assert ks.mark_document_stale(sync_session, doc.id) is True
        sync_session.commit()
        sync_session.refresh(doc)
        assert doc.index_status == ks.INDEX_STATUS_STALE

        # 已是 stale 不再改写，避免把「排队中」倒退成别的状态
        assert ks.mark_document_stale(sync_session, doc.id) is False
        assert ks.mark_document_stale(sync_session, uuid.uuid4()) is False

    @staticmethod
    def _make_doc(
        sync_session,
        *,
        content: str,
        index_status: str,
        stored_hash: str | None = None,
    ) -> KnowledgeDocument:
        doc = KnowledgeDocument(
            id=uuid.uuid4(),
            title="测试文档",
            doc_type="guideline",
            content=content,
            content_hash=stored_hash or ks.content_sha256(content),
            source="测试来源",
            source_tier="professional",
            index_status=index_status,
            chunk_count=0,
            is_active=True,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        sync_session.add(doc)
        sync_session.commit()
        return doc


# ==================== populate_chunk_tsv 的方言分支 ====================


class TestTsvDialect:
    def test_sqlite_branch_writes_plain_text(self, sync_session, monkeypatch) -> None:
        """SQLite 分支只验证「写进去了」。

        to_tsvector 是 PG 专有函数，SQLite 里不存在，因此本用例刻意走非 PG 分支，
        断言值是应用层分词串。PG 分支的真实行为只能在容器里验（批次 A 冒烟已确认
        ``UPDATE ... SET tsv = to_tsvector('simple', :tok)`` 写法可行）。
        """
        doc = TestWorkerSide._make_doc(
            sync_session, content="布洛芬用于退热。", index_status="pending"
        )
        chunk = KnowledgeChunk(
            document_id=doc.id,
            chunk_index=0,
            content="布洛芬用于退热。",
            content_hash=ks.content_sha256("布洛芬用于退热。"),
            token_count=8,
            title=doc.title,
            doc_type=doc.doc_type,
            source=doc.source,
            source_tier=doc.source_tier,
        )
        sync_session.add(chunk)
        sync_session.commit()

        from app.services.rag.text_processing import tokenize_for_tsv

        ks.populate_chunk_tsv(sync_session, [chunk.id], [tokenize_for_tsv("布洛芬用于退热。")])
        sync_session.commit()
        sync_session.refresh(chunk)

        assert chunk.tsv is not None
        assert "布洛芬" in chunk.tsv

    def test_noop_on_empty_ids(self, sync_session) -> None:
        """空列表直接返回，不发任何 SQL。"""
        assert ks.populate_chunk_tsv(sync_session, [], []) is None


# ==================== 任务状态机（worker 侧）====================


class TestTaskStateSync:
    def _make_task(self, sync_session, status_value: str):
        task = ks.AITask(
            id=uuid.uuid4(),
            task_type=ks.TASK_TYPE_VECTORIZE,
            status=status_value,
            progress=0,
            retry_count=0,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        sync_session.add(task)
        sync_session.commit()
        return task

    def test_pending_to_running_sets_started_at(self, sync_session) -> None:
        task = self._make_task(sync_session, "pending")
        ks.set_task_status_sync(sync_session, task, "running", progress=0)
        sync_session.commit()
        sync_session.refresh(task)
        assert task.status == "running"
        assert task.started_at is not None

    def test_completed_writes_result_and_timestamps(self, sync_session) -> None:
        task = self._make_task(sync_session, "running")
        task.started_at = utcnow()
        ks.set_task_status_sync(
            sync_session, task, "completed", result={"chunks_created": 7}, progress=100
        )
        sync_session.commit()
        sync_session.refresh(task)
        assert task.status == "completed"
        assert task.result == {"chunks_created": 7}
        assert task.completed_at is not None
        assert task.duration_ms is not None and task.duration_ms >= 0

    def test_illegal_transition_rejected(self, sync_session) -> None:
        """completed 是终态，不能被后续消息复活。"""
        from app.core.exceptions import InvalidStateTransitionError

        task = self._make_task(sync_session, "completed")
        with pytest.raises(InvalidStateTransitionError):
            ks.set_task_status_sync(sync_session, task, "running")

    def test_retrying_increments_counter(self, sync_session) -> None:
        task = self._make_task(sync_session, "running")
        ks.set_task_status_sync(sync_session, task, "retrying")
        ks.bump_retry_count_sync(sync_session, task)
        sync_session.commit()
        sync_session.refresh(task)
        assert task.status == "retrying"
        assert task.retry_count == 1

    def test_error_message_truncated(self, sync_session) -> None:
        task = self._make_task(sync_session, "running")
        ks.set_task_status_sync(sync_session, task, "failed", error_message="E" * 5000)
        sync_session.commit()
        sync_session.refresh(task)
        assert len(task.error_message) <= 2000

    def test_load_task_for_update(self, sync_session) -> None:
        """SQLite 无行锁，此用例只验证读取路径不因 with_for_update 报错。"""
        task = self._make_task(sync_session, "pending")
        loaded = ks.load_task_for_update_sync(sync_session, task.id)
        assert loaded is not None and loaded.id == task.id
        assert ks.load_task_for_update_sync(sync_session, uuid.uuid4()) is None


# ==================== 派发与查重（异步 service）====================


class TestDispatch:
    async def test_start_vectorization_creates_task_and_dispatches(
        self, db: AsyncSession, monkeypatch
    ) -> None:
        """打桩 dispatch_task：测试环境没有 broker，绝不能真连 Redis。"""
        import app.tasks.celery_app as celery_module

        dispatched: list[dict[str, Any]] = []

        class FakeResult:
            id = "celery-fake-id"

        def fake_dispatch(task_name, *args, **kwargs):
            dispatched.append({"name": task_name, "args": args, "kwargs": kwargs})
            return FakeResult()

        monkeypatch.setattr(_real(celery_module), "dispatch_task", fake_dispatch)

        user_id = uuid.uuid4()
        task = await ks.start_vectorization(db, user_id)

        assert task.task_type == ks.TASK_TYPE_VECTORIZE
        assert task.status == "pending"
        assert task.celery_task_id == "celery-fake-id"

        assert len(dispatched) == 1
        call = dispatched[0]
        assert call["name"] == "app.tasks.knowledge_tasks.vectorize_knowledge"
        assert call["args"] == (str(task.id),)
        assert call["kwargs"]["queue"] == _real(celery_module).QUEUE_VECTORIZE
        # task_id 显式传入实现 Celery 侧去重
        assert call["kwargs"]["task_id"] == str(task.id)

    async def test_duplicate_dispatch_rejected(self, db: AsyncSession, monkeypatch) -> None:
        """已有占位中的同型任务时第二次派发必须被拒（409 语义）。"""
        import app.tasks.celery_app as celery_module

        monkeypatch.setattr(
            _real(celery_module), "dispatch_task", lambda *a, **kw: type("R", (), {"id": "x"})()
        )
        await ks.start_vectorization(db, uuid.uuid4())

        from app.core.exceptions import DuplicateTaskError

        with pytest.raises(DuplicateTaskError) as excinfo:
            await ks.start_vectorization(db, uuid.uuid4())
        assert excinfo.value.detail.get("task_type") == ks.TASK_TYPE_VECTORIZE

    @pytest.mark.parametrize("status_value", ["pending", "running", "retrying"])
    async def test_all_active_statuses_block_new_dispatch(
        self, db: AsyncSession, monkeypatch, status_value: str
    ) -> None:
        """队列积压（pending）与重试中（retrying）同样算占位，只拦 running 会漏。"""
        import app.tasks.celery_app as celery_module

        monkeypatch.setattr(
            _real(celery_module), "dispatch_task", lambda *a, **kw: type("R", (), {"id": "x"})()
        )
        task = await ks.start_vectorization(db, uuid.uuid4())
        task.status = status_value
        await db.commit()

        from app.core.exceptions import DuplicateTaskError

        with pytest.raises(DuplicateTaskError):
            await ks.start_vectorization(db, uuid.uuid4())

    async def test_completed_task_does_not_block(self, db: AsyncSession, monkeypatch) -> None:
        import app.tasks.celery_app as celery_module

        monkeypatch.setattr(
            _real(celery_module), "dispatch_task", lambda *a, **kw: type("R", (), {"id": "x"})()
        )
        task = await ks.start_vectorization(db, uuid.uuid4())
        task.status = "completed"
        await db.commit()

        again = await ks.start_vectorization(db, uuid.uuid4())
        assert again.id != task.id

    async def test_dispatch_failure_marks_task_failed(
        self, db: AsyncSession, monkeypatch
    ) -> None:
        """broker 不可用时不能留一条 pending 记录永久占位。"""
        import app.tasks.celery_app as celery_module

        monkeypatch.setattr(_real(celery_module), "dispatch_task", lambda *a, **kw: None)

        from app.core.exceptions import TaskQueueUnavailableError

        with pytest.raises(TaskQueueUnavailableError):
            await ks.start_vectorization(db, uuid.uuid4())

    async def test_rebuild_task_type_routes_to_rebuild_name(
        self, db: AsyncSession, monkeypatch
    ) -> None:
        import app.tasks.celery_app as celery_module

        seen: list[dict[str, Any]] = []
        monkeypatch.setattr(
            _real(celery_module),
            "dispatch_task",
            lambda name, *a, **kw: seen.append({"name": name, "kw": kw})
            or type("R", (), {"id": "y"})(),
        )
        task = await ks.start_vectorization(
            db,
            uuid.uuid4(),
            task_type=ks.TASK_TYPE_REBUILD,
            payload={"model_version": "text-embedding-3-large"},
        )
        assert task.task_type == ks.TASK_TYPE_REBUILD
        assert seen[0]["name"] == "app.tasks.knowledge_tasks.rebuild_all_vectors"
        # model_version 是 rebuild 任务的必填关键字参数
        assert seen[0]["kw"]["model_version"] == "text-embedding-3-large"

    async def test_acquire_lock_degrades_on_sqlite(self, db: AsyncSession) -> None:
        """SQLite 没有 pg_try_advisory_xact_lock，必须降级放行而非抛错。"""
        assert await ks.acquire_dispatch_lock(db, "knowledge:test") is True


# ==================== HTTP 端点 ====================


ADMIN = (uuid.uuid4(), "root_admin", "admin", 6)


@pytest.mark.parametrize("client", [None], indirect=True)
class TestKnowledgeEndpointsAnonymous:
    async def test_all_endpoints_require_auth(self, client: AsyncClient) -> None:
        """不带凭证一律 401，且错误体带机器可读 code。"""
        checks = [
            ("get", "/api/v1/knowledge/documents"),
            ("post", "/api/v1/knowledge/documents"),
            ("delete", "/api/v1/knowledge/documents/" + str(uuid.uuid4())),
            ("post", "/api/v1/knowledge/rebuild"),
            ("get", "/api/v1/knowledge/tasks/" + str(uuid.uuid4())),
            ("post", "/api/v1/knowledge/search"),
        ]
        for method, path in checks:
            response = await getattr(client, method)(path)
            assert response.status_code == 401, f"{method.upper()} {path} 应要求登录"
            assert error_code_of(response) == "auth_invalid_credentials"


@pytest.mark.parametrize(
    "client", [(uuid.uuid4(), "cashier_wu", "cashier", 4)], indirect=True
)
class TestKnowledgeEndpointsNoPermission:
    async def test_cashier_denied_everything(self, client: AsyncClient) -> None:
        """收费员在权限矩阵里没有任何 knowledge 授权，全部动作都应 403。"""
        r1 = await client.get("/api/v1/knowledge/documents")
        assert r1.status_code == 403
        assert error_code_of(r1) == "auth_forbidden"

        r2 = await client.post(
            "/api/v1/knowledge/documents",
            json={"title": "A", "content": "B", "doc_type": "drug"},
        )
        assert r2.status_code == 403

        r3 = await client.post("/api/v1/knowledge/rebuild")
        assert r3.status_code == 403

        r4 = await client.post("/api/v1/knowledge/search", json={"query": "高血压"})
        assert r4.status_code == 403


@pytest.mark.parametrize("client", [ADMIN], indirect=True)
class TestKnowledgeEndpointsAdmin:
    async def test_routes_present_in_openapi(self, client: AsyncClient) -> None:
        # 方法本身只读 app.openapi()，参数 client 是类级 indirect 参数化
        # 的收集要求（pytest 要求类内每个测试都能解析 client fixture）。
        _ = client
        paths = set(app.openapi()["paths"])
        for expected in (
            "/api/v1/knowledge/documents",
            "/api/v1/knowledge/documents/{doc_id}",
            "/api/v1/knowledge/rebuild",
            "/api/v1/knowledge/tasks/{task_id}",
            "/api/v1/knowledge/search",
        ):
            assert expected in paths, f"缺少路由 {expected}"

    async def test_create_returns_201_and_pending(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/knowledge/documents",
            json={
                "title": "儿童退热用药提示",
                "content": "儿童用药须遵医嘱，不得照搬成人剂量。",
                "doc_type": "drug_manual",
                "source": "国家药监局",
                "source_tier": "authoritative",
            },
        )
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["index_status"] == "pending"
        assert body["chunk_count"] == 0
        assert body["content_length"] > 0
        assert uuid.UUID(body["id"])

    async def test_create_rejects_bad_doc_type(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/knowledge/documents",
            json={"title": "A", "content": "B", "doc_type": "wat"},
        )
        assert response.status_code == 422

    async def test_crud_roundtrip(self, client: AsyncClient) -> None:
        created = await client.post(
            "/api/v1/knowledge/documents",
            json={"title": "原始标题", "content": "原始正文", "doc_type": "condition"},
        )
        assert created.status_code == 201
        doc_id = created.json()["id"]

        detail = await client.get(f"/api/v1/knowledge/documents/{doc_id}")
        assert detail.status_code == 200
        assert detail.json()["content"] == "原始正文"

        listed = await client.get("/api/v1/knowledge/documents", params={"page_size": 10})
        assert listed.status_code == 200
        body = listed.json()
        assert body["total"] == 1
        assert body["page"] == 1
        assert body["items"][0]["title"] == "原始标题"
        assert "content" not in body["items"][0], "列表摘要不得带正文"

        patched = await client.patch(
            f"/api/v1/knowledge/documents/{doc_id}",
            json={"title": "改后标题"},
        )
        assert patched.status_code == 200
        assert patched.json()["title"] == "改后标题"

        deleted = await client.delete(f"/api/v1/knowledge/documents/{doc_id}")
        assert deleted.status_code == 204

        gone = await client.get(f"/api/v1/knowledge/documents/{doc_id}")
        assert gone.status_code == 404
        assert error_code_of(gone) == "resource_not_found"

    async def test_get_unknown_id_is_404_not_500(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/v1/knowledge/documents/{uuid.uuid4()}")
        assert response.status_code == 404

    async def test_malformed_uuid_is_404(self, client: AsyncClient) -> None:
        """格式错误的 ID 统一按「取不到资源」处理，前端只需一个分支。"""
        response = await client.get("/api/v1/knowledge/documents/not-a-uuid")
        assert response.status_code == 404
        assert error_code_of(response) == "resource_not_found"

    async def test_search_returns_zero_results_not_500(self, client: AsyncClient) -> None:
        """SQLite 下降级验证：两路都空、is_zero_result=true，绝不 500。

        这正是 retriever 的降级承诺在真实 HTTP 栈上的体现——测试环境没有
        pgvector，也不能因为缺它就报错。
        """
        response = await client.post("/api/v1/knowledge/search", json={"query": "高血压怎么办"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["results"] == []
        diag = body["diagnostics"]
        assert diag["vector_hit_count"] == 0
        assert diag["keyword_hit_count"] == 0
        assert diag["both_hit_count"] == 0
        assert diag["is_zero_result"] is True

    async def test_search_rejects_empty_query(self, client: AsyncClient) -> None:
        response = await client.post("/api/v1/knowledge/search", json={"query": ""})
        assert response.status_code == 422

    async def test_rebuild_accepted_and_queryable(
        self, client: AsyncClient, monkeypatch
    ) -> None:
        """派发打桩后走完整 HTTP 链路：202 → 用返回的 task_id 查到 pending。"""
        import app.tasks.celery_app as celery_module

        monkeypatch.setattr(
            _real(celery_module),
            "dispatch_task",
            lambda name, *a, **kw: type("R", (), {"id": "celery-http-1"})(),
        )

        response = await client.post("/api/v1/knowledge/rebuild")
        assert response.status_code == 202, response.text
        body = response.json()
        assert body["task_type"] == ks.TASK_TYPE_VECTORIZE
        assert body["celery_task_id"] == "celery-http-1"

        task = await client.get(f"/api/v1/knowledge/tasks/{body['task_id']}")
        assert task.status_code == 200
        assert task.json()["status"] == "pending"

    async def test_rebuild_conflict_on_second_click(
        self, client: AsyncClient, monkeypatch
    ) -> None:
        import app.tasks.celery_app as celery_module

        monkeypatch.setattr(
            _real(celery_module),
            "dispatch_task",
            lambda name, *a, **kw: type("R", (), {"id": "celery-http-2"})(),
        )
        first = await client.post("/api/v1/knowledge/rebuild")
        assert first.status_code == 202

        second = await client.post("/api/v1/knowledge/rebuild")
        assert second.status_code == 409
        assert error_code_of(second) == "task_already_running"
        # 冲突里带回已有任务 ID，前端可直接跳转看进度
        assert second.json()["detail"]["task_id"] == first.json()["task_id"]

    async def test_task_unknown_id_404(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/v1/knowledge/tasks/{uuid.uuid4()}")
        assert response.status_code == 404


@pytest.mark.parametrize(
    "client", [(uuid.uuid4(), "dr_zhou", "doctor", 2)], indirect=True
)
class TestKnowledgeEndpointsDoctor:
    """医生在矩阵里只有 knowledge:read，写操作必须被拒。"""

    async def test_doctor_can_read_but_not_write(self, client: AsyncClient) -> None:
        assert (await client.get("/api/v1/knowledge/documents")).status_code == 200
        assert (await client.post("/api/v1/knowledge/rebuild")).status_code == 403
        assert (
            await client.post(
                "/api/v1/knowledge/documents",
                json={"title": "A", "content": "B", "doc_type": "drug"},
            )
        ).status_code == 403


# ==================== Celery 任务注册契约（静态断言）====================


class TestCeleryTaskContract:
    """任务签名与调度参数是部署契约，改坏了 worker 就接不到活。

    真正的执行逻辑需要 broker + worker + 真库，属容器集成范畴；
    这里只保证装饰器参数与本批次约定一致。
    """

    def test_tasks_registered_with_expected_names_and_queue(self) -> None:
        from app.tasks import knowledge_tasks  # noqa: F401  - 触发注册

        tasks = celery_app_tasks()
        for name in (
            "app.tasks.knowledge_tasks.vectorize_knowledge",
            "app.tasks.knowledge_tasks.rebuild_all_vectors",
        ):
            assert name in tasks, f"任务 {name} 未注册"

        vec = tasks["app.tasks.knowledge_tasks.vectorize_knowledge"]
        assert vec.queue == "vectorize"
        assert vec.max_retries == 3
        assert vec.time_limit == 1800
        assert vec.soft_time_limit == 1700

        reb = tasks["app.tasks.knowledge_tasks.rebuild_all_vectors"]
        assert reb.queue == "vectorize"
        assert reb.max_retries == 1
        assert reb.time_limit == 3600

    def test_bodies_are_implemented(self) -> None:
        """body 不再是 raise NotImplementedError 的桩。"""
        import inspect

        from app.tasks import knowledge_tasks

        for func in (
            knowledge_tasks.vectorize_knowledge,
            knowledge_tasks.rebuild_all_vectors,
        ):
            src = inspect.getsource(func.run)  # type: ignore[attr-defined]
            assert "NotImplementedError" not in src


def celery_app_tasks() -> dict[str, Any]:
    from app.tasks.celery_app import celery_app

    return dict(celery_app.tasks)


# ==================== TestClient 冒烟（不碰数据库）====================


def test_health_still_ok() -> None:
    """挂了新 router 之后应用仍能构建。

    用 TestClient 但不发业务请求：/health 会真连 PG，测试环境连不上只是报
    degraded，仍返回 200，可作为「应用可实例化 + openapi 可生成」的最廉价证据。
    """
    with TestClient(app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["version"]
