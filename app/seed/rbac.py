"""角色与权限种子数据。

写入 ``roles`` / ``permissions`` / ``role_permissions`` 三张表。
角色 ID 固定（见 ``ROLE_DEFINITIONS``），便于权限矩阵可读与幂等重跑。

写入前先调 ``validate_matrix()`` 自检：矩阵里若用了未登记的资源名或
动作名（如把 ``dispense`` 拼成 ``dispencing``），错误会静默——授权记录
写进去了，但 ``require_permission`` 查的是正确拼写，该角色永远拿不到权限
且不报任何错。这类 bug 在测试里极难发现，因此必须在写库前拦住。
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.auth import Permission, Role, RolePermission
from app.services.rbac_matrix import (
    ROLE_DEFINITIONS,
    ROLE_PERMISSIONS,
    validate_matrix,
)

logger = logging.getLogger(__name__)


async def seed_rbac(db: AsyncSession) -> dict[str, int]:
    """写入角色、权限与关联关系，返回各部分新建条数。

    幂等实现：
    - roles 按固定 ID upsert（ID 是业务语义的一部分，不是代理键）
    - permissions 按 (resource, action) 查重，只插缺失的
    - role_permissions 先查该角色已有关联，只补缺失的，不删除多余的
      ——管理员可能临时授过额外权限，重跑种子不应冲掉

    Raises:
        ValueError: 权限矩阵自身校验不通过时抛出，阻止写入脏数据。
    """
    problems = validate_matrix()
    if problems:
        # 快速失败优于写入后权限静默失效
        raise ValueError("RBAC 权限矩阵校验失败：\n" + "\n".join(problems))

    created = {"roles": 0, "permissions": 0, "role_permissions": 0}

    # ---------- 1. 角色 ----------
    for role_id, role_name, description in ROLE_DEFINITIONS:
        existing = await db.get(Role, role_id)
        if existing is None:
            db.add(Role(id=role_id, name=role_name.value, description=description))
            created["roles"] += 1
        elif existing.name != role_name.value:
            # ID 与角色名不匹配说明种子数据被手工改乱，必须报错而非静默覆盖：
            # 权限矩阵是按角色名声明的，ID 错位会导致整张授权表失效
            raise ValueError(
                f"角色 ID {role_id} 已存在但名称为 {existing.name}，"
                f"与种子定义 {role_name.value} 不一致，请检查数据库"
            )
    await db.flush()

    # ---------- 2. 权限项 ----------
    rows = (await db.execute(select(Permission))).scalars().all()
    existing_permissions = {(row.resource, row.action): row.id for row in rows}

    needed = set()
    for permissions in ROLE_PERMISSIONS.values():
        needed |= permissions

    for resource, action in sorted(needed):
        if (resource, action) in existing_permissions:
            continue
        permission = Permission(resource=resource, action=action)
        db.add(permission)
        await db.flush()  # 需要立即拿到自增 ID 供关联表使用
        existing_permissions[(resource, action)] = permission.id
        created["permissions"] += 1

    # ---------- 3. 角色-权限关联 ----------
    # 一次性载入全部关联，避免在循环里逐条查询（N+1）。
    # 必须用 .scalars()：.all() 返回的是单列包装行（row[0] 才是实体），
    # 直接 row.role_id 会 AttributeError——空表时循环不进，这个错误只在
    # 第二次及以后执行 seed_rbac 时才暴露（init-db 容器重启即触发）。
    link_rows = (await db.execute(select(RolePermission))).scalars().all()
    existing_links = {(row.role_id, row.permission_id) for row in link_rows}

    role_id_by_name = {
        role_name.value: role_id for role_id, role_name, _ in ROLE_DEFINITIONS
    }

    for role_name, permissions in ROLE_PERMISSIONS.items():
        role_id = role_id_by_name[role_name.value]
        for resource, action in sorted(permissions):
            permission_id = existing_permissions.get((resource, action))
            if permission_id is None:  # pragma: no cover - 上面已确保写入
                continue
            if (role_id, permission_id) in existing_links:
                continue
            db.add(RolePermission(role_id=role_id, permission_id=permission_id))
            existing_links.add((role_id, permission_id))
            created["role_permissions"] += 1

    await db.flush()
    logger.info(
        "RBAC 种子数据：角色 %d、权限 %d、关联 %d",
        created["roles"],
        created["permissions"],
        created["role_permissions"],
    )
    return created


__all__ = ["seed_rbac"]
