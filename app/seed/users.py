"""演示账号种子数据。

为六类角色各建一个可登录账号，便于演示完整业务闭环与面试现场跑通。

**安全提示**：这些账号密码是公开的演示凭据，只应存在于开发/演示环境。
生产部署必须删除本模块的调用，或强制首次登录改密——
一套任何人都能登录的「医生账号」在真实系统里等于没有权限控制。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password
from app.models.auth import Department, RoleName, User
from app.models.patient import PatientProfile

logger = logging.getLogger(__name__)

# 演示环境统一密码。满足 validate_password 的 8-32 位且含字母与数字要求。
DEMO_PASSWORD = "Imcs2026demo"


@dataclass(frozen=True, slots=True)
class DemoUserSeed:
    """一个演示账号。"""

    username: str
    real_name: str
    role: RoleName
    phone: str
    # 所属科室 code；患者与管理员为空
    department_code: str | None = None
    title: str | None = None
    specialty: str | None = None
    # 仅患者账号使用
    gender: str | None = None
    allergy_history: tuple[str, ...] = ()
    chronic_diseases: tuple[str, ...] = ()


DEMO_USERS: tuple[DemoUserSeed, ...] = (
    # ---------- 患者 ----------
    # 过敏史刻意填了「青霉素」：演示处方安全校验时，医生开阿莫西林
    # （青霉素类）应被拦截，这是用药安全闭环最直观的演示点。
    DemoUserSeed(
        username="patient01",
        real_name="张小明",
        role=RoleName.PATIENT,
        phone="+8613800000001",
        gender="M",
        allergy_history=("青霉素", "磺胺类"),
        chronic_diseases=("高血压",),
    ),
    DemoUserSeed(
        username="patient02",
        real_name="李静",
        role=RoleName.PATIENT,
        phone="+8613800000002",
        gender="F",
        allergy_history=(),
        chronic_diseases=("2型糖尿病",),
    ),
    # ---------- 医生 ----------
    DemoUserSeed(
        username="doctor01",
        real_name="王建国",
        role=RoleName.DOCTOR,
        phone="+8613900000001",
        department_code="CARDIO",
        title="主任医师",
        specialty="高血压、冠心病、心律失常",
    ),
    DemoUserSeed(
        username="doctor02",
        real_name="陈美玲",
        role=RoleName.DOCTOR,
        phone="+8613900000002",
        department_code="INTERNAL",
        title="副主任医师",
        specialty="呼吸道感染、胃肠疾病",
    ),
    DemoUserSeed(
        username="doctor03",
        real_name="刘志强",
        role=RoleName.DOCTOR,
        phone="+8613900000003",
        department_code="NEURO",
        title="主治医师",
        specialty="头痛、头晕、脑血管病",
    ),
    # ---------- 检验科 ----------
    DemoUserSeed(
        username="lab01",
        real_name="赵检验",
        role=RoleName.LAB_TECH,
        phone="+8613700000001",
        department_code="LAB",
        title="主管技师",
    ),
    # ---------- 收费员 ----------
    DemoUserSeed(
        username="cashier01",
        real_name="孙收费",
        role=RoleName.CASHIER,
        phone="+8613600000001",
        title="收费员",
    ),
    # ---------- 药师 ----------
    DemoUserSeed(
        username="pharmacist01",
        real_name="周药师",
        role=RoleName.PHARMACIST,
        phone="+8613500000001",
        department_code="PHARMACY",
        title="主管药师",
    ),
    # ---------- 超级管理员 ----------
    DemoUserSeed(
        username="admin",
        real_name="系统管理员",
        role=RoleName.ADMIN,
        phone="+8613400000001",
        title="超级管理员",
    ),
)

# 角色名 -> 固定角色 ID，与 services/rbac_matrix.ROLE_DEFINITIONS 保持一致。
# 这里重复声明而不从 ROLE_DEFINITIONS 反查，是因为种子账号需要的是
# 「角色名 -> ID」映射，反查要遍历元组；显式写出更直观。
# 若两处不一致，seed_rbac 的 ID 校验会先报错，不会静默产生错误授权。
ROLE_ID_BY_NAME: dict[RoleName, int] = {
    RoleName.PATIENT: 1,
    RoleName.DOCTOR: 2,
    RoleName.LAB_TECH: 3,
    RoleName.CASHIER: 4,
    RoleName.PHARMACIST: 5,
    RoleName.ADMIN: 6,
}


async def seed_users(db: AsyncSession, *, password: str = DEMO_PASSWORD) -> int:
    """写入演示账号，返回新建条数。

    幂等实现：按 username 查重，已存在则跳过（不覆盖密码，
    避免把用户改过的密码重置回演示密码）。

    依赖 ``seed_rbac`` 与 ``seed_departments`` 先执行：
    角色 ID 与科室必须已存在，否则外键约束会失败。
    """
    password_hash = hash_password(password)

    # 一次性载入科室与已存在用户名，避免循环内查询
    departments = {
        row.code: row.id
        for row in (await db.execute(select(Department))).scalars().all()
    }
    existing_usernames = set(
        (await db.execute(select(User.username))).scalars().all()
    )

    created = 0
    patient_seeds: list[tuple[DemoUserSeed, User]] = []

    for seed in DEMO_USERS:
        if seed.username in existing_usernames:
            continue

        department_id = None
        if seed.department_code:
            department_id = departments.get(seed.department_code)
            if department_id is None:
                # 科室缺失说明 seed_departments 没先跑。直接报错而不是
                # 静默建成无科室的医生——无科室医生不会出现在任何排班里，
                # 这种账号在演示时会让人困惑「为什么挂不上号」。
                raise ValueError(
                    f"科室 {seed.department_code} 不存在，"
                    f"请先执行 seed_departments（账号 {seed.username}）"
                )

        user = User(
            username=seed.username,
            password_hash=password_hash,
            real_name=seed.real_name,
            phone=seed.phone,
            role_id=ROLE_ID_BY_NAME[seed.role],
            department_id=department_id,
            title=seed.title,
            specialty=seed.specialty,
            is_active=True,
        )
        db.add(user)
        created += 1

        if seed.role == RoleName.PATIENT:
            patient_seeds.append((seed, user))

    # flush 后才能拿到 user.id 用于建患者档案
    await db.flush()

    for seed, user in patient_seeds:
        db.add(
            PatientProfile(
                user_id=user.id,
                gender=seed.gender,
                allergy_history=list(seed.allergy_history) or None,
                chronic_diseases=list(seed.chronic_diseases) or None,
            )
        )

    await db.flush()
    logger.info("演示账号种子数据：新建 %d 个（含 %d 份患者档案）", created, len(patient_seeds))
    return created


__all__ = ["DemoUserSeed", "DEMO_USERS", "DEMO_PASSWORD", "ROLE_ID_BY_NAME", "seed_users"]
