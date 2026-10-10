"""种子数据统一入口。

执行顺序由外键依赖决定，不可调换：
1. 科室（departments）——无依赖
2. 角色与权限（roles / permissions / role_permissions）——无依赖
3. 用户（users）——依赖 roles 与 departments
4. 患者档案（patient_profiles）——依赖 users，随用户一并写入
5. 医生排班（doctor_schedules）——依赖 users 与 departments

整个流程在**单个事务**内提交：任何一步失败都回滚，不会留下
「角色建好了但账号没建」这种半成品状态。半成品状态最难排查，
因为下次重跑时幂等逻辑会跳过已存在的部分，错误被掩盖。

知识库种子数据（病症、健康宣教、就医警示）由 ``app/seed/knowledge.py``
提供，在科室与用户之后写入（知识文档可选关联科室 ID）。
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session_factory
from app.seed.departments import seed_departments
from app.seed.knowledge import seed_knowledge
from app.seed.rbac import seed_rbac
from app.seed.schedules import seed_schedules
from app.seed.users import DEMO_PASSWORD, seed_users

logger = logging.getLogger(__name__)


async def seed_all(db: AsyncSession, *, include_schedules: bool = True) -> dict[str, object]:
    """按依赖顺序写入全部种子数据。

    调用方负责 commit——这样能把种子写入与调用方的其他初始化操作
    合并到同一个事务里。本函数只 flush，不 commit。
    """
    stats: dict[str, object] = {}

    # 1. 科室
    stats["departments_created"] = await seed_departments(db)

    # 2. 角色与权限。矩阵自检失败会抛 ValueError，此时尚未 commit，
    #    不会产生任何脏数据。
    stats["rbac"] = await seed_rbac(db)

    # 3. 用户与患者档案
    stats["users_created"] = await seed_users(db)

    # 4. 知识文档。依赖科室已存在（可选关联 department_id），
    #    不依赖用户，放在排班之前。
    stats["knowledge_created"] = await seed_knowledge(db)

    # 5. 医生排班。依赖用户已存在，因此排在最后。
    if include_schedules:
        stats["schedules_created"] = await seed_schedules(db)

    await db.flush()
    return stats


async def run_seed(include_schedules: bool = True) -> dict[str, object]:
    """独立执行种子写入并提交。供命令行与容器初始化调用。"""
    async with async_session_factory() as db:
        stats = await seed_all(db, include_schedules=include_schedules)
        await db.commit()
    logger.info("种子数据写入完成：%s", stats)
    return stats


def main() -> None:
    """命令行入口：``python -m app.seed.run_all``。"""
    parser = argparse.ArgumentParser(description="写入 IMCS 种子数据")
    parser.add_argument(
        "--skip-schedules",
        action="store_true",
        help="跳过医生排班数据（只建账号与权限时使用）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出 DEBUG 级别日志",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    stats = asyncio.run(run_seed(include_schedules=not args.skip_schedules))

    print("\n===== 种子数据写入结果 =====")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"\n演示账号统一密码：{DEMO_PASSWORD}")
    print("患者 patient01 / patient02")
    print("医生 doctor01（心血管）/ doctor02（内科）/ doctor03（神经内科）")
    print("检验科 lab01　收费员 cashier01　药师 pharmacist01　管理员 admin")
    print("\n注意：以上为公开演示凭据，生产环境必须删除或强制改密。")


if __name__ == "__main__":
    main()
