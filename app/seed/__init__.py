"""种子数据脚本。

从零建库策略：PostgreSQL 全新建表后由本包灌入初始数据，不迁移旧 MySQL 数据。
种子脚本**幂等**——重复执行不产生重复记录，因为初始化流程可能被多次触发
（本地调试、CI、容器重建），非幂等的种子脚本会让人不敢重跑。
"""

from app.seed.departments import DEPARTMENTS, seed_departments
from app.seed.rbac import seed_rbac
from app.seed.users import DEMO_USERS, seed_users

__all__ = [
    "DEPARTMENTS",
    "DEMO_USERS",
    "seed_departments",
    "seed_rbac",
    "seed_users",
]
