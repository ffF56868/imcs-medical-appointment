"""IMCS v1 路由聚合。

单独一层聚合而非把 ``include_router`` 直接写在入口文件里：
后续每个业务域（挂号、就诊、处方、知识库…）都要挂到同一个 ``/api/v1`` 前缀下，
集中登记能让「系统对外暴露了哪些资源」在一处看全，也便于按 tag 生成有序文档。

版本前缀放在这里而不是各 router 的 prefix 里，是为了让入口文件只关心
「挂一版 API」，未来出 v2 时新增一个 api_router_v2 即可并行灰度。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.auth import router as auth_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth_router)


__all__ = ["api_router"]
