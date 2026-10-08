"""IMCS 重构期间保留的旧版 RAG 系统代码。

这里存放演进前的 SQLModel 同步数据模型与 Pydantic 请求定义。重构按阶段推进，
旧代码在新模块接管对应职责后逐步删除，不要在此新增功能。

- ``models``：旧 SQLModel 表定义（MySQL 后端）
- ``schemas``：旧请求/响应模型
"""
