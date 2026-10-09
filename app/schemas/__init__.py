"""Pydantic v2 请求/响应模型包。

与 ``app/models``（SQLAlchemy ORM）分层的理由：
- ORM 对象带 relationship 与会话状态，直接序列化容易触发隐式懒加载或泄漏
  ``password_hash`` 这类敏感列
- 入参校验规则（账号归一化、密码强度）属于协议层关注点，不应写进表模型

约定：**所有校验规则复用 ``app.core.security``，本包不重写任何正则**。
"""
