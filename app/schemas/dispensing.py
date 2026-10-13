"""发药接口的请求/响应模型（阶段 5 发药域）。

字段口径对齐 ``Prescription`` / ``DispensingRecord``（models/pharmacy.py）。
约定与收费域、前序批次一致：金额整数「分」、UUID 对外字符串、响应显式列字段。

============================ 审核建模选择 ============================

药师侧有两个动作，本模块用 **两个独立请求体** 表达，而不是一个混合的
"dispense" 请求同时带审核意见：

1. ``ApproveRequest`` —— 处方审核（四查十对）。落到 DispensingRecord 上：
   建/更新一条记录并置 ``verified=True`` + ``verification_notes``。
2. ``DispenseRequest`` —— 实际发药。要求对应记录已 ``verified=True``（先审后发），
   写 pharmacist_id / dispensed_at / notes，并把处方 paid→dispensed。

之所以拆两个动作而不是合并：RBAC 矩阵里 pharmacist 同时持有
``prescription:approve`` 与 ``dispensing:create/update``——它们是两个授权位，
合并成一个端点会让「只被授权审核、尚未授权发药」的角色无法区分。
拆开后每个端点挂各自的权限闸门，与实际写入对齐（见 dispensing.py 路由）。

审核采用「先建 verified=False 草稿 → approve 时置 True」的路径（详见
dispense_service 模块 docstring 的方案 B），因此 ApproveRequest 只需给出
审核意见；若该处方还没有发药记录，approve 会当场建一条并直接置 verified=True。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _as_uuid(value: object, field_name: str) -> uuid.UUID:
    """字符串 → UUID，失败抛 ValueError（协议层 → 422）。同 billing schema。"""
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field_name} 必须是合法的 UUID 字符串") from exc


# ==================== 请求 ====================


class ApproveRequest(BaseModel):
    """药师审核处方（四查十对）。notes 必填：审核是责任行为，必须留痕。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    notes: str = Field(
        min_length=1,
        max_length=1000,
        description="审核意见，如『用法用量合理，无配伍禁忌』",
    )


class DispenseRequest(BaseModel):
    """发药确认。要求该处方已通过审核（DispensingRecord.verified=True）。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    notes: str | None = Field(
        default=None,
        max_length=1000,
        description="发药备注，如用药交代、患者签收情况（可选）",
    )


# ==================== 响应 ====================


class DispensingRecordOut(BaseModel):
    """发药记录（含处方快照字段，供药师工作台与患者取药视图共用）。"""

    id: str
    prescription_id: str
    pharmacist_id: str
    verified: bool = Field(description="是否已完成处方审核（先审后发的闸门）")
    verification_notes: str | None = None
    dispensed_at: datetime | None = None
    notes: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    # 处方侧冗余信息：让前端一屏看到「谁的什么方、多少钱、发没发」，
    # 省一次处方详情往返。这些不是 DispensingRecord 的列，由 service 装配。
    patient_id: str | None = None
    prescription_status: str | None = None
    total_amount: int | None = Field(default=None, description="处方金额，单位分")


class DispensingListResponse(BaseModel):
    items: list[DispensingRecordOut]
    total: int
    page: int
    page_size: int


__all__ = [
    "ApproveRequest",
    "DispenseRequest",
    "DispensingListResponse",
    "DispensingRecordOut",
]
