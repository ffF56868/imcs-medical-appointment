"""业务状态机校验。

各业务实体的状态流转规则集中在此。转换矩阵定义在模型模块里
（如 ``clinical.APPOINTMENT_TRANSITIONS``），本模块提供统一的校验入口。

**为什么在 Service 层校验而不用数据库触发器或 CHECK 约束**：
- 触发器逻辑藏在数据库里，代码审查与单元测试都覆盖不到，出问题时排查成本高
- CHECK 约束无法表达「从状态 A 只能到 B 或 C」这类跨行状态依赖
- Service 层校验配合单元测试能穷举所有迁移路径，且错误信息可以带上
  当前状态与目标状态，前端能给出精确提示

代价是理论上可以绕过 Service 层直接写库导致状态非法。本项目所有写入
都经过 Service 层，不存在旁路，因此这个代价可以接受。
"""

from __future__ import annotations

from collections.abc import Iterable

from app.core.exceptions import InvalidStateTransitionError
from app.models.ai import AITaskStatus
from app.models.billing import CHARGE_ORDER_TRANSITIONS
from app.models.clinical import (
    APPOINTMENT_TRANSITIONS,
    EXAM_REQUEST_TRANSITIONS,
    VISIT_TRANSITIONS,
)
from app.models.pharmacy import PRESCRIPTION_TRANSITIONS

# AI 任务状态流转。retrying 可以回到 running，failed 允许人工重试。
AI_TASK_TRANSITIONS: dict[str, set[str]] = {
    AITaskStatus.PENDING.value: {
        AITaskStatus.RUNNING.value,
        AITaskStatus.CANCELLED.value,
        AITaskStatus.FAILED.value,
    },
    AITaskStatus.RUNNING.value: {
        AITaskStatus.COMPLETED.value,
        AITaskStatus.FAILED.value,
        AITaskStatus.RETRYING.value,
        AITaskStatus.CANCELLED.value,
    },
    AITaskStatus.RETRYING.value: {
        AITaskStatus.RUNNING.value,
        AITaskStatus.FAILED.value,
        AITaskStatus.CANCELLED.value,
    },
    # failed 允许回到 pending，支持人工触发重试
    AITaskStatus.FAILED.value: {AITaskStatus.PENDING.value},
    AITaskStatus.COMPLETED.value: set(),
    AITaskStatus.CANCELLED.value: set(),
}

# 实体名 -> 转换矩阵。新增业务实体时在此登记。
TRANSITION_TABLES: dict[str, dict[str, set[str]]] = {
    "appointment": APPOINTMENT_TRANSITIONS,
    "visit": VISIT_TRANSITIONS,
    "exam_request": EXAM_REQUEST_TRANSITIONS,
    "prescription": PRESCRIPTION_TRANSITIONS,
    "charge_order": CHARGE_ORDER_TRANSITIONS,
    "ai_task": AI_TASK_TRANSITIONS,
}

# 终态集合：到达后不可再变更。用于快速判断与前端禁用操作按钮。
TERMINAL_STATES: dict[str, frozenset[str]] = {
    "appointment": frozenset({"completed", "cancelled", "no_show"}),
    "visit": frozenset({"closed"}),
    "exam_request": frozenset({"reviewed"}),
    "prescription": frozenset({"dispensed", "cancelled"}),
    "charge_order": frozenset({"refunded", "cancelled"}),
    "ai_task": frozenset({"completed", "cancelled"}),
}


def can_transition(entity: str, current: str, target: str) -> bool:
    """判断状态迁移是否合法。未知实体名视为非法，避免静默放行。"""
    table = TRANSITION_TABLES.get(entity)
    if table is None:
        return False
    return target in table.get(current, set())


def assert_transition(entity: str, current: str, target: str) -> str:
    """校验状态迁移，非法则抛 ``InvalidStateTransitionError``。

    返回 target 便于链式赋值::

        order.status = assert_transition("charge_order", order.status, "paid")
    """
    if not can_transition(entity, current, target):
        allowed = sorted(TRANSITION_TABLES.get(entity, {}).get(current, set()))
        raise InvalidStateTransitionError(
            f"{entity} 无法从「{current}」变更为「{target}」"
            + (f"，允许的目标状态：{'、'.join(allowed)}" if allowed else "，该状态为终态"),
            current=current,
            target=target,
            entity=entity,
        )
    return target


def allowed_targets(entity: str, current: str) -> list[str]:
    """列出当前状态可迁移到的全部目标状态。

    前端据此渲染可用操作按钮，避免展示必然失败的操作。
    """
    return sorted(TRANSITION_TABLES.get(entity, {}).get(current, set()))


def is_terminal(entity: str, current: str) -> bool:
    """是否终态。"""
    return current in TERMINAL_STATES.get(entity, frozenset())


def validate_status_value(entity: str, status: str) -> str:
    """校验状态值本身是否属于该实体的合法状态集合。

    用于写入前拦截拼写错误——状态值写错会导致后续所有流转判断失效，
    且这类错误在测试里不容易发现（因为测试用的都是正确值）。
    """
    table = TRANSITION_TABLES.get(entity)
    if table is None:
        raise InvalidStateTransitionError(
            f"未知的业务实体：{entity}", current=status, entity=entity
        )
    valid = set(table.keys())
    # 终态可能只出现在 value 侧（作为迁移目标），需一并纳入合法集合
    for targets in table.values():
        valid |= targets
    if status not in valid:
        raise InvalidStateTransitionError(
            f"{entity} 的状态值非法：{status}，合法值为 {'、'.join(sorted(valid))}",
            current=status,
            entity=entity,
        )
    return status


def bulk_validate(
    entity: str, transitions: Iterable[tuple[str, str]]
) -> list[tuple[str, str, bool]]:
    """批量校验一组迁移，返回 (当前, 目标, 是否合法)。

    供单元测试穷举状态机所有路径使用——这是保证状态机正确性的
    最有效手段，比逐个手写断言更不容易漏。
    """
    return [
        (current, target, can_transition(entity, current, target))
        for current, target in transitions
    ]


__all__ = [
    "AI_TASK_TRANSITIONS",
    "TRANSITION_TABLES",
    "TERMINAL_STATES",
    "can_transition",
    "assert_transition",
    "allowed_targets",
    "is_terminal",
    "validate_status_value",
    "bulk_validate",
]
