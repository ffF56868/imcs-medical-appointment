"""RBAC 权限矩阵定义。

权限矩阵集中在一个模块里声明，而不是散落在各路由的装饰器上，
理由有两个：
1. 医疗系统的权限边界是安全关键信息，必须能一眼看全、便于审计
2. 新增资源时容易漏配某个角色，集中声明能在 review 时发现

命名规则：``资源:动作``。资源用业务实体名，动作用 CRUD 语义 +
少量业务专属动作（如 approve 审核、dispense 发药、interpret 解读）。

关于 admin（超级管理员）：``require_permission`` 对 admin 直接放行，
因此它的临床业务权限无需逐条列举。但下面仍为 admin 声明了**系统管理类**
权限，目的是让权限矩阵自身完整可读——审计时能看清管理员的职责范围是
「用户、排班、知识库、监控」，而**刻意不包含**诊疗记录的读写。
"""

from __future__ import annotations

from app.models.auth import RoleName

# 资源清单。新增业务实体时在此登记，保证命名一致。
RESOURCES = (
    "appointment",
    "visit",
    "medical_record",
    "exam_request",
    "exam_report",
    "exam_interpretation",
    "prescription",
    "dispensing",
    "charge_order",
    "patient_profile",
    "schedule",
    "knowledge",
    "triage",
    "ai_monitor",
    "evaluation",
    "audit_log",
    "user",
)

# 动作清单
ACTIONS = (
    "create",
    "read",
    "read_all",  # 跨用户读取（如药师读全部待发药处方）
    "update",
    "delete",
    "approve",  # 审核通过
    "dispense",  # 发药
    "pay",  # 收费
    "interpret",  # 触发/查看 AI 解读
    "rebuild",  # 重建知识库索引
)

# 角色 -> 权限集合。这是整个系统的授权真相来源。
ROLE_PERMISSIONS: dict[RoleName, set[tuple[str, str]]] = {
    # ---------- 患者 ----------
    # 患者只能操作自己的数据，read_all 一律不给。
    # 行级隔离在 service 层用 patient_id 过滤，此处只控动作。
    RoleName.PATIENT: {
        ("patient_profile", "read"),
        ("patient_profile", "update"),
        ("triage", "create"),
        ("triage", "read"),
        ("appointment", "create"),
        ("appointment", "read"),
        ("appointment", "delete"),  # 取消自己的挂号
        ("schedule", "read"),  # 查可挂号源
        ("visit", "read"),  # 读自己的就诊记录
        ("medical_record", "read"),
        ("exam_request", "read"),
        ("exam_report", "read"),
        ("exam_interpretation", "read"),
        ("prescription", "read"),
        ("charge_order", "read"),
        ("dispensing", "read"),
    },
    # ---------- 医生 ----------
    # 医生可读自己患者的全部诊疗数据，但不能收费、不能发药、
    # 不能改检验报告——三权分立是医疗系统的基本要求。
    RoleName.DOCTOR: {
        ("patient_profile", "read"),
        ("appointment", "read"),
        ("appointment", "update"),  # 接诊时改为 in_progress
        ("visit", "create"),
        ("visit", "read"),
        ("visit", "update"),
        ("medical_record", "create"),
        ("medical_record", "read"),
        ("medical_record", "update"),
        ("exam_request", "create"),
        ("exam_request", "read"),
        ("exam_report", "read"),
        ("exam_interpretation", "read"),
        ("exam_interpretation", "interpret"),
        ("prescription", "create"),
        ("prescription", "read"),
        ("prescription", "update"),
        ("charge_order", "create"),  # 医生开单，收费员收钱
        ("charge_order", "read"),
        ("dispensing", "read"),
        ("triage", "read"),
        ("schedule", "read"),
        ("schedule", "update"),  # 维护自己的排班
        ("knowledge", "read"),
    },
    # ---------- 检验科 ----------
    # 只能操作检查申请与报告，接触不到处方与收费。
    # 不给 exam_request:create——开单权在医生，检验科只执行。
    RoleName.LAB_TECH: {
        ("exam_request", "read_all"),
        ("exam_request", "update"),  # 接收、开始检查
        ("exam_report", "create"),
        ("exam_report", "read_all"),
        ("exam_report", "update"),
        ("exam_interpretation", "interpret"),
        ("exam_interpretation", "read_all"),
        ("patient_profile", "read"),  # 核对样本身份需要
        ("knowledge", "read"),
    },
    # ---------- 收费员 ----------
    # 只能读与收，不能改医嘱内容。read_all 是必需的：
    # 收费员要处理全院患者的缴费，不限于某个医生。
    RoleName.CASHIER: {
        ("charge_order", "read_all"),
        ("charge_order", "pay"),
        ("charge_order", "update"),  # 退费
        ("prescription", "read_all"),  # 核对药费明细
        ("exam_request", "read_all"),  # 核对检查费明细
        ("patient_profile", "read"),
    },
    # ---------- 药师 ----------
    # 核心是 dispense 与 approve：药师审核处方后才发药。
    # 不给 prescription:update——药师不能改医生开的处方，
    # 有问题应驳回让医生重开，这是处方权的边界。
    RoleName.PHARMACIST: {
        ("prescription", "read_all"),
        ("prescription", "approve"),
        ("dispensing", "create"),
        ("dispensing", "read_all"),
        ("dispensing", "update"),
        ("patient_profile", "read"),  # 核对过敏史，用药安全必需
        ("charge_order", "read_all"),  # 确认已缴费才发药
        ("knowledge", "read"),
    },
    # ---------- 超级管理员 ----------
    # 只保留系统管理权限。刻意**不给**临床业务权限
    # （visit/prescription/exam_report 的读写）：
    # 管理员能改配置和账号，但不应该能篡改诊疗记录，
    # 否则审计日志就失去了意义。
    RoleName.ADMIN: {
        ("user", "create"),
        ("user", "read_all"),
        ("user", "update"),
        ("user", "delete"),
        ("schedule", "create"),
        ("schedule", "read_all"),
        ("schedule", "update"),
        ("schedule", "delete"),
        ("knowledge", "create"),
        ("knowledge", "read_all"),
        ("knowledge", "update"),
        ("knowledge", "delete"),
        ("knowledge", "approve"),
        ("knowledge", "rebuild"),
        ("ai_monitor", "read_all"),
        ("evaluation", "create"),
        ("evaluation", "read_all"),
        ("evaluation", "update"),
        ("evaluation", "delete"),
        ("audit_log", "read_all"),
    },
}

# 角色展示信息。id 固定，便于种子数据幂等写入与权限矩阵可读性。
ROLE_DEFINITIONS: tuple[tuple[int, RoleName, str], ...] = (
    (1, RoleName.PATIENT, "患者：智能分诊、预约挂号、查看诊疗记录与报告解读"),
    (2, RoleName.DOCTOR, "医生：接诊、书写病历、开检查单与处方"),
    (3, RoleName.LAB_TECH, "检验科：接收检查申请、录入报告、触发 AI 解读"),
    (4, RoleName.CASHIER, "收费员：核对费用、收费与退费"),
    (5, RoleName.PHARMACIST, "药师：审核处方、发药、用药安全把关"),
    (6, RoleName.ADMIN, "超级管理员：用户与排班管理、知识库维护、AI 运行监控"),
)


def all_permissions() -> set[tuple[str, str]]:
    """全部角色的权限并集。种子数据据此写入 permissions 表。"""
    result: set[tuple[str, str]] = set()
    for permissions in ROLE_PERMISSIONS.values():
        result |= permissions
    return result


def permissions_for_role(role: RoleName) -> set[tuple[str, str]]:
    """取某角色的权限集合。"""
    return set(ROLE_PERMISSIONS.get(role, set()))


def validate_matrix() -> list[str]:
    """校验权限矩阵自身的合法性，返回问题列表（空列表表示无问题）。

    检查两类错误：
    1. 矩阵里用了未在 ``RESOURCES`` / ``ACTIONS`` 登记的资源名或动作名
       ——拼写错误（如把 ``dispense`` 写成 ``dispencing``）会让权限
       静默失效：授权记录写进去了，但 ``require_permission`` 查的是
       正确拼写，于是该角色永远拿不到这个权限，且不报任何错。
    2. 角色是否都已在矩阵中声明——新增角色时容易忘记配权限，
       结果是能登录但什么都做不了。

    在种子脚本写库前调用，把这类错误暴露在启动阶段而不是运行期。
    """
    problems: list[str] = []
    valid_resources = set(RESOURCES)
    valid_actions = set(ACTIONS)

    for role, permissions in ROLE_PERMISSIONS.items():
        for resource, action in sorted(permissions):
            if resource not in valid_resources:
                problems.append(f"角色 {role.value} 使用了未登记的资源：{resource}")
            if action not in valid_actions:
                problems.append(f"角色 {role.value} 使用了未登记的动作：{action}")

    for role in RoleName:
        if role not in ROLE_PERMISSIONS:
            problems.append(f"角色 {role.value} 未声明任何权限")

    return problems


__all__ = [
    "RESOURCES",
    "ACTIONS",
    "ROLE_PERMISSIONS",
    "ROLE_DEFINITIONS",
    "all_permissions",
    "permissions_for_role",
    "validate_matrix",
]
