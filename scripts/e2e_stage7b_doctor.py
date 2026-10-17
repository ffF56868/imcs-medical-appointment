"""Stage7 批次2 冒烟：医生端全链路（挂号→签到→队列→接诊→病历→检查→结案→越权）。

用法：python scripts/e2e_stage7b_doctor.py
纯业务流验证，不调 LLM、零积分消耗。
"""
import sys
from datetime import date, timedelta

import requests

BASE = "http://127.0.0.1:8010/api/v1"
PASSWORD = "Imcs2026demo"

passed: list[str] = []
failed: list[str] = []


def check(name: str, ok: bool, extra: str = "") -> None:
    (passed if ok else failed).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' | ' + extra) if extra else ''}")


def call(method: str, path: str, token: str | None = None, body: dict | None = None,
         params: dict | None = None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.request(method, BASE + path, headers=headers, json=body, params=params, timeout=30)
    data = None
    if r.content:
        try:
            data = r.json()
        except ValueError:
            data = {"raw": r.text[:200]}
    return r.status_code, data


def login(account: str) -> tuple[str, str]:
    s, d = call("POST", "/auth/login", body={"account": account, "password": PASSWORD})
    assert s == 200, f"登录 {account} 失败: {s} {d}"
    return d["access_token"], d["user"]["id"]


def main() -> int:
    print("== 1. 准备：患者02 挂号 + 签到 ==")
    t_pat2, _ = login("patient02")
    today = date.today().isoformat()
    end = (date.today() + timedelta(days=14)).isoformat()
    s, d = call("GET", "/schedules", t_pat2,
                params={"status": "open", "start_date": today, "end_date": end, "page_size": 50})
    if s != 200 or not d.get("items"):
        print(f"  SKIP  今天起14天内无可约号源（s={s}），种子排班可能过期")
        return 1
    # 找 doctor01 的号逐个试挂（已跑通过的时段会 409，顺延到下一个空档），
    # 保证脚本可反复重放；全部冲突才放弃。
    t_doc, doc_id = login("doctor01")
    doc_scheds = [x for x in d["items"] if x["doctor_id"] == doc_id] or d["items"]
    appt_id = ""
    appt_date = ""
    s, d2 = -1, None
    for cand in doc_scheds:
        s, d2 = call("POST", "/appointments", t_pat2,
                    body={"schedule_id": cand["id"], "chief_complaint": "反复头晕一周，测血压偏高"})
        if s in (200, 201):
            appt_id, appt_date = d2["id"], cand["schedule_date"]
            break
        if s != 409:
            break
    if not appt_id:
        print(f"  SKIP  doctor01 近期号源均与既有预约冲突（s={s}），请明天再跑")
        return finish()
    check("患者挂号成功", True, f"date={appt_date}")

    # 签到权限是 appointment:update（医生/管理员触发，见 appointment.py docstring），
    # 业务口径是「叫号时由医生端确认到诊」，这里用医生 token 执行。
    s, d = call("PATCH", f"/appointments/{appt_id}/check-in", t_doc)
    check("患者签到成功", s == 200 and d and d.get("status") == "checked_in", f"s={s}")

    print("== 2. 医生待诊队列 ==")
    s, d = call("GET", "/consult/appointments/pending", t_doc, params={"date": appt_date})
    check("队列接口 200", s == 200, f"s={s}")
    row = next((x for x in (d or {}).get("items", []) if x["appointment_id"] == appt_id), None)
    check("队列含刚签到的预约", row is not None)
    if not row:
        print("  队列明细：", [(x['appointment_id'][:8], x['status']) for x in (d or {}).get('items', [])][:10])
        return finish()
    check("队列回填主诉", (row.get("chief_complaint") or "").find("头晕") >= 0, str(row.get("chief_complaint"))[:30])
    check("队列初始无 visit_id", row.get("visit_id") is None)

    print("== 3. 开始接诊 ==")
    s, d = call("POST", f"/consult/appointments/{appt_id}/start", t_doc)
    check("开始接诊 201", s == 201, f"s={s}")
    if s != 201:
        return finish()
    visit = d["visit"]
    visit_id = visit["id"]
    check("病历带入主诉", (visit.get("chief_complaint") or "").find("头晕") >= 0)
    check("病历初始状态 in_progress", visit["status"] == "in_progress")
    check("响应含既往病历字段", "prior_visits" in d)

    print("== 4. 重复开始接诊应 409 ==")
    s, d = call("POST", f"/consult/appointments/{appt_id}/start", t_doc)
    check("重复接诊 409", s == 409, f"s={s}")

    print("== 5. 未定稿直接结案应 409 ==")
    s, d = call("PATCH", f"/consult/visits/{visit_id}/close", t_doc)
    check("无病历结案被拒 409", s == 409, f"s={s}")

    print("== 6. 书写病历（首次保存定稿）==")
    payload = {
        "present_illness": "一周前无明显诱因出现头晕，头位变动时加重，伴恶心，无呕吐、无视物旋转。",
        "past_history": "高血压病史2年，间断服药；无药物过敏史。",
        "physical_exam": "BP 150/95 mmHg，心率78次/分，神经系统检查未见定位体征。",
        "diagnosis": "高血压2级；耳源性眩晕待排",
        "icd_codes": ["I10", "h42.1 ", "I10"],
        "treatment_advice": "低盐饮食，规律服用降压药，一周后复测血压；必要时耳鼻喉科就诊。",
    }
    s, d = call("PUT", f"/consult/visits/{visit_id}/record", t_doc, body=payload)
    check("保存病历 200", s == 200, f"s={s}")
    if s == 200:
        check("触发定稿 transitioned", d.get("transitioned") is True)
        check("状态流转到 documented", d["visit"]["status"] == "documented")
        check("ICD 去重且大写规范化", d["visit"]["icd_codes"] == ["I10", "H42.1"], str(d["visit"]["icd_codes"]))
        check("诊断已落库", (d["visit"].get("diagnosis") or "").find("高血压") >= 0)

    print("== 7. 开检查单 ==")
    s, d = call("POST", f"/consult/visits/{visit_id}/exams", t_doc,
               body={"exam_type": "ecg", "exam_name": "心电图", "body_part": "胸部", "urgency": "urgent"})
    check("开单 201", s == 201, f"s={s}")
    if s == 201:
        check("表内项目带演示价", d.get("fee_amount") == 3000, f"fee={d.get('fee_amount')}")
        check("初始状态 ordered", d.get("status") == "ordered")
    # 用后端价目表外的真实临床项目验证「表外→费用0待核价」路径
    s, d = call("POST", f"/consult/visits/{visit_id}/exams", t_doc,
               body={"exam_type": "sleep_monitor", "exam_name": "多导睡眠监测", "urgency": "routine"})
    check("表外项目可开单且费用0", s == 201 and d.get("fee_amount") == 0, f"s={s}")
    s, d = call("GET", f"/consult/visits/{visit_id}/exams", t_doc)
    check("检查单列表 2 张", s == 200 and len(d or []) == 2, f"n={len(d or [])}")

    print("== 8. 队列回填 visit_id 且按钮语义正确 ==")
    s, d = call("GET", "/consult/appointments/pending", t_doc, params={"date": appt_date})
    row = next((x for x in (d or {}).get("items", []) if x["appointment_id"] == appt_id), None)
    check("队列回填 visit_id", row is not None and row.get("visit_id") == visit_id)
    check("队列回填 visit_status", row is not None and row.get("visit_status") == "documented")

    print("== 9. 结束就诊（跨实体事务）==")
    s, d = call("PATCH", f"/consult/visits/{visit_id}/close", t_doc)
    check("结案 200", s == 200, f"s={s}")
    if s == 200:
        check("病历已 closed", d["visit"]["status"] == "closed")
        check("预约同步 completed", d.get("appointment_status") == "completed")

    print("== 10. 结案后拒改 ==")
    s, d = call("PUT", f"/consult/visits/{visit_id}/record", t_doc,
               body={"diagnosis": "手滑改一下"})
    check("结案后保存被拒 409", s == 409, f"s={s}")
    s, d = call("PATCH", f"/consult/visits/{visit_id}/close", t_doc)
    check("重复结案 409", s == 409, f"s={s}")

    print("== 11. 行级隔离 ==")
    s, d = call("GET", f"/consult/visits/{visit_id}", t_pat2)
    check("患者可读自己的病历", s == 200 and d["id"] == visit_id, f"s={s}")
    s, _ = call("PUT", f"/consult/visits/{visit_id}/record", t_pat2, body={"diagnosis": "篡改"})
    check("患者不能写病历", s in (401, 403, 404), f"s={s}")
    t_pat1, _ = login("patient01")
    s, _ = call("GET", f"/consult/visits/{visit_id}", t_pat1)
    check("他人患者不可见（404 不泄露存在性）", s == 404, f"s={s}")
    s, d = call("GET", "/consult/appointments/pending", t_pat1)
    check("患者查队列返回空而非报错", s == 200 and (d or {}).get("total") == 0, f"s={s}")

    return finish()


def finish() -> int:
    print()
    print(f"通过 {len(passed)} 项，失败 {len(failed)} 项")
    for f in failed:
        print("  FAILED:", f)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
