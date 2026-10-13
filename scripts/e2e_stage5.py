"""阶段5端到端验证：处方 → 收费 → 发药 全链路（真实 HTTP 打容器 API）。

覆盖链路：
  登录六角色 → 挂号 → 签到 → 开始接诊 → 写病历 →
  开处方(安全基线药) → 确认开具 →
  开阿莫西林 → 确认被过敏拦截(422) → 带 override 理由强行开具(200) →
  建收费单(处方+挂号费) → 重复关联同一处方被拒(409) →
  医生支付被拒(403) → 收费员支付(处方联动 paid) →
  药师未审核直接发药被拒(409) → 药师审核 → 药师发药(处方 dispensed) →
  已发药退费被拒(409) →
  患者读自己的处方/取药记录(200) → 他人患者读(404) → 药师读处方详情(200) →
  药师改处方被拒(403) → 患者支付被拒(403) →
  结束就诊(200) → 就诊后开方被拒(409)

用法：docker compose 起栈后，python scripts/e2e_stage5.py [--base http://127.0.0.1:8010]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEMO_PASSWORD = "Imcs2026demo"

PASS = 0
FAIL = 0


def call(method: str, url: str, token: str | None = None, body: dict | None = None,
         expect: tuple[int, ...] = (200,)):
    """发一次 HTTP 请求，返回 (status, json)。状态不符预期则记 FAIL 并返回。"""
    global PASS, FAIL
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    try:
        payload = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        payload = None
    mark = "✅" if status in expect else "❌"
    if status in expect:
        PASS += 1
    else:
        FAIL += 1
    label = f"{method} {url.split('/api/v1')[-1]}"
    print(f"{mark} {label} -> {status} (expect {expect})")
    if status not in expect and payload is not None:
        print(f"   body: {json.dumps(payload, ensure_ascii=False)[:300]}")
    return status, payload


def login(base: str, account: str) -> tuple[str, str]:
    """登录拿 access_token 与 user id。"""
    _, r = call("POST", f"{base}/api/v1/auth/login",
                body={"account": account, "password": DEMO_PASSWORD},
                expect=(200,))
    return r["access_token"], r["user"]["id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8010")
    args = ap.parse_args()
    base = args.base
    V1 = f"{base}/api/v1"

    # 0) 健康检查 + 六角色登录
    call("GET", f"{base}/health", expect=(200,))
    t_pat1, id_pat1 = login(base, "patient01")      # 张小明：青霉素/磺胺类过敏
    t_doc, id_doc = login(base, "doctor01")          # 王建国 心血管内科
    t_lab, _ = login(base, "lab01")
    t_cash, _ = login(base, "cashier01")
    t_pharm, _ = login(base, "pharmacist01")
    t_pat2, id_pat2 = login(base, "patient02")       # 李静：无青霉素过敏史

    # 1) 挂号链路：号源 → 预约 → 签到 → 接诊 → 病历
    _, sched = call("GET", f"{V1}/schedules?department_code=CARDIO",
                    token=t_pat1, expect=(200,))
    items = sched.get("items") or sched if isinstance(sched, list) else sched.get("items", [])
    slot = next((s for s in items if s.get("status") == "open"), items[0])
    appt_body_tpl = {"chief_complaint": "咳嗽三天，阶段5端到端验证"}
    appt_id = None
    for slot in items[:8]:
        body = {**appt_body_tpl, "schedule_id": slot["id"]}
        st, appt = call("POST", f"{V1}/appointments", token=t_pat1, body=body,
                        expect=(201, 200, 409))
        if st in (201, 200):
            appt_id = appt["id"]
            break
        if st == 409 and "detail" in appt:
            conflict = appt["detail"].get("appointment_conflict_id")
            if conflict:
                call("DELETE", f"{V1}/appointments/{conflict}", token=t_pat1,
                     expect=(200, 204, 409))
    assert appt_id, "未能创建预约"
    call("PATCH", f"{V1}/appointments/{appt_id}/check-in", token=t_doc,
         expect=(200,))
    _, started = call("POST", f"{V1}/consult/appointments/{appt_id}/start",
                      token=t_doc, expect=(200, 201))
    visit_id = started["visit"]["id"]
    call("PUT", f"{V1}/consult/visits/{visit_id}/record", token=t_doc,
         body={"diagnosis": "急性支气管炎", "present_illness": "咳嗽三天，低热"},
         expect=(200,))

    # 2) 处方：基线药开方 → 确认
    safe_item = {"drug_code": "E2E001", "drug_name": "布洛芬缓释胶囊",
                 "dosage": "300mg", "frequency": "bid", "quantity": 1,
                 "unit": "盒", "unit_price": 19.90}
    _, rx1 = call("POST", f"{V1}/prescriptions", token=t_doc,
                  body={"visit_id": visit_id, "diagnosis": "急性支气管炎",
                        "items": [safe_item]}, expect=(201,))
    assert rx1["total_amount"] == 1990, f"金额口径错误: {rx1['total_amount']}"
    call("POST", f"{V1}/prescriptions/{rx1['id']}/confirm", token=t_doc,
         body={}, expect=(200,))

    # 过敏拦截 → override 放行
    amox = {"drug_code": "E2E002", "drug_name": "阿莫西林胶囊",
            "dosage": "500mg", "frequency": "tid", "quantity": 2,
            "unit": "盒", "unit_price": 25.50}
    _, rx2 = call("POST", f"{V1}/prescriptions", token=t_doc,
                  body={"visit_id": visit_id, "items": [amox]}, expect=(201,))
    st, blocked = call("POST", f"{V1}/prescriptions/{rx2['id']}/confirm",
                       token=t_doc, body={}, expect=(422,))
    print(f"   拦截码={blocked['error']['code'] if 'error' in blocked else blocked}")
    call("POST", f"{V1}/prescriptions/{rx2['id']}/confirm", token=t_doc,
         body={"safety_override_reason": "药敏试验阴性，既往耐受"}, expect=(200,))

    # 3) 收费：建单 → 重复关联被拒 → 医生无权付 → 收费员付（处方联动 paid）
    _, order = call("POST", f"{V1}/billing/orders", token=t_doc,
                    body={"visit_id": visit_id, "items": [
                        {"type": "prescription", "prescription_id": rx1["id"]},
                        {"type": "other", "name": "挂号费", "amount": 500}]},
                    expect=(201,))
    assert order["total_amount"] == 2490, f"收费单金额错误: {order['total_amount']}"
    call("POST", f"{V1}/billing/orders", token=t_doc,
         body={"visit_id": visit_id, "items": [
             {"type": "prescription", "prescription_id": rx1["id"]}]},
         expect=(409,))
    call("POST", f"{V1}/billing/orders/{order['id']}/pay", token=t_doc,
         body={"channel": "mock"}, expect=(403,))
    call("POST", f"{V1}/billing/orders/{order['id']}/pay", token=t_pat1,
         body={"channel": "mock"}, expect=(403,))
    call("POST", f"{V1}/billing/orders/{order['id']}/pay", token=t_cash,
         body={"channel": "mock"}, expect=(200,))
    _, rx1_after = call("GET", f"{V1}/prescriptions/{rx1['id']}", token=t_doc, expect=(200,))
    assert rx1_after["status"] == "paid", f"支付联动失败: {rx1_after['status']}"

    # 4) 发药：先审后发
    call("POST", f"{V1}/dispensing/prescriptions/{rx1['id']}/dispense",
         token=t_pharm, body={}, expect=(409,))          # 未审核被拦
    _, rec = call("POST", f"{V1}/dispensing/prescriptions/{rx1['id']}/approve",
                  token=t_pharm, body={"notes": "四查十对无误"}, expect=(200,))
    call("POST", f"{V1}/dispensing/prescriptions/{rx1['id']}/dispense",
         token=t_pharm, body={"notes": "当面交付，用药交代完毕"}, expect=(200,))
    _, rx1_final = call("GET", f"{V1}/prescriptions/{rx1['id']}", token=t_doc, expect=(200,))
    assert rx1_final["status"] == "dispensed"

    # 药已发出 → 整单退费被拒
    call("POST", f"{V1}/billing/orders/{order['id']}/refund", token=t_cash,
         body={"reason": "想退"}, expect=(409,))

    # 5) 权限与隔离
    call("GET", f"{V1}/prescriptions/{rx1['id']}", token=t_pat1, expect=(200,))   # 本人可读
    call("GET", f"{V1}/prescriptions/{rx1['id']}", token=t_pat2, expect=(404,))   # 他人 404
    call("GET", f"{V1}/prescriptions/{rx1['id']}", token=t_pharm, expect=(200,))  # 药师 read_all
    call("PUT", f"{V1}/prescriptions/{rx1['id']}", token=t_pharm,
         body={"items": [safe_item]}, expect=(403,))                               # 药师不可改方
    _, disp_rec = call("GET", f"{V1}/dispensing/records", token=t_pat1, expect=(200,))
    assert disp_rec["total"] >= 1, "患者应能查到自己的取药记录"

    # 6) 闭环收尾：结案 → 结案后开方被拒
    call("PATCH", f"{V1}/consult/visits/{visit_id}/close", token=t_doc,
         body={"treatment_advice": "多饮水，一周后复诊"}, expect=(200,))
    call("POST", f"{V1}/prescriptions", token=t_doc,
         body={"visit_id": visit_id, "items": [safe_item]}, expect=(409,))

    print(f"\n===== 阶段5 e2e：{PASS} PASS / {FAIL} FAIL =====")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
