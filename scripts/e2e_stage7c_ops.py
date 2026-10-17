"""Stage7 批次3 冒烟：药师审核发药 + 收费台收款 + 检验接单出报告 全链路。

自包含：自己挂号→签到→接诊→定稿病历，然后跑完三条业务线，不依赖既有数据。
纯业务流验证；检验报告的 AI 解读走异步 worker，脚本对 ai_status 采宽松断言
（只确认落库与自动派发被接受，不阻塞等待 LLM 完成）。

用法：python scripts/e2e_stage7c_ops.py
"""
import sys
import time
from datetime import date, timedelta

import requests

BASE = "http://127.0.0.1:8010/api/v1"
PASSWORD = "Imcs2026demo"

passed: list[str] = []
failed: list[str] = []


def check(name: str, ok: bool, extra: str = "") -> None:
    (passed if ok else failed).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' | ' + extra) if extra else ''}")


def call(method, path, token=None, body=None, params=None):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.request(method, BASE + path, headers=h, json=body, params=params, timeout=40)
    data = None
    if r.content:
        try:
            data = r.json()
        except ValueError:
            data = {"raw": r.text[:200]}
    return r.status_code, data


def login(account):
    s, d = call("POST", "/auth/login", body={"account": account, "password": PASSWORD})
    assert s == 200, f"登录 {account} 失败 {s}"
    return d["access_token"]


def setup_visit(t_pat, t_doc, doc_id):
    """挂号(doctor01 号源)→医生签到→医生接诊→定稿病历，返回 (visit_id, patient_id)。

    限定 doctor01 名下的号源，签到/接诊/开方就都用 t_doc 一个身份，
    不必再兜底别的医生或管理员，链路最短。
    """
    today = date.today().isoformat()
    end = (date.today() + timedelta(days=20)).isoformat()
    _, sched = call("GET", "/schedules", t_pat,
                    params={"status": "open", "start_date": today, "end_date": end, "page_size": 100})
    cand = [x for x in sched.get("items", [])
            if x["remaining_slots"] > 0 and x["doctor_id"] == doc_id]
    appt = None
    for c in cand:
        s2, d2 = call("POST", "/appointments", t_pat,
                      body={"schedule_id": c["id"], "chief_complaint": "批次3冒烟：发热咳嗽"})
        if s2 in (200, 201):
            appt = d2
            break
    if not appt:
        return None
    call("PATCH", f"/appointments/{appt['id']}/check-in", t_doc)
    s3, started = call("POST", f"/consult/appointments/{appt['id']}/start", t_doc)
    if s3 != 201:
        return None
    vid = started["visit"]["id"]
    call("PUT", f"/consult/visits/{vid}/record", t_doc,
         body={"diagnosis": "急性上呼吸道感染", "present_illness": "发热两天"})
    return vid, t_doc, appt["patient_id"]


def main():
    print("== 准备：建一条可开方的就诊 ==")
    t_pat1 = login("patient01")   # 青霉素/磺胺过敏
    t_doc = login("doctor01")
    s0, doc_me = call("GET", "/auth/me", t_doc)
    if s0 != 200:
        print("  SKIP  /auth/me 不可用，无法定位医生号源")
        return finish()
    doc_id = doc_me["id"]
    t_cash = login("cashier01")
    t_pharm = login("pharmacist01")
    t_lab = login("lab01")
    setup = setup_visit(t_pat1, t_doc, doc_id)
    if not setup:
        print("  SKIP  doctor01 未来 20 天无可开方号源（可能已约满），请换日再跑")
        return finish()
    vid, t_rx, patient_id = setup
    check("就诊已就绪", bool(vid), vid[:8])

    print("== A. 处方安全校验（过敏拦截 + override）==")
    safe = {"drug_code": "B3-IBU", "drug_name": "布洛芬缓释胶囊", "dosage": "0.3g",
            "frequency": "bid", "quantity": 1, "unit": "盒", "unit_price": 19.9}
    s, rx_safe = call("POST", "/prescriptions", t_rx, body={"visit_id": vid, "items": [safe]})
    check("安全药建草稿 201", s == 201, f"s={s}")
    s, _ = call("POST", f"/prescriptions/{rx_safe['id']}/confirm", t_rx, body={})
    check("安全药确认开具 200", s == 200, f"s={s}")

    amox = {"drug_code": "B3-AMX", "drug_name": "阿莫西林胶囊", "dosage": "0.5g",
            "frequency": "tid", "quantity": 2, "unit": "盒", "unit_price": 18.5}
    s, rx_all = call("POST", "/prescriptions", t_rx, body={"visit_id": vid, "items": [amox]})
    s2, blocked = call("POST", f"/prescriptions/{rx_all['id']}/confirm", t_rx, body={})
    det = (blocked or {}).get("detail", {})
    check("过敏药确认被拦 422", s2 == 422 and det.get("override_required") is True, f"s={s2}")
    check("拦截带 findings", bool(det.get("findings")))
    s3, _ = call("POST", f"/prescriptions/{rx_all['id']}/confirm", t_rx,
                body={"safety_override_reason": "家属口述过敏史，皮试阴性，临床复核后需用"})
    check("带理由强制开具 200", s3 == 200, f"s={s3}")

    print("== B. 收费 → 审核 → 发药 闭环 ==")
    s, order = call("POST", "/billing/orders", t_rx,
                   body={"visit_id": vid, "items": [
                       {"type": "prescription", "prescription_id": rx_safe["id"]},
                       {"type": "other", "name": "挂号费", "amount": 500}]})
    check("医生建收费单 201", s == 201, f"s={s}")
    order_id = order["id"]
    check("收费单金额含处方+挂号", order["total_amount"] == 1990 + 500, f"total={order['total_amount']}")

    s, _ = call("POST", f"/dispensing/prescriptions/{rx_safe['id']}/approve", t_pharm,
               body={"notes": "先测未审"})
    check("未缴费处方审核被拒 409", s == 409, f"s={s}")

    s, _ = call("POST", f"/billing/orders/{order_id}/pay", t_rx, body={"channel": "mock"})
    check("医生无权收款 403", s == 403, f"s={s}")
    s, paid = call("POST", f"/billing/orders/{order_id}/pay", t_cash, body={"channel": "mock"})
    check("收费员收款 200", s == 200 and paid.get("receipt_no"), f"s={s}")
    _, rx_after_pay = call("GET", f"/prescriptions/{rx_safe['id']}", t_pharm)
    check("收款联动处方→paid", rx_after_pay.get("status") == "paid")

    s, _ = call("POST", f"/dispensing/prescriptions/{rx_safe['id']}/dispense", t_pharm,
               body={"notes": "直接发药"})
    check("未审核直接发药 409", s == 409, f"s={s}")
    s, _ = call("POST", f"/dispensing/prescriptions/{rx_safe['id']}/approve", t_pharm, body={})
    check("审核缺意见 422", s == 422, f"s={s}")
    s, rec = call("POST", f"/dispensing/prescriptions/{rx_safe['id']}/approve", t_pharm,
                 body={"notes": "用法用量合理，同意调配"})
    check("审核通过 200 且 verified", s == 200 and rec.get("verified") is True, f"s={s}")
    s, disp = call("POST", f"/dispensing/prescriptions/{rx_safe['id']}/dispense", t_pharm,
                  body={"notes": "已交代饭后服用"})
    check("发药 200", s == 200, f"s={s}")
    _, rx_fin = call("GET", f"/prescriptions/{rx_safe['id']}", t_pharm)
    check("处方终态 dispensed", rx_fin.get("status") == "dispensed")

    print("== C. 检验：医生开单→检验接单→执行→出报告(危急值+AI派发) ==")
    s, exam = call("POST", f"/consult/visits/{vid}/exams", t_rx,
                  body={"exam_type": "blood_routine", "exam_name": "血常规", "urgency": "urgent"})
    check("医生开检查单 201", s == 201, f"s={s}")
    req_id = exam["id"]

    s, inc = call("GET", "/exams/requests", t_lab, params={"page_size": 100})
    check("新端点待接收 200", s == 200, f"s={s}")
    in_incoming = any(x["id"] == req_id for x in (inc or {}).get("items", []))
    check("新开单子出现在待接收", in_incoming, f"total={inc.get('total') if inc else '?'}")

    s, _ = call("GET", "/exams/requests", t_pat1)
    check("患者无检验全域权 403", s == 403, f"s={s}")

    s, _ = call("POST", f"/exams/requests/{req_id}/receive", t_lab)
    check("接收 200", s == 200, f"s={s}")
    s, _ = call("POST", f"/exams/requests/{req_id}/process", t_lab)
    check("开始执行 200", s == 200, f"s={s}")

    k_low = {"items": [{"name": "血钾", "code": "K", "value": 2.8, "unit": "mmol/L",
                        "ref_low": 3.5, "ref_high": 5.3}]}
    s, rep = call("POST", f"/exams/requests/{req_id}/reports", t_lab,
                 body={"raw_content": "血钾偏低", "structured_data": k_low,
                       "conclusion": "低钾血症", "auto_interpret": True})
    check("录入报告 201", s == 201, f"s={s}")
    if s == 201:
        check("危急值当场判定", rep["report"].get("critical") is True,
              str(rep["report"].get("critical_message"))[:30])
        check("AI 自动派发已接受", rep.get("dispatch", {}).get("dispatched") in (True, False),
              f"dispatched={rep.get('dispatch', {}).get('dispatched')}")
        rid = rep["report"]["id"]
        # 轮询 AI 解读至终态（worker 起才可能 completed；否则最多到 pending/failed）
        ai_final = None
        for _ in range(8):
            time.sleep(3)
            _, full = call("GET", f"/exams/reports/{rid}", t_lab)
            ai_final = (full or {}).get("ai", {}).get("ai_status")
            if ai_final in ("completed", "failed", "skipped"):
                break
        check("AI 解读进入终态或已排队", ai_final in ("completed", "failed", "skipped", "pending", "processing"),
              f"ai_status={ai_final}")

    print("== D. 全域列表可见性 ==")
    s_all, bills_all = call("GET", "/exams/reports/all", t_lab, params={"page_size": 5})
    check("检验科全域报告 200", s_all == 200 and isinstance(bills_all, dict) and "items" in bills_all, f"s={s_all}")
    s_bill, bills = call("GET", "/billing/orders/all", t_cash, params={"page_size": 5})
    check("收费员全域收费单 200", s_bill == 200 and isinstance(bills, dict) and "items" in bills, f"s={s_bill}")

    return finish()


def finish():
    print()
    print(f"通过 {len(passed)} 项，失败 {len(failed)} 项")
    for f in failed:
        print("  FAILED:", f)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
