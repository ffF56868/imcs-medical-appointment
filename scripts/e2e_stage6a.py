"""阶段6批次A端到端验证：可观测链路（真实 HTTP 打容器 API + PostgreSQL）。

覆盖：
  登录 → 响应头带 X-Trace-Id →
  触发一次知识检索（写入 rag_retrieval_logs，带自定义 trace_id）→
  触发一次急症分诊（写入 workflow_traces + triage_results）→
  admin 按 trace_id 回放检索链路（retrievals 段非空）→
  admin 按分诊 trace_id 回放（workflows 段非空、branch_taken=emergency）→
  admin 手动聚合当前分钟桶（ok=True）→
  admin 看板 overview（结构字段齐全）→
  admin 取 llm 调用明细（embed 调用应已落库）→
  非管理员访问监控端点被拒（403）→
  非法 trace_id 回放 404

用法：起栈后 python scripts/e2e_stage6a.py [--base http://127.0.0.1:8010]
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


def call(method, url, token=None, body=None, expect=(200,), *, with_headers=False):
    global PASS, FAIL
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status, raw, hdrs = resp.status, resp.read(), resp.headers
    except urllib.error.HTTPError as e:
        status, raw, hdrs = e.code, e.read(), e.headers
    try:
        payload = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        payload = None
    ok = status in expect
    PASS += 1 if ok else 0
    FAIL += 0 if ok else 1
    mark = "✅" if ok else "❌"
    label = f"{method} {url.split('/api/v1')[-1]}"
    print(f"{mark} {label} -> {status} (expect {expect})")
    if not ok and payload is not None:
        print(f"   body: {json.dumps(payload, ensure_ascii=False)[:280]}")
    return (status, payload, hdrs) if with_headers else (status, payload)


def login(base, account):
    _, r = call("POST", f"{base}/api/v1/auth/login",
                body={"account": account, "password": DEMO_PASSWORD}, expect=(200,))
    return r["access_token"] if r else None


def check(name, cond):
    """纯断言型检查，不打 HTTP。"""
    global PASS, FAIL
    mark = "✅" if cond else "❌"
    print(f"{mark} 断言：{name}")
    PASS += 1 if cond else 0
    FAIL += 0 if cond else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8010")
    args = ap.parse_args()
    base = args.base
    V1 = f"{base}/api/v1"

    call("GET", f"{base}/health", expect=(200,))

    t_pat = login(base, "patient01")
    t_doc = login(base, "doctor01")
    # admin 用户名：种子用 admin，兼容可能的 admin01
    t_adm = login(base, "admin") or login(base, "admin01")
    assert t_pat and t_doc and t_adm, "登录失败，检查种子账号"

    # 1) 响应头带 X-Trace-Id（中间件生效）
    _, _, hdrs = call("GET", f"{V1}/auth/me", token=t_pat, expect=(200,), with_headers=True)
    check("响应头含 X-Trace-Id", bool(hdrs.get("X-Trace-Id")))

    # 2) 触发一次知识检索，带自定义 trace_id → 落 rag_retrieval_logs
    #    knowledge:read 只有医生/药师/管理员持有（患者没有），故用医生令牌
    trace_rag = ("e2e-rag-" + "0" * 36)[:36]
    call("POST", f"{V1}/knowledge/search", token=t_doc,
         body={"query": "布洛芬成人一次吃多少", "trace_id": trace_rag}, expect=(200,))

    # 3) 触发一次急症分诊（零 LLM、走急诊分支）→ workflow_traces + triage_results
    _, tri = call("POST", f"{V1}/triage/assess", token=t_pat,
                  body={"chief_complaint": "突然胸痛倒地意识不清叫不醒"}, expect=(201,))
    tri_trace = (tri or {}).get("trace_id")
    check("分诊返回 trace_id", bool(tri_trace))
    check("分诊命中急症分支", (tri or {}).get("decision_source") == "emergency")

    # 4) admin 回放检索链路：retrievals 段应命中那条自定义 trace
    _, rec = call("GET", f"{V1}/monitoring/traces/{trace_rag}", token=t_adm, expect=(200,))
    check("检索链路回放 retrievals 段非空", bool(rec) and rec.get("counts", {}).get("retrievals", 0) >= 1)

    # 5) admin 回放分诊链路：workflows 段非空，branch_taken=emergency
    _, rec2 = call("GET", f"{V1}/monitoring/traces/{tri_trace}", token=t_adm, expect=(200,))
    wfs = (rec2 or {}).get("workflows") or []
    check("分诊链路回放 workflows 段非空", len(wfs) >= 1)
    if wfs:
        check("回放工作流分支为急诊", wfs[0].get("branch_taken") == "emergency")
    check("分诊链路含 triage_results 段", (rec2 or {}).get("counts", {}).get("triage_results", 0) >= 1)

    # 6) admin 手动聚合当前分钟桶（可能写 0 行，但必须 ok=True 不报错）
    from datetime import datetime, timezone
    bucket = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()
    _, agg = call("POST", f"{V1}/monitoring/aggregate", token=t_adm,
                  body={"bucket_start": bucket}, expect=(200,))
    check("聚合返回 ok=True", bool(agg) and agg.get("ok") is True)

    # 7) admin 看板 overview 结构齐全
    _, ov = call("GET", f"{V1}/monitoring/overview", token=t_adm, expect=(200,))
    check("overview 含 metrics 与 queue", bool(ov) and "metrics" in ov and "queue" in ov)

    # 8) admin 取 llm 调用明细（检索 embed 若成功应已落库；至少结构合法）
    _, calls = call("GET", f"{V1}/monitoring/llm-calls", token=t_adm,
                    expect=(200,), body=None)
    check("llm-calls 返回 items/total", bool(calls) and "items" in calls and "total" in calls)

    # 9) 非管理员访问监控被拒 403
    call("GET", f"{V1}/monitoring/overview", token=t_doc, expect=(403,))
    call("GET", f"{V1}/monitoring/llm-calls", token=t_pat, expect=(403,))

    # 10) 不存在/无链路的 trace_id 回放 → 404
    call("GET", f"{V1}/monitoring/traces/no-such-trace-id-0000", token=t_adm, expect=(404,))

    print(f"\n===== 阶段6批次A e2e：{PASS} PASS / {FAIL} FAIL =====")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
