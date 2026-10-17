"""Stage7 批次4 冒烟：管理端三页面（监控/评测/知识库）接口链路验证。

只验证前端将要调用的真实路径与响应形状；不触发 RAGAS 与索引重建
（那两条烧积分/长占 worker，阶段6已容器级验证过）。
用法：python scripts/e2e_stage7d_admin.py
"""
import sys
from datetime import datetime, timedelta, timezone

import requests

BASE = "http://127.0.0.1:8010/api/v1"
PW = "Imcs2026demo"

ok = 0
fail = []


def check(name, cond, extra=""):
    global ok
    if cond:
        ok += 1
        print(f"  [OK] {name} {extra}")
    else:
        fail.append(name)
        print(f"  [FAIL] {name} {extra}")


def login(account):
    r = requests.post(f"{BASE}/auth/login", json={"account": account, "password": PW}, timeout=15)
    assert r.status_code == 200, f"login {account}: {r.text[:200]}"
    return r.json()["access_token"]


def get(path, token, params=None):
    r = requests.get(f"{BASE}{path}", headers={"Authorization": f"Bearer {token}"}, params=params, timeout=30)
    return r


def main():
    print("== 管理员登录 ==")
    t_adm = login("admin")
    check("admin 登录", bool(t_adm))

    print("== 监控页 ==")
    r = get("/monitoring/overview", t_adm, {"window_minutes": 60})
    check("GET /monitoring/overview", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        d = r.json()
        check("overview 字段齐", all(k in d for k in ("source", "bucket_count", "metrics", "queue")))
        check("metrics 含错误率", "llm_error_rate" in (d.get("metrics") or {}))

    r = get("/monitoring/metrics", t_adm, {"metrics": "llm_call_count,llm_error_rate", "limit": 480})
    check("GET /monitoring/metrics", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        d = r.json()
        check("metrics 响应结构", "items" in d and "metric_names" in d)

    r = get("/monitoring/llm-calls", t_adm, {"page": 1, "page_size": 15})
    check("GET /monitoring/llm-calls", r.status_code == 200)
    trace_id = None
    if r.status_code == 200:
        items = r.json()["items"]
        check("明细含 trace_id 列", all("trace_id" in it for it in items[:1]))
        trace_id = next((it["trace_id"] for it in items if it.get("trace_id")), None)

    if trace_id:
        r = get(f"/monitoring/traces/{trace_id}", t_adm)
        check("GET /monitoring/traces/{id}", r.status_code == 200, f"trace={trace_id[:8]}")
        if r.status_code == 200:
            d = r.json()
            check("回放四段结构", all(k in d for k in ("counts", "summary", "llm_calls", "timeline")))
    else:
        check("取到可回放的 trace_id", False, "窗口内无带 trace 的调用")

    print("== 评测页 ==")
    r = get("/evaluation/cases", t_adm, {"page": 1, "page_size": 20})
    check("GET /evaluation/cases", r.status_code == 200)
    if r.status_code == 200:
        items = r.json()["items"]
        check("题库非空", len(items) > 0, f"total={r.json()['total']}")
        if items:
            need = ("question", "expected_name", "expected_type", "alternative_names",
                    "answer_keywords", "citation_names", "expected_refusal", "category")
            check("题目字段齐", all(k in items[0] for k in need))

    for kind in ("retrieval", "quality"):
        r = requests.post(f"{BASE}/evaluation/runs", params={"kind": kind},
                          headers={"Authorization": f"Bearer {t_adm}"}, timeout=300)
        check(f"POST /evaluation/runs?kind={kind}", r.status_code == 201, f"status={r.status_code}")
        if r.status_code == 201:
            d = r.json()
            check(f"{kind} 运行含指标", d.get("total_count", 0) > 0 or d.get("metrics"),
                  f"count={d.get('total_count')}")

    r = get("/evaluation/runs", t_adm, {"page": 1, "page_size": 15})
    check("GET /evaluation/runs", r.status_code == 200 and r.json()["total"] > 0)
    run_id = r.json()["items"][0]["id"] if r.status_code == 200 and r.json()["items"] else None
    if run_id:
        r = get(f"/evaluation/runs/{run_id}", t_adm)
        check("GET /evaluation/runs/{id} 含明细", r.status_code == 200 and "results" in r.json())

    r = get("/evaluation/trends", t_adm, {"kind": "retrieval"})
    check("GET /evaluation/trends", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        d = r.json()
        check("趋势含 points 数组", isinstance(d.get("points"), list))

    print("== 知识库页 ==")
    r = get("/knowledge/documents", t_adm, {"page": 1, "page_size": 20})
    check("GET /knowledge/documents", r.status_code == 200)
    doc_id = None
    if r.status_code == 200:
        items = r.json()["items"]
        check("文档列表非空", len(items) > 0, f"total={r.json()['total']}")
        if items:
            doc_id = items[0]["id"]
            need = ("title", "doc_type", "source_tier", "index_status", "chunk_count",
                    "is_reviewed", "is_active", "content_length")
            check("摘要字段齐", all(k in items[0] for k in need))

    if doc_id:
        r = get(f"/knowledge/documents/{doc_id}", t_adm)
        check("GET /knowledge/documents/{id} 含正文", r.status_code == 200 and "content" in r.json())

    r = requests.post(f"{BASE}/knowledge/search", json={"query": "高血压首选药物"},
                      headers={"Authorization": f"Bearer {t_adm}"}, timeout=120)
    check("POST /knowledge/search", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        d = r.json()
        check("检索含 diagnostics", "diagnostics" in d and "results" in d)

    # 权限面：医生不得访问监控/评测（ai_monitor、evaluation 不在其集合）。
    # 知识库医生可读是设计内（rbac_matrix 第 114 行 knowledge:read），
    # 但患者没有该权限——用 patient01 验证管理性读权限不外溢。
    t_doc = login("doctor01")
    for path in ("/monitoring/overview", "/evaluation/cases"):
        r = get(path, t_doc)
        check(f"医生访问 {path} 被拒", r.status_code == 403, f"status={r.status_code}")
    t_pat = login("patient01")
    r = get("/knowledge/documents", t_pat)
    check("患者访问 /knowledge/documents 被拒", r.status_code == 403, f"status={r.status_code}")

    print()
    print(f"结果：{ok} 通过 / {len(fail)} 失败")
    if fail:
        print("失败项：" + "；".join(fail))
        sys.exit(1)


if __name__ == "__main__":
    main()

