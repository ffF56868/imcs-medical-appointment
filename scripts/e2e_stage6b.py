"""阶段6批次B端到端验证：AI 评测体系（真实 HTTP 打容器 API + PostgreSQL + evaluate 队列）。

覆盖链路：
admin 登录 → 题库就位（种子42条）→ 触发检索评测（同步出三指标+门禁）→
二次评测出 compared 档 → 趋势查询 → 明细回读（钉死"列表瘦身不抹库"回归）→
非管理员 403 → 派发 RAGAS（202）→ worker 真跑 LLM 四指标 → 轮询到 completed →
明细含四指标分数。

用法：栈已启动（含 worker-evaluate）后 `python scripts/e2e_stage6b.py`。
"""

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8010/api/v1"
PASSWORD = "Imcs2026demo"

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    (PASSED if ok else FAILED).append(name)
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f"  <- {detail}" if detail and not ok else ""))
    return ok


def call(method: str, path: str, token: str | None = None, body: dict | None = None):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode()
            return resp.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def login(username: str) -> str:
    status, data = call("POST", "/auth/login", body={"account": username, "password": PASSWORD})
    if status != 200 or not data:
        print(f"登录失败 {username}: HTTP {status} {data}")
        sys.exit(1)
    return data["access_token"]


def main() -> int:
    admin = login("admin")
    patient = login("patient01")
    check("管理员与患者登录", bool(admin) and bool(patient))

    # ---------- 1. 题库就位 ----------
    status, data = call("GET", "/evaluation/cases?limit=100", admin)
    total = (data or {}).get("total", 0) if status == 200 else 0
    check("评测题库种子就位（42题）", status == 200 and total == 42, f"HTTP {status} total={total}")
    refusal_qs = {
        c["question"] for c in (data or {}).get("items", []) if c.get("expected_refusal")
    }
    check("题库含拒答题", len(refusal_qs) >= 4, f"拒答 {len(refusal_qs)} 题")

    # ---------- 2. 检索评测（同步） ----------
    t0 = time.time()
    status, run1 = call("POST", "/evaluation/runs?kind=retrieval", admin)
    ok = status == 201 and run1 and run1.get("status") == "completed"
    check("触发检索评测并同步完成", ok, f"HTTP {status} {run1}")
    if ok:
        m = run1
        check(
            "三指标产出且为0-1小数",
            all(m.get(k) is not None and 0.0 <= m[k] <= 1.0
                for k in ("top1_accuracy", "recall_at_3", "precision_at_3")),
            str({k: m.get(k) for k in ("top1_accuracy", "recall_at_3", "precision_at_3")}),
        )
        check("评测覆盖全部非拒答题", m.get("total_count") == 42 - len(refusal_qs),
              f"total={m.get('total_count')}")
        check("首轮门禁为 baseline/compared",
              m.get("metrics", {}).get("gate", {}).get("status") in ("baseline", "compared"),
              str(m.get("metrics", {}).get("gate", {}).get("status")))
        results_qs = {r["question"] for r in m.get("results", [])}
        check("拒答题被检索评测排除", results_qs and not (results_qs & refusal_qs),
              f"交集 {results_qs & refusal_qs}")
        run1_id = m["id"]
    else:
        run1_id = None

    # ---------- 3. 二次评测出 compared + 明细不被抹（瘦身回归钉） ----------
    time.sleep(1.2)  # 保证 created_at 有先后
    status, run2 = call("POST", "/evaluation/runs?kind=retrieval", admin)
    if check("二次检索评测完成", status == 201 and run2 and run2.get("status") == "completed",
             f"HTTP {status}"):
        gate = run2.get("metrics", {}).get("gate", {})
        check("二次门禁为 compared 档", gate.get("status") == "compared", str(gate))
        check("两次指标同值（确定性评测）",
              run2.get("top1_accuracy") == run1.get("top1_accuracy"),
              f"{run2.get('top1_accuracy')} vs {run1.get('top1_accuracy')}")
    if run1_id:
        status, detail = call("GET", f"/evaluation/runs/{run1_id}", admin)
        # 列表瘦身 bug 的表现就是这里读回 None：库里明细必须还在
        check("首轮明细回读仍完整（列表瘦身未抹库）",
              status == 200 and detail and detail.get("results") is not None,
              f"HTTP {status} results={'None' if not detail else detail.get('results') is None}")

    # ---------- 4. 趋势 ----------
    status, trend = call("GET", "/evaluation/trends?kind=retrieval", admin)
    check("检索评测趋势含≥2个点", status == 200 and trend and len(trend.get("points", [])) >= 2,
          f"HTTP {status} points={len((trend or {}).get('points', []))}")

    # ---------- 5. 权限边界 ----------
    for path in ("/evaluation/cases", "/evaluation/runs", "/evaluation/trends?kind=retrieval"):
        status, _ = call("GET", path, patient)
        check(f"患者访问 {path.split('?')[0]} 被拒(403)", status == 403, f"HTTP {status}")
    status, _ = call("POST", "/evaluation/runs?kind=retrieval", patient)
    check("患者触发评测被拒(403)", status == 403, f"HTTP {status}")

    # ---------- 6. RAGAS 异步派发 → worker 真跑 ----------
    status, disp = call("POST", "/evaluation/ragas", admin, body={"sample_size": 3})
    ragas_run_id = None
    if status == 202 and disp and disp.get("eval_run_id") and disp.get("ai_task_id"):
        check("RAGAS 派发返回202并带任务号", True)
        st2, _ = call("POST", "/evaluation/ragas", admin, body={"sample_size": 3})
        check("评测进行中重复派发被拒(409)", st2 == 409, f"HTTP {st2}")
        ragas_run_id = disp["eval_run_id"]
    elif status == 409:
        # 上一轮脚本已派发过且仍在执行：直接接管轮询那条运行
        _, rows = call("GET", "/evaluation/runs?kind=ragas&limit=1", admin)
        items = (rows or {}).get("items", [])
        ragas_run_id = items[0]["id"] if items else None
        check("RAGAS 派发返回202并带任务号", bool(ragas_run_id),
              "409（上一轮运行仍在执行），接管轮询之")
    else:
        check("RAGAS 派发返回202并带任务号", False, f"HTTP {status} {disp}")

    if ragas_run_id:
        final = None
        deadline = time.time() + 540  # 9 分钟：3题×(生成+4指标judge) 的真 LLM 链路
        while time.time() < deadline:
            time.sleep(10)
            st, row = call("GET", f"/evaluation/runs/{ragas_run_id}", admin)
            if st == 200 and row and row.get("status") in ("completed", "failed"):
                final = row
                break
            if st == 200 and row:
                print(f"    … RAGAS 运行中 status={row.get('status')}")
        if check("RAGAS worker 完成执行", bool(final) and final.get("status") == "completed",
                 f"final={final and (final.get('status'), final.get('error_message'))}"):
            check("RAGAS 明细含抽样问答记录",
                  isinstance(final.get("results"), list) and len(final["results"]) >= 1,
                  f"results={type(final.get('results'))}")
            ragas = final.get("metrics", {}).get("ragas", {}) if isinstance(final.get("metrics"), dict) else {}
            # 顶层列也带四指标
            check("四指标分数落库（0-1）",
                  all(final.get(k) is not None and 0.0 <= float(final[k]) <= 1.0
                      for k in ("faithfulness", "answer_relevancy",
                                "context_precision", "context_recall")),
                  str({k: final.get(k) for k in ("faithfulness", "answer_relevancy",
                                                 "context_precision", "context_recall")}))
            print("    RAGAS 分数：faithfulness=%s relevancy=%s ctx_precision=%s ctx_recall=%s"
                  % tuple(final.get(k) for k in ("faithfulness", "answer_relevancy",
                                                 "context_precision", "context_recall")))

    print(f"\n用时 {time.time() - t0:.0f}s。结果：通过 {len(PASSED)}，失败 {len(FAILED)}")
    if FAILED:
        for name in FAILED:
            print(f"  FAIL: {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
