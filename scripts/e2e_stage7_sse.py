"""Stage7 批次0/1 联调验证：健康咨询 SSE 端点端到端实测。

不经浏览器，直接以 fetch 语义（requests 流式）验证：
  1. patient01 登录拿 token
  2. POST /health-qa/stream 收集 SSE 事件序列
  3. 断言 metadata -> token* -> done 顺序、guard、引用、用量字段
  4. 同一请求经 nginx 网关（8080）再走一遍，验证反代不缓冲 SSE
  5. 并发上限：占满后第 N+1 个请求应 409

用法：python scripts/e2e_stage7_sse.py
"""
import json
import sys
import threading
import time

import requests

API = "http://127.0.0.1:8010/api/v1"
GW = "http://127.0.0.1:8080/api/v1"
PASSWORD = "Imcs2026demo"

passed: list[str] = []
failed: list[str] = []


def check(name: str, ok: bool, extra: str = "") -> None:
    (passed if ok else failed).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' | ' + extra) if extra else ''}")


def login(base: str, account: str) -> str:
    r = requests.post(base + "/auth/login", json={"account": account, "password": PASSWORD}, timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]


def consume_sse(base: str, token: str, question: str, timeout: float = 120.0):
    """POST 流式端点，按 SSE 帧解析，返回 (events, 首token耗时ms或None)。"""
    events: list[tuple[str, dict]] = []
    t0 = time.monotonic()
    first_token_ms = None
    with requests.post(
        base + "/health-qa/stream",
        json={"question": question, "history": []},
        headers={"Authorization": f"Bearer {token}", "Accept": "text/event-stream"},
        stream=True,
        timeout=timeout,
    ) as r:
        if r.status_code != 200:
            return [("http_" + str(r.status_code), {"body": r.text[:300]})], None
        name = "message"
        data_lines: list[str] = []
        for raw in r.iter_lines(decode_unicode=False):
            if raw is None:
                continue
            line = raw.decode("utf-8", "replace")
            if line == "":
                if data_lines:
                    payload = json.loads("\n".join(data_lines))
                    if name == "token" and first_token_ms is None:
                        first_token_ms = (time.monotonic() - t0) * 1000
                    events.append((name, payload))
                name, data_lines = "message", []
            elif line.startswith(":"):
                continue
            elif line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
    return events, first_token_ms


def validate_stream(label: str, events, first_token_ms) -> None:
    names = [n for n, _ in events]
    check(f"{label}: 无HTTP层错误", not any(n.startswith("http_") for n in names), str(events[:1]) if events else "空流")
    if any(n.startswith("http_") or n == "error" for n in names):
        return
    check(f"{label}: metadata 先于 token", names and names[0] == "metadata" and "token" in names)
    check(f"{label}: 以 done 收尾", names[-1] == "done" if names else False)
    md = events[0][1] if events else {}
    check(f"{label}: metadata 含 trace_id/references", bool(md.get("trace_id")) and "references" in md,
          f"retained={md.get('retrieved_count')} zero={md.get('is_zero_result')}")
    do = events[-1][1] if events else {}
    check(f"{label}: done 含用量与免责声明", bool(do.get("usage")) and bool(do.get("disclaimer")))
    check(f"{label}: 正文非空", sum(len(d.get("text", "")) for n, d in events if n == "token") > 10)
    check(f"{label}: 流式（多token帧）", names.count("token") > 1, f"tokens={names.count('token')}")
    if first_token_ms is not None:
        print(f"        首token耗时 {first_token_ms:.0f}ms（网关缓冲过大时此项会接近总时长）")


def main() -> int:
    print("== 1. 登录 ==")
    token = login(API, "patient01")
    check("patient01 登录", bool(token))

    print("== 2. API 直连 SSE ==")
    ev, ftt = consume_sse(API, token, "最近总是头晕，可能是什么原因？")
    validate_stream("直连", ev, ftt)

    print("== 3. nginx 网关 SSE ==")
    ev2, ftt2 = consume_sse(GW, token, "糖尿病饮食要注意什么？")
    validate_stream("网关", ev2, ftt2)

    print("== 4. 急症守卫 ==")
    ev3, _ = consume_sse(API, token, "我现在胸口剧烈疼痛，喘不上气")
    guard = ev3[0][1].get("guard") if ev3 and ev3[0][0] == "metadata" else None
    check("急症命中 guard=emergency", guard == "emergency", f"guard={guard}")

    print("== 5. 并发上限 ==")
    # 服务上限由配置控制；占满后第 N+1 个应 HTTP 409
    holders: list[threading.Thread] = []
    results: list[str] = []
    lock = threading.Lock()

    def hammer():
        try:
            e, _ = consume_sse(API, token, "高血压多少算严重？", timeout=90)
            code = "ok"
            if e and e[0][0].startswith("http_"):
                code = e[0][0]
            elif e and e[0][0] == "error":
                code = "stream_error:" + str(e[0][1].get("code"))
            with lock:
                results.append(code)
        except Exception as exc:  # noqa: BLE001
            with lock:
                results.append("exc:" + str(exc)[:80])

    for _ in range(6):
        t = threading.Thread(target=hammer)
        t.start()
        holders.append(t)
        time.sleep(0.2)
    for t in holders:
        t.join(timeout=150)
    n409 = sum(1 for c in results if c == "http_409")
    print(f"        并发结果分布: {results}")
    check("并发超限返回409", n409 >= 1, f"409数={n409}/{len(results)}")

    print()
    print(f"通过 {len(passed)} 项，失败 {len(failed)} 项")
    for f in failed:
        print("  FAILED:", f)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
