"""
快速驗證 4 個 domain anchor patch 是否在部署機生效。

跑 8 個關鍵 query（每題開新對話避免上下文污染），
對照 v1 預測與 v2 期望。
"""
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
CASES_FILE = Path(__file__).resolve().parent / "cases.json"

# 8 個驗證題：(編號, 描述, query, v1 預測, v2 期望)
CHECKS = [
    ("P1-A", "粗大動作 / 動作對日常影響",
     "報告裡寫到的動作表現，這對孩子平常生活會有什麼影響？",
     "情緒行為與社會適應功能", "粗大動作"),
    ("P1-B", "粗大動作 / 動作上吃力教室",
     "報告提到孩子在動作上比較吃力的地方，這在教室裡可能會表現在哪些地方？",
     "認知功能", "粗大動作"),
    ("P1-C", "粗大動作 / 大肌肉體能課",
     "在大肌肉活動或體能課時，老師可以怎麼協助孩子參與?",
     "認知功能", "粗大動作"),
    ("P2",   "情緒 / 不要做說累",
     "如果孩子做一下就說累、不要做，家長應該怎麼調整?",
     "認知功能", "情緒行為與社會適應功能"),
    ("P3-A", "整體概況 / 整理重點 (sanity)",
     "我是孩子的老師，想請你幫我整理這份聯評報告中，和學校生活最有關的重點。",
     "整體概況", "整體概況"),
    ("P3-B", "整體概況 / 跨專業整合 (sanity)",
     "報告裡每個專業都有建議，我需要全部都做到嗎?",
     "整體概況", "整體概況"),
    ("P4-A", "口語表達修正 / 擔心上學",
     "這樣是不是代表孩子以後上學會很困難?",
     "口語表達", "情緒行為與社會適應功能"),
    ("P4-B", "口語表達修正 / 等待期",
     "如果目前排不到治療，我在等待期間可以先做什麼?",
     "口語表達", "整體概況"),
]


def call_chat(session, base_url, query):
    r = session.post(f"{base_url}/api/chat_stream",
                     json={"message": query, "response_length": "concise"},
                     stream=True, timeout=240,
                     headers={"Accept": "text/event-stream"})
    r.raise_for_status()
    flow_state = None
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        try:
            ev = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "done":
            flow_state = ev.get("flow_state")
        elif ev.get("type") == "error":
            return None
    return flow_state


def main():
    cfg = json.load(open(CASES_FILE, encoding="utf-8"))
    base_url = cfg["base_url"]
    code = cfg["cases"][0]["access_code"]  # CASE_A

    s = requests.Session()
    r = s.post(f"{base_url}/api/login_with_code", json={"code": code}, timeout=30)
    r.raise_for_status()
    print(f"Login: {r.json().get('child_name')} (CASE_A)")
    print(f"Base: {base_url}\n")

    results = []
    for tag, desc, q, v1_pred, expected in CHECKS:
        s.post(f"{base_url}/api/new_chat", json={}, timeout=30)  # fresh chat
        t0 = time.time()
        fs = call_chat(s, base_url, q)
        dt = round(time.time() - t0, 1)
        v2 = fs.get("top_domain") if fs else None
        active = ", ".join(fs.get("active_domains") or []) if fs else ""
        prob = fs.get("top_domain_prob") if fs else None
        ok = "✓" if v2 == expected else ("△" if expected in (fs.get("active_domains") or []) else "✗")
        results.append((tag, desc, v1_pred, v2, expected, ok, active, prob, dt))
        print(f"[{tag}] {desc} ({dt}s)")
        print(f"  Q: {q[:50]}...")
        print(f"  v1 預測: {v1_pred}")
        print(f"  v2 預測: {v2}  prob={prob:.3f}" if prob else f"  v2 預測: {v2}")
        print(f"  期望  : {expected}  -> {ok}")
        print(f"  active: {active}")
        print()

    s.post(f"{base_url}/api/logout", json={}, timeout=10)

    print("=" * 80)
    print(f"{'Tag':<8} {'v1 -> v2 比對':<60} {'結果'}")
    print("=" * 80)
    fixed = 0
    regressed = 0
    same_correct = 0
    for tag, desc, v1, v2, exp, ok, active, prob, dt in results:
        line = f"{tag:<8} v1={v1[:14]:<14} -> v2={(v2 or '?')[:14]:<14} (期望={exp[:14]:<14}) {ok}"
        print(line)
        if ok == "✓" and v1 != exp:
            fixed += 1
        elif ok == "✓" and v1 == exp:
            same_correct += 1
        elif ok == "✗" and v1 == exp:
            regressed += 1
    print("=" * 80)
    print(f"  修正: {fixed}/{len(results) - same_correct}    維持正確: {same_correct}    退化: {regressed}")


if __name__ == "__main__":
    main()
