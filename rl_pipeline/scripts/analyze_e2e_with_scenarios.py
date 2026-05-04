"""
進階 E2E 分析：把 DB 對話與 generated_scenarios.json 對齊，計算每個 scenario 的命中率。

核心邏輯：
1. 讀 generated_scenarios.json 拿到 (scenario_name, category, type, steps)
2. 從 MySQL 撈最近的 session
3. 用 user query 字串完全匹配，把 session → scenario
4. 對每個 turn 計算 task_hit (嚴格) / task_hit_compat (相容)
5. 分軌統計：原 79 個 vs 新 3 個 (H→D / K→D / H→C) vs OOD

用法：
    python rl_pipeline/scripts/analyze_e2e_with_scenarios.py
    python rl_pipeline/scripts/analyze_e2e_with_scenarios.py --hours 12
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import pymysql

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
from config import MYSQL_CONFIG  # noqa

# ── 從 test_e2e_scenarios 共用相容群定義 ──────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_e2e_scenarios import (  # noqa
    CATEGORY_TASK_MAP,
    TASK_COMPATIBILITY,
    OOD_CATEGORY,
    _build_compat_set,
)

SCENARIOS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_scenarios.json")


def connect():
    return pymysql.connect(
        host=MYSQL_CONFIG["host"], port=MYSQL_CONFIG["port"],
        user=MYSQL_CONFIG["user"], password=MYSQL_CONFIG["password"],
        database=MYSQL_CONFIG["database"], charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def fetch_sessions(since: datetime, conn):
    """撈出 since 以來的 session_id 對應的所有 messages，按 session 分群"""
    sql = """
        SELECT id, session_id, user_id, child_id, message, is_user_message,
               sent_at, flow_state
        FROM chat_message
        WHERE sent_at >= %s
        ORDER BY sent_at ASC, id ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (since,))
        rows = list(cur.fetchall())

    sessions = defaultdict(list)
    for r in rows:
        if r["session_id"]:
            sessions[r["session_id"]].append(r)
    return sessions


def session_to_turns(msgs):
    """把 session 內的 user/assistant 訊息配對成 turns"""
    turns = []
    pending_user = None
    for m in msgs:
        if m["is_user_message"]:
            pending_user = m
        else:
            if pending_user:
                fs = m.get("flow_state")
                parsed = None
                if fs:
                    try:
                        d = json.loads(fs)
                        parsed = {
                            "task_pred": d.get("task_pred") or d.get("task_label"),
                            "secondary_tasks": d.get("secondary_tasks") or [],
                            "memory_action": d.get("memory_action") or d.get("retrieval_action"),
                            "clarify_type": d.get("clarify_type"),
                            "task_top_score": d.get("task_top_score"),
                            "active_domains": d.get("active_domains") or [],
                        }
                    except json.JSONDecodeError:
                        pass

                turns.append({
                    "user_query": pending_user["message"].strip(),
                    "assistant_msg": m["message"],
                    "flow": parsed,
                    "sent_at": m["sent_at"],
                })
                pending_user = None
    return turns


def match_session_to_scenario(turns, scenarios):
    """
    用 turns 的 user_query 嘗試完全匹配某個 scenario.steps
    回傳 (scenario, match_count) 或 (None, 0)
    """
    if not turns:
        return None, 0

    first_q = turns[0]["user_query"]
    candidates = []
    for sc in scenarios:
        if sc["steps"] and sc["steps"][0].strip() == first_q:
            candidates.append(sc)

    if not candidates:
        return None, 0

    # 若多個 scenario T0 相同，找對到最多 turn 的
    best_sc = None
    best_match = 0
    for sc in candidates:
        n_match = 0
        for t, q in zip(turns, sc["steps"]):
            if t["user_query"] == q.strip():
                n_match += 1
            else:
                break
        if n_match > best_match:
            best_match = n_match
            best_sc = sc
    return best_sc, best_match


def task_hit(predicted_tasks, category):
    if category == OOD_CATEGORY:
        return True
    expected = CATEGORY_TASK_MAP.get(category, [])
    if not expected:
        return True
    return any(t in predicted_tasks for t in expected)


def task_hit_compat(predicted_tasks, category):
    if category == OOD_CATEGORY:
        return True
    compat = _build_compat_set(category)
    if not compat:
        return True
    return any(t in compat for t in predicted_tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--since", default=None)
    args = parser.parse_args()

    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d %H:%M")
    else:
        since = datetime.utcnow() - timedelta(hours=args.hours)

    print(f"[INFO] MySQL: {MYSQL_CONFIG['host']} / {MYSQL_CONFIG['database']}")
    print(f"[INFO] 撈訊息範圍：>= {since}")

    # 載 scenarios
    with open(SCENARIOS_FILE, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    print(f"[INFO] 載入 {len(scenarios)} 個 scenario from {SCENARIOS_FILE}")

    # 撈 DB
    conn = connect()
    try:
        sessions = fetch_sessions(since, conn)
    finally:
        conn.close()

    print(f"[INFO] 撈出 {len(sessions)} 個 session")

    # 對齊 sessions 到 scenarios
    matched = []   # list of (scenario, turns)
    unmatched_sessions = []

    for sid, msgs in sessions.items():
        turns = session_to_turns(msgs)
        if not turns:
            continue
        sc, n_match = match_session_to_scenario(turns, scenarios)
        if sc and n_match >= 1:
            matched.append({
                "scenario": sc,
                "turns": turns[:len(sc["steps"])],
                "match_count": n_match,
                "session_id": sid,
            })
        else:
            unmatched_sessions.append(sid)

    print(f"[INFO] 對齊：{len(matched)} 個 session 對到 scenario，{len(unmatched_sessions)} 個未對齊")

    # 統計
    print("\n" + "=" * 70)
    print("[Scenario 級別命中率]")
    print("=" * 70)

    total_turns = 0
    strict_hit = 0
    compat_hit = 0
    by_category = defaultdict(lambda: {"n_turn": 0, "n_strict": 0, "n_compat": 0, "n_scenarios": 0})
    by_type = defaultdict(lambda: {"n_turn": 0, "n_strict": 0, "n_compat": 0})

    new_categories = {"H→D", "K→D", "H→C"}
    new_scenario_results = []

    for entry in matched:
        sc = entry["scenario"]
        cat = sc.get("metadata", {}).get("category", "")
        sc_type = sc.get("metadata", {}).get("type", "")

        by_category[cat]["n_scenarios"] += 1

        for t in entry["turns"]:
            if not t["flow"]:
                continue
            tl = t["flow"]["task_pred"]
            sec = t["flow"]["secondary_tasks"]
            all_tasks = ([tl] if tl else []) + (sec or [])

            total_turns += 1
            sh = task_hit(all_tasks, cat)
            ch = task_hit_compat(all_tasks, cat)
            if sh: strict_hit += 1
            if ch: compat_hit += 1

            by_category[cat]["n_turn"] += 1
            by_category[cat]["n_strict"] += int(sh)
            by_category[cat]["n_compat"] += int(ch)
            by_type[sc_type]["n_turn"] += 1
            by_type[sc_type]["n_strict"] += int(sh)
            by_type[sc_type]["n_compat"] += int(ch)

        if cat in new_categories:
            new_scenario_results.append(entry)

    print(f"\n總體（{total_turns} turns from {len(matched)} scenarios）")
    print(f"  Task 嚴格命中率   : {strict_hit}/{total_turns} = {strict_hit / total_turns * 100:.1f}%")
    print(f"  Task 相容命中率   : {compat_hit}/{total_turns} = {compat_hit / total_turns * 100:.1f}%")

    print(f"\n依 type 分軌：")
    for t, s in sorted(by_type.items()):
        if s["n_turn"] == 0: continue
        print(f"  {t:<20} n={s['n_turn']:3}  strict={s['n_strict'] / s['n_turn'] * 100:5.1f}%  compat={s['n_compat'] / s['n_turn'] * 100:5.1f}%")

    print(f"\n依 category 分軌（top 20）：")
    sorted_cats = sorted(by_category.items(), key=lambda x: -x[1]["n_turn"])[:20]
    print(f"  {'Category':<10} {'#sc':>4} {'#turn':>6} {'Strict':>8} {'Compat':>8}")
    for cat, s in sorted_cats:
        if s["n_turn"] == 0: continue
        st = s["n_strict"] / s["n_turn"] * 100
        cp = s["n_compat"] / s["n_turn"] * 100
        marker = "  <-- NEW" if cat in new_categories else ""
        print(f"  {cat:<10} {s['n_scenarios']:>4} {s['n_turn']:>6}  {st:>6.1f}%  {cp:>6.1f}%{marker}")

    # 新加 3 個 scenario 的詳細逐 turn 結果
    print("\n" + "=" * 70)
    print(f"[新加 H→D / K→D / H→C scenario 詳細逐 turn]")
    print("=" * 70)
    if not new_scenario_results:
        print("  沒有對到新 scenario（可能 user query 不一致）")
    for entry in new_scenario_results:
        sc = entry["scenario"]
        cat = sc.get("metadata", {}).get("category")
        print(f"\n  [{cat}] {sc['name']}")
        for i, (step, t) in enumerate(zip(sc["steps"], entry["turns"])):
            if not t["flow"]:
                print(f"    T{i}: (no flow_state)")
                continue
            tl = t["flow"]["task_pred"] or "?"
            sec = t["flow"]["secondary_tasks"]
            sec_str = f"+{sec}" if sec else ""
            cl = t["flow"]["clarify_type"] or "-"
            mem = t["flow"]["memory_action"] or "-"
            score = t["flow"]["task_top_score"]
            score_str = f"{score:.2f}" if score is not None else "?"

            all_tasks = ([tl] if tl != "?" else []) + sec
            sh = task_hit(all_tasks, cat)
            ch = task_hit_compat(all_tasks, cat)
            mark = "✓" if sh else ("~" if ch else "✗")
            print(f"    T{i} [{mark}] task={tl}{sec_str:<10}  clarify={cl:<15}  mem={mem:<10}  score={score_str}")
            print(f"         q={step[:60]}")


if __name__ == "__main__":
    main()
