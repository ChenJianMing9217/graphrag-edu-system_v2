"""
從 MySQL ChatMessage 撈出最近的 E2E 測試紀錄，分析 Layer 1 修改效果。

用法：
    python rl_pipeline/scripts/analyze_e2e_from_db.py
    python rl_pipeline/scripts/analyze_e2e_from_db.py --hours 6
    python rl_pipeline/scripts/analyze_e2e_from_db.py --since "2026-04-29 14:00"

判斷 E2E 樣本：使用 access code 登入的 child_id 對應的對話紀錄。
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import pymysql

# 加入 app_v7 root 到 sys.path 才能 import config
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from config import MYSQL_CONFIG  # noqa


def connect():
    return pymysql.connect(
        host=MYSQL_CONFIG["host"],
        port=MYSQL_CONFIG["port"],
        user=MYSQL_CONFIG["user"],
        password=MYSQL_CONFIG["password"],
        database=MYSQL_CONFIG["database"],
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def fetch_messages(since: datetime, conn):
    """撈出指定時間後的所有 ChatMessage（含 user / assistant 兩邊）"""
    sql = """
        SELECT id, session_id, msg_uuid, user_id, child_id,
               message, is_user_message, sent_at, flow_state, feedback_value
        FROM chat_message
        WHERE sent_at >= %s
        ORDER BY sent_at ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (since,))
        return list(cur.fetchall())


def parse_flow(row):
    """從 flow_state JSON 拆出我們關心的欄位"""
    fs = row.get("flow_state")
    if not fs:
        return None
    try:
        d = json.loads(fs)
    except json.JSONDecodeError:
        return None

    return {
        "task_pred": d.get("task_pred") or d.get("task_label"),
        "task_dist": d.get("task_dist") or {},
        "secondary_tasks": d.get("secondary_tasks") or [],
        "task_top_score": d.get("task_top_score"),
        "memory_action": d.get("memory_action") or d.get("retrieval_action"),
        "clarify_type": d.get("clarify_type"),
        "scope_pred": d.get("scope_pred"),
        "active_domains": d.get("active_domains") or [],
        "top_domain": d.get("top_domain"),
        "detected_region": d.get("detected_region"),
    }


def analyze(messages):
    """彙整並印出 E2E 分析報告"""
    # 只取 assistant 訊息（user 訊息沒有 flow_state）
    assistant_msgs = [m for m in messages if not m["is_user_message"]]
    user_msgs = [m for m in messages if m["is_user_message"]]

    # 解析 flow_state
    parsed = []
    for m in assistant_msgs:
        f = parse_flow(m)
        if f:
            f["sent_at"] = m["sent_at"]
            f["session_id"] = m["session_id"]
            f["child_id"] = m["child_id"]
            f["feedback_value"] = m["feedback_value"]
            parsed.append(f)

    n = len(parsed)
    if n == 0:
        print("[警告] 沒有任何帶 flow_state 的 assistant 訊息")
        return

    print("=" * 70)
    print(f"E2E 資料分析（assistant 訊息 = {n} 條，user 訊息 = {len(user_msgs)} 條）")
    print(f"時間範圍：{parsed[0]['sent_at']}  ~  {parsed[-1]['sent_at']}")
    print("=" * 70)

    # ── 1. Session 統計 ──────────────────────────────────────────
    sess_count = len({m["session_id"] for m in assistant_msgs})
    child_count = len({m["child_id"] for m in assistant_msgs if m["child_id"]})
    print(f"\n[1] 資料規模")
    print(f"    Session 數          : {sess_count}")
    print(f"    Child 數            : {child_count}")
    print(f"    平均每 session turn : {n / sess_count if sess_count else 0:.1f}")

    # ── 2. Task 分類分布 ─────────────────────────────────────────
    task_counter = Counter(p["task_pred"] for p in parsed if p["task_pred"])
    print(f"\n[2] Task 分類分布（top 14 + OOD 預期）")
    print(f"    {'Task':<5} {'Count':>6} {'Pct':>7}")
    for t, c in sorted(task_counter.items(), key=lambda x: -x[1]):
        pct = c / n * 100
        print(f"    {t:<5} {c:>6} {pct:>6.1f}%")

    # ── 3. 多任務識別率（Layer 1 重點）─────────────────────────────
    multi = [p for p in parsed if p["secondary_tasks"]]
    multi_rate = len(multi) / n * 100
    print(f"\n[3] 多任務識別（Layer 1 修改重點）")
    print(f"    secondary_tasks 非空 : {len(multi)} / {n} = {multi_rate:.1f}%")
    if multi:
        sec_combos = Counter(
            f"{p['task_pred']}+{','.join(sorted(p['secondary_tasks']))}"
            for p in multi
        )
        print(f"    Top 10 主+副組合：")
        for combo, c in sec_combos.most_common(10):
            print(f"      {combo:<30} {c}")

    # ── 4. Clarify 觸發分布（TASK_SOFT 是 Layer 1 重點）──────────
    clarify_counter = Counter(p["clarify_type"] for p in parsed)
    print(f"\n[4] Clarify Type 分布（TASK_SOFT 是 Layer 1 重點）")
    for ct, c in sorted(clarify_counter.items(), key=lambda x: -x[1]):
        ct_str = "None" if ct is None else ct
        pct = c / n * 100
        print(f"    {ct_str:<20} {c:>6} {pct:>6.1f}%")

    # ── 5. Memory 動作分布 ──────────────────────────────────────
    mem_counter = Counter(p["memory_action"] for p in parsed)
    print(f"\n[5] Memory / Retrieval Action 分布")
    for m, c in sorted(mem_counter.items(), key=lambda x: -x[1]):
        m_str = "None" if m is None else m
        pct = c / n * 100
        print(f"    {m_str:<25} {c:>6} {pct:>6.1f}%")

    # ── 6. task_top_score 分布（OOD 偵測訊號）────────────────────
    scores = [p["task_top_score"] for p in parsed if p["task_top_score"] is not None]
    if scores:
        scores.sort()
        n_low = sum(1 for s in scores if s < 0.55)
        print(f"\n[6] task_top_score 分布（< 0.55 觸發 OUT_OF_DOMAIN）")
        print(f"    平均           : {sum(scores) / len(scores):.3f}")
        print(f"    中位數         : {scores[len(scores) // 2]:.3f}")
        print(f"    最低 / 最高    : {min(scores):.3f} / {max(scores):.3f}")
        print(f"    < 0.55 (OOD)   : {n_low} / {len(scores)} = {n_low / len(scores) * 100:.1f}%")

    # ── 7. Domain 分布 ──────────────────────────────────────────
    all_domains = [d for p in parsed for d in (p["active_domains"] or [])]
    if all_domains:
        dom_counter = Counter(all_domains)
        print(f"\n[7] Domain 分布（top 10 active）")
        for d, c in dom_counter.most_common(10):
            print(f"    {d:<25} {c}")

    # ── 8. Feedback 分布（如有治療師標）──────────────────────────
    fb_counter = Counter(p["feedback_value"] for p in parsed)
    if any(p["feedback_value"] != 0 for p in parsed):
        print(f"\n[8] Feedback 分布（1=讚 / -1=倒讚 / 0=未標）")
        for fb, c in sorted(fb_counter.items()):
            print(f"    {fb:>+3}    {c:>6}")

    # ── 9. 異常偵測 ─────────────────────────────────────────────
    print(f"\n[9] 可能異常情境")
    no_task = sum(1 for p in parsed if not p["task_pred"])
    if no_task:
        print(f"    無 task_pred         : {no_task}")
    very_low_score = sum(1 for p in parsed if p["task_top_score"] is not None and p["task_top_score"] < 0.40)
    if very_low_score:
        print(f"    task_top_score<0.40  : {very_low_score}（高度離題訊號）")
    long_sess = Counter(p["session_id"] for p in parsed)
    super_long = [s for s, c in long_sess.items() if c > 8]
    if super_long:
        print(f"    超長 session(>8 turn): {len(super_long)} 個")

    # ── 10. Layer 1 驗收指標彙整 ─────────────────────────────────
    print(f"\n" + "=" * 70)
    print("[Layer 1 修改驗收指標]")
    print("=" * 70)
    print(f"  * TASK_SOFT 觸發率   : {clarify_counter.get('TASK_SOFT', 0) / n * 100:.1f}%  (修改前約 5-10%，預期升至 15-25%)")
    print(f"  * 多任務識別率       : {multi_rate:.1f}%  (修改前約 10%，預期升至 30-40%)")
    print(f"  * OOD 偵測率         : {clarify_counter.get('OUT_OF_DOMAIN', 0) / n * 100:.1f}%  (應與 OOD scenario 比例對齊)")
    print(f"  * SLOT_REGION 觸發率 : {clarify_counter.get('SLOT_REGION', 0) / n * 100:.1f}%  (H/K 任務缺地區時)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=6.0,
                        help="撈最近 N 小時的訊息（預設 6）")
    parser.add_argument("--since", default=None,
                        help="指定起始時間（如 '2026-04-29 14:00'），覆寫 --hours")
    args = parser.parse_args()

    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d %H:%M")
    else:
        since = datetime.utcnow() - timedelta(hours=args.hours)

    print(f"[INFO] 連線 MySQL: {MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']} / {MYSQL_CONFIG['database']}")
    print(f"[INFO] 撈取時間範圍：>= {since}")

    try:
        conn = connect()
    except Exception as e:
        print(f"[ERROR] 連線失敗：{e}")
        print(f"[提示] 請確認 192.168.150.136 是否能連通，或檢查 config.py")
        sys.exit(1)

    try:
        msgs = fetch_messages(since, conn)
    finally:
        conn.close()

    print(f"[INFO] 撈出 {len(msgs)} 條訊息")
    if not msgs:
        print(f"[警告] 該時段無資料，請改 --hours 或 --since")
        return

    analyze(msgs)


if __name__ == "__main__":
    main()
