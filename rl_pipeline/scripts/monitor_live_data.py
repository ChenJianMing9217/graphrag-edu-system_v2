"""
monitor_live_data.py — 治療師測試期間的監控儀表板

功能：
  - 每日 / 每小時統計治療師對話資料
  - Memory action 分布、Clarify 類型分布、反饋分布
  - 異常 query 偵測（task_top_score 低、回應時間長等）
  - 輸出 console + 寫 markdown 報告

用法：
  python rl_pipeline/scripts/monitor_live_data.py             # 看今日
  python rl_pipeline/scripts/monitor_live_data.py --days 7    # 看最近 7 天
  python rl_pipeline/scripts/monitor_live_data.py --watch     # 即時監控 (每 5 分鐘刷新)
  python rl_pipeline/scripts/monitor_live_data.py --export report.md
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime, timedelta
from collections import Counter, defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app import app, db, ChatMessage, Child


# ============================================================
# 工具函數
# ============================================================

def parse_flow(flow_state_json: str) -> dict:
    """安全 parse flow_state"""
    if not flow_state_json:
        return {}
    try:
        return json.loads(flow_state_json)
    except Exception:
        return {}


def fetch_recent_messages(days: int = 1):
    """取得最近 N 天的 bot 訊息（含 flow_state）"""
    cutoff = datetime.utcnow() - timedelta(days=days)
    with app.app_context():
        msgs = (
            ChatMessage.query
            .filter(ChatMessage.is_user_message == False)
            .filter(ChatMessage.sent_at >= cutoff)
            .order_by(ChatMessage.sent_at.desc())
            .all()
        )
        return [{
            "id": m.id,
            "msg_uuid": m.msg_uuid,
            "session_id": m.session_id,
            "child_id": m.child_id,
            "user_id": m.user_id,
            "sent_at": m.sent_at,
            "feedback_value": m.feedback_value,
            "flow_state": parse_flow(m.flow_state),
            "message_preview": (m.message[:80] if m.message else "") + ("..." if m.message and len(m.message) > 80 else ""),
        } for m in msgs]


def fetch_user_messages(days: int = 1):
    """取得最近 N 天的 user 訊息"""
    cutoff = datetime.utcnow() - timedelta(days=days)
    with app.app_context():
        msgs = (
            ChatMessage.query
            .filter(ChatMessage.is_user_message == True)
            .filter(ChatMessage.sent_at >= cutoff)
            .order_by(ChatMessage.sent_at.desc())
            .all()
        )
        return [{
            "child_id": m.child_id,
            "sent_at": m.sent_at,
            "message": m.message,
        } for m in msgs]


def fetch_active_children():
    """取得所有兒童帳號"""
    with app.app_context():
        children = Child.query.all()
        return {c.id: c.name for c in children}


# ============================================================
# 統計分析
# ============================================================

def analyze_messages(bot_msgs: list, user_msgs: list) -> dict:
    """從訊息列表產生完整統計"""
    if not bot_msgs:
        return {"empty": True}

    # 基礎統計
    total_turns = len(bot_msgs)
    total_user_queries = len(user_msgs)
    sessions = set(m["session_id"] for m in bot_msgs if m["session_id"])
    children = set(m["child_id"] for m in bot_msgs if m["child_id"])

    # 反饋分布
    feedback_counts = Counter()
    for m in bot_msgs:
        v = m["feedback_value"] or 0
        if v == 1:
            feedback_counts["positive"] += 1
        elif v == -1:
            feedback_counts["negative"] += 1
        else:
            feedback_counts["none"] += 1

    feedback_rate = (
        (feedback_counts["positive"] + feedback_counts["negative"])
        / total_turns * 100 if total_turns else 0
    )
    positive_rate = (
        feedback_counts["positive"]
        / max(feedback_counts["positive"] + feedback_counts["negative"], 1) * 100
    )

    # Memory action 分布
    memory_counts = Counter()
    clarify_counts = Counter()
    task_counts = Counter()
    domain_counts = Counter()
    low_score_queries = []  # task_top_score < 0.55 的可疑 query

    for m in bot_msgs:
        flow = m["flow_state"]
        memory_counts[flow.get("memory_action", "?")] += 1
        clarify_counts[flow.get("clarify_type") or "None"] += 1
        task_counts[flow.get("task_pred", "?")] += 1
        domain_counts[flow.get("top_domain", "?")] += 1

        # 收集 OOD 樣本
        ts = flow.get("task_top_score")
        if ts is not None and ts < 0.55:
            low_score_queries.append({
                "msg_uuid": m["msg_uuid"],
                "task_top_score": ts,
                "preview": m["message_preview"],
                "sent_at": m["sent_at"],
            })

    # 收集治療師關注的負反饋樣本
    negative_samples = [m for m in bot_msgs if m["feedback_value"] == -1]

    # Per-child 統計
    per_child = defaultdict(lambda: {"turns": 0, "positive": 0, "negative": 0})
    for m in bot_msgs:
        cid = m["child_id"] or 0
        per_child[cid]["turns"] += 1
        if m["feedback_value"] == 1:
            per_child[cid]["positive"] += 1
        elif m["feedback_value"] == -1:
            per_child[cid]["negative"] += 1

    return {
        "empty": False,
        "total_turns": total_turns,
        "total_user_queries": total_user_queries,
        "sessions": len(sessions),
        "children_count": len(children),
        "feedback_counts": dict(feedback_counts),
        "feedback_rate": feedback_rate,
        "positive_rate": positive_rate,
        "memory_counts": dict(memory_counts),
        "clarify_counts": dict(clarify_counts),
        "task_counts": dict(task_counts),
        "domain_counts": dict(domain_counts),
        "low_score_queries": low_score_queries[:20],   # 最多 20 筆
        "negative_samples": negative_samples[:20],     # 最多 20 筆
        "per_child": dict(per_child),
    }


# ============================================================
# 印出報告
# ============================================================

def print_report(stats: dict, child_name_map: dict):
    """印出 console 友善格式"""
    print("\n" + "=" * 70)
    print(f"  📊 治療師測試監控報告 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if stats.get("empty"):
        print("\n  ⚠️  無資料")
        return

    # 1. 整體規模
    print("\n  ① 整體規模")
    print(f"     總 turn 數:   {stats['total_turns']}")
    print(f"     User queries: {stats['total_user_queries']}")
    print(f"     Sessions:    {stats['sessions']}")
    print(f"     使用兒童帳號: {stats['children_count']}")

    # 2. 反饋分布
    print(f"\n  ② 反饋分布（治療師按讚/踩）")
    fb = stats["feedback_counts"]
    print(f"     👍 有幫助:  {fb.get('positive', 0)}")
    print(f"     👎 無幫助:  {fb.get('negative', 0)}")
    print(f"     未反饋:     {fb.get('none', 0)}")
    print(f"     反饋率:     {stats['feedback_rate']:.1f}%")
    if fb.get('positive', 0) + fb.get('negative', 0) > 0:
        print(f"     正面比例:   {stats['positive_rate']:.1f}%")

    # 3. Memory Action 分布
    print(f"\n  ③ Memory Action 分布")
    mem = stats["memory_counts"]
    total = sum(mem.values())
    for action in ["STAY", "REFRESH"]:
        cnt = mem.get(action, 0)
        pct = cnt / total * 100 if total else 0
        print(f"     {action:<10}: {cnt:4d} ({pct:5.1f}%)")

    # 4. Clarify 類型分布
    print(f"\n  ④ Clarify 類型分布")
    clarify = stats["clarify_counts"]
    total = sum(clarify.values())
    for ct in ["None", "DOMAIN_HARD", "CONTEXT_MISSING", "SLOT_REGION", "TASK_SOFT", "OUT_OF_DOMAIN"]:
        cnt = clarify.get(ct, 0)
        pct = cnt / total * 100 if total else 0
        if cnt > 0:
            print(f"     {ct:<18}: {cnt:4d} ({pct:5.1f}%)")

    # 5. Task 分布
    print(f"\n  ⑤ Task 分類分布（top 10）")
    sorted_tasks = sorted(stats["task_counts"].items(), key=lambda x: -x[1])[:10]
    for task, cnt in sorted_tasks:
        print(f"     {task:<5}: {cnt}")

    # 6. Domain 分布
    print(f"\n  ⑥ Domain 分布（top 5）")
    sorted_domains = sorted(stats["domain_counts"].items(), key=lambda x: -x[1])[:5]
    for domain, cnt in sorted_domains:
        print(f"     {domain:<15}: {cnt}")

    # 7. Per-child 使用統計
    print(f"\n  ⑦ 各兒童帳號使用統計")
    sorted_children = sorted(stats["per_child"].items(), key=lambda x: -x[1]["turns"])
    for cid, data in sorted_children[:10]:
        name = child_name_map.get(cid, f"unknown_{cid}")
        print(f"     [{cid}] {name:<20} turns={data['turns']:4d}  "
              f"👍={data['positive']:3d}  👎={data['negative']:3d}")

    # 8. 異常 query（low task_top_score）
    if stats["low_score_queries"]:
        print(f"\n  ⑧ 疑似偏離主題的 query（task_top_score < 0.55，最多 20 筆）")
        for q in stats["low_score_queries"][:10]:
            print(f"     [score={q['task_top_score']:.2f}] {q['preview']}")

    # 9. 治療師打 👎 的負反饋樣本
    if stats["negative_samples"]:
        print(f"\n  ⑨ 治療師打 👎 的回應（最多 20 筆，請人工檢視）")
        for s in stats["negative_samples"][:10]:
            print(f"     [{s['sent_at'].strftime('%m/%d %H:%M')}] {s['message_preview']}")

    print("\n" + "=" * 70)


def export_markdown(stats: dict, child_name_map: dict, path: str):
    """匯出 markdown 報告"""
    lines = [
        f"# 治療師測試監控報告",
        f"",
        f"產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## 整體規模",
        f"- 總 turn 數：{stats['total_turns']}",
        f"- Sessions：{stats['sessions']}",
        f"- 使用兒童帳號：{stats['children_count']}",
        f"",
        f"## 反饋分布",
        f"- 👍 有幫助：{stats['feedback_counts'].get('positive', 0)}",
        f"- 👎 無幫助：{stats['feedback_counts'].get('negative', 0)}",
        f"- 反饋率：{stats['feedback_rate']:.1f}%",
        f"- 正面比例：{stats.get('positive_rate', 0):.1f}%",
        f"",
        f"## Memory Action 分布",
    ]
    for action, cnt in stats["memory_counts"].items():
        lines.append(f"- {action}: {cnt}")
    lines.append("")
    lines.append("## Clarify 類型分布")
    for ct, cnt in stats["clarify_counts"].items():
        if cnt > 0:
            lines.append(f"- {ct}: {cnt}")

    if stats.get("negative_samples"):
        lines.append("")
        lines.append("## ⚠️ 負反饋樣本（需人工檢視）")
        for s in stats["negative_samples"]:
            lines.append(f"- `{s['sent_at']}` {s['message_preview']}")

    if stats.get("low_score_queries"):
        lines.append("")
        lines.append("## ⚠️ 疑似偏離主題 query")
        for q in stats["low_score_queries"]:
            lines.append(f"- score={q['task_top_score']:.2f} | {q['preview']}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[Export] 已寫入 {path}")


# ============================================================
# 主程式
# ============================================================

def run_once(days: int, export_path: str = None):
    """跑一次完整分析"""
    bot_msgs = fetch_recent_messages(days=days)
    user_msgs = fetch_user_messages(days=days)
    child_name_map = fetch_active_children()

    stats = analyze_messages(bot_msgs, user_msgs)
    print_report(stats, child_name_map)

    if export_path:
        export_markdown(stats, child_name_map, export_path)


def watch_mode(days: int, interval_sec: int = 300):
    """每 5 分鐘刷新一次"""
    print(f"\n[Watch Mode] 每 {interval_sec} 秒刷新，按 Ctrl+C 離開")
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        run_once(days)
        print(f"\n下次刷新: {(datetime.now() + timedelta(seconds=interval_sec)).strftime('%H:%M:%S')}")
        try:
            time.sleep(interval_sec)
        except KeyboardInterrupt:
            print("\n已離開 watch mode")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="治療師測試期間監控儀表板")
    parser.add_argument("--days", type=int, default=1, help="分析範圍（天，預設 1）")
    parser.add_argument("--watch", action="store_true", help="即時監控模式（每 5 分鐘刷新）")
    parser.add_argument("--interval", type=int, default=300, help="watch 模式刷新間隔（秒，預設 300）")
    parser.add_argument("--export", type=str, default=None, help="輸出 markdown 報告路徑")
    args = parser.parse_args()

    if args.watch:
        watch_mode(days=args.days, interval_sec=args.interval)
    else:
        run_once(days=args.days, export_path=args.export)
