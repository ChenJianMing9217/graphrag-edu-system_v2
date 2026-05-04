"""
gpt_relabel_db_samples.py — 用 GPT 為 SQL ChatMessage 標記 Memory Action

目的：
  把 SQL 資料庫中既有的對話樣本（已有 flow_state 但無 ground truth Memory action 標籤）
  用 GPT 重新標記為 STAY / REFRESH，產生額外的 SFT 訓練資料。

設計理念：
  - 每筆 turn 給 GPT 看：prev_query、current_query、DST 特徵（tv/overlap/entropy）
  - GPT 根據早療對話脈絡判斷該 STAY 還是 REFRESH
  - 輸出 JSONL，可加入 sft_dataset 重新預訓練

成本估計：
  - 每筆 turn ≈ 1 次 LLM 呼叫，~600 tokens
  - 282 筆 ≈ $5-6 USD（gpt-5.4-mini）

用法：
  python rl_pipeline/scripts/gpt_relabel_db_samples.py
  python rl_pipeline/scripts/gpt_relabel_db_samples.py --max-samples 50  # 試跑
  python rl_pipeline/scripts/gpt_relabel_db_samples.py --output sft_relabel.jsonl
"""
import os
import sys
import json
import argparse
import re
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from openai import OpenAI

# ============================================================
# 設定
# ============================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = "gpt-5.4-mini"

DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(__file__),
    "../dataset/sft_relabel_from_db.jsonl"
)


# ============================================================
# GPT 標記函數
# ============================================================

SYSTEM_PROMPT = """你是早療對話系統的標註專家。系統有一個 Memory Agent 負責判斷
「使用者本輪 query 應該延續上一輪話題（STAY）還是切換新話題（REFRESH）」。

請根據以下資訊判斷：
1. STAY = 與前一輪同一個主題/領域延續，可以沿用前一輪的 domain
2. REFRESH = 切換到新的主題/領域，應該重新檢索

判斷依據（重要性由高到低）：
- DST 信號：tv_distance（高=切換）、topic_overlap（高=延續）
- query 內容：是否提及新的領域關鍵詞（粗大動作、認知、語言、情緒、補助等）
- 邏輯延續性：「那」「還有」「另外」=可能延續或新話題；明確新主題=REFRESH

特別規則：
- 第一輪（turn_idx=0）一律是 REFRESH（無前文可延續）
- 純 follow-up（「為什麼？」「具體呢？」）= STAY
- 用戶從報告解讀跳到居家訓練 = REFRESH（domain 變了）
- 用戶從粗大動作問同領域不同方面 = STAY

請以 JSON 格式回覆：
{"action": "STAY" or "REFRESH", "confidence": 0.0-1.0, "reason": "簡短說明"}"""


def build_user_prompt(sample: dict) -> str:
    """組裝給 GPT 的 user prompt"""
    turn_idx = sample.get("turn_idx", 0)
    prev_query = sample.get("prev_query") or "（無前文，第一輪）"
    current_query = sample.get("current_query", "")
    dst = sample.get("dst_features", {})

    return f"""請判斷以下對話的 Memory Action：

[輪次] turn_{turn_idx}
[上一輪 query] {prev_query}
[當前 query] {current_query}

[DST 信號]
- tv_distance: {dst.get('tv_distance', 0.5):.3f}（0=完全延續，1=完全切換）
- topic_overlap: {dst.get('topic_overlap', 0.5):.3f}（高=延續）
- domain_entropy: {dst.get('entropy', 0.5):.3f}（高=不明確）
- context_sim: {dst.get('context_sim', 0.5):.3f}（與上下文相似度）

請判斷該 STAY 還是 REFRESH，並以 JSON 格式回答。"""


def call_gpt(client: OpenAI, sample: dict, max_retry: int = 2) -> dict:
    """呼叫 GPT 並 parse 回覆"""
    user_prompt = build_user_prompt(sample)

    for attempt in range(max_retry + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1.0,
                max_completion_tokens=300,
            )
            text = resp.choices[0].message.content.strip()

            # 嘗試 parse JSON
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
                action = result.get("action", "").upper()
                if action in ("STAY", "REFRESH"):
                    return {
                        "action": action,
                        "confidence": float(result.get("confidence", 0.7)),
                        "reason": result.get("reason", ""),
                        "raw_response": text,
                    }
        except Exception as e:
            if attempt == max_retry:
                print(f"  ⚠️ GPT 標記失敗: {e}")
                break
            continue

    return None  # 標記失敗


# ============================================================
# SQL 資料載入
# ============================================================

def load_db_samples(max_samples: int = None) -> list:
    """從 SQL ChatMessage 載入 bot 回應紀錄，組成可標記的 samples"""
    from app import app, db, ChatMessage

    samples = []
    with app.app_context():
        bot_msgs = (
            ChatMessage.query
            .filter(ChatMessage.is_user_message == False)
            .filter(ChatMessage.flow_state.isnot(None))
            .order_by(ChatMessage.session_id, ChatMessage.sent_at)
            .all()
        )
        print(f"[Loader] 共讀取 {len(bot_msgs)} 筆 bot 訊息")

        # 按 session 分組
        sessions = defaultdict(list)
        for msg in bot_msgs:
            sid = msg.session_id or f"legacy_{msg.id}"
            sessions[sid].append(msg)

        # 同 child 的 user 訊息（按時間排序）
        all_users = (
            ChatMessage.query
            .filter(ChatMessage.is_user_message == True)
            .order_by(ChatMessage.child_id, ChatMessage.sent_at)
            .all()
        )
        users_by_child = defaultdict(list)
        for um in all_users:
            users_by_child[um.child_id or 0].append(um)

        # 組裝每筆 turn 的 sample
        for sid, msgs in sessions.items():
            for turn_idx, bot_msg in enumerate(msgs):
                try:
                    flow = json.loads(bot_msg.flow_state)
                except Exception:
                    continue

                # 找出對應的 current_query（bot_msg.sent_at 之前的最近 user 訊息）
                child_users = users_by_child.get(bot_msg.child_id or 0, [])
                current_query = ""
                prev_query = ""
                for i, um in enumerate(child_users):
                    if um.sent_at < bot_msg.sent_at:
                        current_query = um.message
                        if i > 0:
                            prev_query = child_users[i - 1].message

                if not current_query:
                    continue

                samples.append({
                    "session_id": sid,
                    "turn_idx": turn_idx,
                    "msg_id": bot_msg.id,
                    "prev_query": prev_query,
                    "current_query": current_query,
                    "dst_features": {
                        "entropy": float(flow.get("normalized_entropy", 0.5)),
                        "tv_distance": float(flow.get("tv_distance", 0.5)),
                        "topic_overlap": float(flow.get("topic_overlap", 0.5)),
                        "context_sim": float(flow.get("context_sim", 0.5)),
                        "is_multi_domain": bool(flow.get("is_multi_domain", False)),
                    },
                    "system_predicted_action": flow.get("memory_action", "?"),
                })

                if max_samples and len(samples) >= max_samples:
                    break
            if max_samples and len(samples) >= max_samples:
                break

    return samples


# ============================================================
# 主程式
# ============================================================

def relabel_samples(samples: list, output_path: str):
    """逐筆標記並寫入 JSONL"""
    if not OPENAI_API_KEY:
        print("⚠️  未設定 OPENAI_API_KEY，無法呼叫 GPT")
        return

    client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.openai.com/v1")
    print(f"[Relabel] 開始標記 {len(samples)} 筆樣本，輸出到 {output_path}")

    success_count = 0
    fail_count = 0
    agree_count = 0   # GPT 與系統判斷一致
    disagree_count = 0  # GPT 與系統不同（潛在學習機會）

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(samples):
            print(f"  [{i + 1}/{len(samples)}] turn_{sample['turn_idx']} | "
                  f"\"{sample['current_query'][:30]}...\"", end=" ")

            result = call_gpt(client, sample)
            if result is None:
                fail_count += 1
                print("❌ FAIL")
                continue

            success_count += 1
            sys_action = sample["system_predicted_action"]
            gpt_action = result["action"]

            if sys_action == gpt_action:
                agree_count += 1
                marker = "✓"
            else:
                disagree_count += 1
                marker = "⚠"

            print(f"{marker} sys={sys_action} gpt={gpt_action} ({result['confidence']:.2f})")

            # 寫入 JSONL
            record = {
                "session_id": sample["session_id"],
                "turn_index": sample["turn_idx"],
                "msg_id": sample["msg_id"],
                "user_query": sample["current_query"],
                "prev_query": sample["prev_query"],
                "memory_action": gpt_action,  # GPT 標記作為 ground truth
                "system_predicted_action": sys_action,
                "agreement": (sys_action == gpt_action),
                "gpt_confidence": result["confidence"],
                "gpt_reason": result["reason"],
                "retrieval_metadata": {
                    "tv_distance": sample["dst_features"]["tv_distance"],
                    "topic_overlap": sample["dst_features"]["topic_overlap"],
                    "context_sim": sample["dst_features"]["context_sim"],
                    "semantic_section_scores": {
                        "domain_entropy": sample["dst_features"]["entropy"],
                    },
                    "is_multi_domain": sample["dst_features"]["is_multi_domain"],
                },
                "_relabel_at": datetime.now().isoformat(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"  標記完成")
    print(f"{'=' * 60}")
    print(f"  總筆數:   {len(samples)}")
    print(f"  成功:     {success_count}")
    print(f"  失敗:     {fail_count}")
    if success_count > 0:
        print(f"  系統與 GPT 一致: {agree_count} ({agree_count/success_count*100:.1f}%)")
        print(f"  系統與 GPT 不同: {disagree_count} ({disagree_count/success_count*100:.1f}%)")
        print(f"  → 不一致樣本是學習機會（系統可能需要從 GPT 觀點修正）")
    print(f"  輸出檔案: {output_path}")


def merge_to_sft_dataset(relabel_path: str, sft_path: str = None):
    """[可選] 把標記結果合併到主 SFT 資料集"""
    if sft_path is None:
        sft_path = os.path.join(
            os.path.dirname(__file__),
            "../dataset/sft_dataset_v4_final.jsonl"
        )

    if not os.path.exists(relabel_path):
        print(f"⚠️  找不到標記檔: {relabel_path}")
        return

    with open(relabel_path, "r", encoding="utf-8") as f:
        new_records = [json.loads(line) for line in f if line.strip()]

    # 過濾低信心度或失敗的標記
    valid_records = [r for r in new_records if r.get("gpt_confidence", 0) >= 0.6]
    print(f"[Merge] 從 {len(new_records)} 筆過濾出 {len(valid_records)} 筆高信心樣本")

    # 寫入 SFT 副本（不覆蓋原檔）
    merged_path = sft_path.replace(".jsonl", "_with_relabel.jsonl")
    if os.path.exists(sft_path):
        with open(sft_path, "r", encoding="utf-8") as f:
            original = [json.loads(line) for line in f if line.strip()]
    else:
        original = []

    merged = original + valid_records
    with open(merged_path, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[Merge] 已合併 {len(original)} 原始 + {len(valid_records)} 標記 = {len(merged)} 筆")
    print(f"[Merge] 輸出: {merged_path}")
    print(f"[Merge] 後續步驟: 修改 pretrain_agents.py 的 SFT_DATASET_PATH 指向此檔，重新預訓練")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用 GPT 為 SQL 既有對話標記 Memory Action")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最多標記筆數（None=全部）")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"輸出 JSONL 路徑（預設: {DEFAULT_OUTPUT}）")
    parser.add_argument("--merge", action="store_true",
                        help="標記後合併到 SFT 資料集")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  GPT 標記 SQL 對話 → SFT 資料")
    print(f"{'=' * 60}")
    print(f"  Model:       {MODEL}")
    print(f"  Max samples: {args.max_samples or 'ALL'}")
    print(f"  Output:      {args.output}")
    print()

    samples = load_db_samples(max_samples=args.max_samples)
    if not samples:
        print("⚠️  無可標記的樣本")
        sys.exit(0)

    print(f"[Load] 共載入 {len(samples)} 筆可標記樣本")
    print(f"  預估成本: ~${len(samples) * 0.02:.2f} USD\n")

    confirm = input("是否繼續？(y/N): ").strip().lower()
    if confirm != "y":
        print("已取消")
        sys.exit(0)

    relabel_samples(samples, args.output)

    if args.merge:
        merge_to_sft_dataset(args.output)
