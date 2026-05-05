"""
Memory Agent 8d 新特徵抽取器（Shadow log；Phase 1 重訓時也用同一份）

不直接餵進現有 7d Memory Agent（會 dim mismatch），純記錄到 turn log。
Phase 1 ablation 重訓時 import 同一份，確保「線上抽特徵」與「離線重訓抽特徵」邏輯一致。

設計原則：
- Pure function（無 state），輸入 dict 出 dict
- 輸出可直接 JSON 序列化（float / bool / int / None）
- 鍵名穩定（一旦定下就不改名，論文 Table 用）
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

# ---------------- Keyword Sets ----------------
# 領域關鍵詞（命中 → 表示 query 明確指涉某 domain）
DOMAIN_KEYWORDS = (
    "粗大", "精細", "認知", "口語", "口腔", "情緒", "感統", "感覺統合",
    "吞嚥", "理解", "表達", "說話",
    "補助", "機構", "在地", "資源", "申請", "費用",
    "縣市", "台北", "台中", "高雄", "桃園", "新北", "台南", "新竹", "嘉義",
    "治療師", "復健", "鑑定", "幼兒園", "學校",
)

# Followup 關鍵詞（命中 → 強烈傾向 STAY，使用者在追問同一主題）
FOLLOWUP_KEYWORDS = (
    "那", "然後", "怎麼", "如何", "呢", "繼續", "所以", "再講", "詳細",
    "更", "為何", "為什麼", "意思", "代表", "例如", "比如", "舉例",
    "解釋", "說明", "再說", "可以嗎", "好嗎",
)

# Switch 關鍵詞（命中 → 傾向 REFRESH，使用者要切換主題）
SWITCH_KEYWORDS = (
    "別的", "其他", "換", "另外", "另一個", "其他方面", "別方面",
    "我想知道", "我想了解", "順便問", "再問一個", "另一題",
)

SHORT_QUERY_THRESHOLD = 12  # ≤12 字算短


# ---------------- Helpers ----------------

def _kw_hit(text: str, kw_set) -> bool:
    if not text:
        return False
    return any(kw in text for kw in kw_set)


def _kw_count(text: str, kw_set) -> int:
    if not text:
        return 0
    return sum(1 for kw in kw_set if kw in text)


# ---------------- Main extractor ----------------

def extract_memory_features_v2(
    *,
    user_query: str,
    domain_entropy: float,
    cur_top_domain: Optional[str],
    cur_top3_domains: List[str],
    prev_top_domain: Optional[str],
    cur_task_dist: Optional[Dict[str, float]],
    prev_task_dist: Optional[Dict[str, float]],
    prev_task: Optional[str],
    tv_distance_raw: float,
) -> Dict[str, float]:
    """
    抽 8d 新特徵 + 4d 屬性（共 12 鍵）。

    輸入：
      user_query: 本輪使用者 query 文字
      domain_entropy: 本輪 domain 分布的歸一化熵（[0,1]）
      cur_top_domain: 本輪原始 top_domain（pre-sticky；若可拿）
      cur_top3_domains: 本輪 domain 機率排序前 3 個（pre-sticky 較佳）
      prev_top_domain: 前一輪 top_domain（pre-sticky）
      cur_task_dist: 本輪 task softmax 分布
      prev_task_dist: 前一輪 task softmax 分布
      prev_task: 前一輪預測 task
      tv_distance_raw: pre-sticky 的 TV 距離（若無 raw 就傳調整後的）

    輸出：dict（純 JSON serializable）
    """
    q = user_query or ""
    q_len = len(q)
    q_len_norm = min(q_len / 30.0, 1.0)

    # --- Tier 1: Domain Continuity ---
    prev_top_eq_current_raw_top = bool(
        prev_top_domain and cur_top_domain and prev_top_domain == cur_top_domain
    )
    prev_top_in_current_top3 = bool(
        prev_top_domain and cur_top3_domains and prev_top_domain in cur_top3_domains
    )

    # --- Tier 3: Linguistic markers ---
    followup_kw_present = _kw_hit(q, FOLLOWUP_KEYWORDS)
    switch_kw_present = _kw_hit(q, SWITCH_KEYWORDS)
    domain_kw_present = _kw_hit(q, DOMAIN_KEYWORDS)

    # --- Tier 2: Composite ambiguity-followup score ---
    # entropy 高 + 沒明確 domain keyword + query 短 → 強訊號 STAY（隱含 followup）
    short_factor = 1.5 if q_len < SHORT_QUERY_THRESHOLD else 1.0
    no_domain_kw_factor = 0.0 if domain_kw_present else 1.0
    ambiguous_followup_score = float(domain_entropy) * no_domain_kw_factor * short_factor

    # --- Tier 4: Distribution change ---
    # task_top1_drop: 上一輪預測 task 的機率，在這一輪掉了多少（越大 → 主題轉得越遠）
    task_top1_drop = 0.0
    if prev_task and prev_task_dist and cur_task_dist:
        prev_p = float(prev_task_dist.get(prev_task, 0.0))
        cur_p = float(cur_task_dist.get(prev_task, 0.0))
        task_top1_drop = max(0.0, prev_p - cur_p)

    return {
        # --- 8d 主特徵（用於後續 MLP 訓練） ---
        "prev_top_eq_current_raw_top": int(prev_top_eq_current_raw_top),  # 0/1
        "prev_top_in_current_top3": int(prev_top_in_current_top3),
        "ambiguous_followup_score": float(ambiguous_followup_score),
        "followup_kw_present": int(followup_kw_present),
        "switch_kw_present": int(switch_kw_present),
        "tv_distance_raw": float(tv_distance_raw),
        "task_top1_drop": float(task_top1_drop),
        "query_len_norm": float(q_len_norm),
        # --- 4 個屬性（debug / 進階特徵 / 可解釋性） ---
        "domain_kw_present": int(domain_kw_present),
        "query_len_chars": int(q_len),
        "has_question_mark": int("?" in q or "？" in q),
        "domain_entropy_raw": float(domain_entropy),
    }


# ---------------- Self-test ----------------
if __name__ == "__main__":
    # 場景 1：明確新主題（粗大→補助）
    f1 = extract_memory_features_v2(
        user_query="目前小朋友的補助該怎麼申請",
        domain_entropy=0.4,
        cur_top_domain="補助",
        cur_top3_domains=["補助", "機構", "整體概況"],
        prev_top_domain="粗大動作",
        cur_task_dist={"H": 0.6, "K": 0.2},
        prev_task_dist={"C": 0.5, "D": 0.2, "H": 0.05},
        prev_task="C",
        tv_distance_raw=0.85,
    )
    print("[新主題] expect REFRESH:", f1)

    # 場景 2：模糊 followup（同 domain 但問訓練）
    f2 = extract_memory_features_v2(
        user_query="那要怎麼訓練",
        domain_entropy=0.7,
        cur_top_domain="整體概況",
        cur_top3_domains=["整體概況", "粗大動作", "認知功能"],
        prev_top_domain="粗大動作",
        cur_task_dist={"E": 0.4, "F": 0.2, "C": 0.1},
        prev_task_dist={"C": 0.5, "D": 0.2, "E": 0.1},
        prev_task="C",
        tv_distance_raw=0.4,
    )
    print("[模糊 followup] expect STAY:", f2)

    # 場景 3：明確同主題追問
    f3 = extract_memory_features_v2(
        user_query="單腳站立的表現怎麼樣？",
        domain_entropy=0.3,
        cur_top_domain="粗大動作",
        cur_top3_domains=["粗大動作", "整體概況", "感覺統合"],
        prev_top_domain="粗大動作",
        cur_task_dist={"C": 0.5, "D": 0.2},
        prev_task_dist={"C": 0.5, "D": 0.2},
        prev_task="C",
        tv_distance_raw=0.15,
    )
    print("[明確 STAY] expect STAY:", f3)
