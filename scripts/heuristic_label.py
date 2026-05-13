"""
用關鍵字啟發式為每 turn 預判「期望 domain」，
再與系統預測對照，輸出疑似錯誤清單。

這是粗略的 first pass — 人工 review 為準。
"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = Path(__file__).resolve().parent / "raw_logs"

# 關鍵字 → domain 映射 (多對多)。匹配到任一即列入候選。
DOMAIN_KEYWORDS = {
    "粗大動作": [
        "動作表現", "動作能力", "動作練習", "動作上", "動作發展",
        "走路", "跑", "跳", "平衡", "核心", "穩定", "肌力",
        "上下樓梯", "粗大", "體能課", "大肌肉", "全身",
    ],
    "精細動作": [
        "精細", "握筆", "寫字", "剪刀", "拼圖", "扣鈕扣",
        "手部", "手指", "雙手協調", "手眼協調", "用湯匙", "用筷子",
    ],
    "感覺統合": [
        "感覺統合", "感統", "前庭", "本體", "觸覺", "感覺尋求",
        "怕高", "怕聲音", "對聲音敏感", "坐不住一直動", "感覺刺激",
    ],
    "口腔動作": [
        "口腔動作", "口肌", "嘴唇", "舌頭", "咀嚼", "吹氣", "流口水",
    ],
    "情緒行為與社會適應功能": [
        "情緒", "哭鬧", "生氣", "崩潰", "脾氣", "焦慮", "擔心",
        "嚴重", "自責", "壓力", "情感", "搶玩具", "社交",
        "不喜歡", "不要做", "說累", "說不", "拒絕", "抗拒",
    ],
    "吞嚥功能": [
        "吞嚥", "嗆", "嗆到", "進食安全", "吃飯慢", "含著不吞",
    ],
    "口語理解": [
        "聽得懂", "聽不懂", "理解指令", "兩步驟", "理解我說",
        "聽指令", "理解語意",
    ],
    "口語表達": [
        "說話", "表達", "詞彙", "句子", "用講的", "說出來",
        "講不清楚", "回答問題",
    ],
    "說話": [
        "發音", "構音", "音清楚", "說話不清楚", "可懂度", "口齒",
    ],
    "認知功能": [
        "注意力", "專注", "記憶", "學習能力", "認知",
        "規則", "步驟", "因果", "概念", "配對", "分類",
    ],
    # 「整體概況」：meta-level, 不指特定能力
    "整體概況": [
        "整份報告", "整體", "報告主要", "全部一起", "整體狀況",
        "報告在說什麼", "整理成", "整理重點", "整體評估",
    ],
}

# 角色 / 場景關鍵字（次要 signal）
SCHOOL_KEYWORDS = ["老師", "教室", "幼兒園", "上學", "學校", "IEP", "特教"]
HOME_KEYWORDS = ["在家", "家裡", "日常", "睡前", "放學後", "上學前"]
RESOURCE_KEYWORDS = ["補助", "申請", "資源", "機構", "治療所", "看哪一科"]
SAFETY_KEYWORDS = ["嗆", "嚴重", "擔心", "危險", "醫師", "急診"]


def label_query(query: str):
    """回傳期望 domains (set) + 提示。"""
    hits = {}
    for d, kws in DOMAIN_KEYWORDS.items():
        matched = [kw for kw in kws if kw in query]
        if matched:
            hits[d] = matched
    return hits


def main():
    rows = []
    for fp in sorted(RAW_DIR.glob("CASE_*_G*.json")):
        d = json.load(open(fp, encoding="utf-8"))
        for t in d["turns"]:
            fs = t.get("flow_state") or {}
            rows.append({
                "case_id": d["case_id"],
                "group": d["group_key"],
                "turn": t["turn_no"],
                "query": t["query"],
                "predicted": fs.get("top_domain"),
                "active": fs.get("active_domains") or [],
                "prob": fs.get("top_domain_prob"),
            })

    suspicious = []
    confirmed = []
    no_kw_match = []
    for r in rows:
        hits = label_query(r["query"])
        expected = set(hits.keys())
        pred = r["predicted"]
        if not expected:
            no_kw_match.append(r)
            continue
        # 系統預測在期望集合內 → 確認 OK
        if pred in expected:
            confirmed.append((r, hits))
        # 系統預測不在期望集合 → 可疑
        else:
            suspicious.append((r, hits))

    print(f"=== 啟發式分析 (90 turn) ===")
    print(f"  ✓ 預測在關鍵字期望內: {len(confirmed)}")
    print(f"  ✗ 預測不在關鍵字期望內: {len(suspicious)} <- 重點檢視")
    print(f"  ? 無關鍵字匹配 (meta 或太抽象): {len(no_kw_match)}")
    print()
    print(f"=== 疑似 domain 錯誤 ({len(suspicious)} 筆) ===")
    for r, hits in suspicious:
        kw_str = ", ".join(f"{d}<{','.join(kws)}>" for d, kws in hits.items())
        print(f"\n[{r['case_id']}/{r['group']} T{r['turn']}] prob={r['prob']:.2f}")
        print(f"  Q: {r['query'][:80]}")
        print(f"  預測: {r['predicted']}")
        print(f"  期望(關鍵字): {kw_str}")

    print(f"\n=== 無關鍵字匹配 ({len(no_kw_match)} 筆) ===")
    for r in no_kw_match:
        print(f"  [{r['case_id']}/{r['group']} T{r['turn']}] {r['predicted']} | {r['query'][:60]}")

    # 統計疑似錯誤的「期望 domain」分布（缺失最嚴重的）
    missing = Counter()
    for r, hits in suspicious:
        for d in hits.keys():
            missing[d] += 1
    print(f"\n=== 疑似錯誤中, 被遺漏的 domain top10 ===")
    for d, n in missing.most_common():
        print(f"  {d:<28} 被忽略 {n} 次")


if __name__ == "__main__":
    main()
