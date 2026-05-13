"""
Apply 4 patches to domain_anchors.json based on observed errors in 90-turn evaluation.

Backups original file to domain_anchors.json.bak before writing.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "dialogue_state_module" / "config" / "domain_anchors.json"
BAK = CFG.with_suffix(".json.bak")

# Patch 1: 粗大動作 — 加抽象 / 生活化層
GROSS_MOTOR_ADD = [
    "動作能力對日常生活的影響、動作問題如何影響上學與學習",
    "動作上比較吃力、動作跟不上同學、動作慢被同學笑",
    "大肌肉活動、體能課、運動課、體育課的參與",
    "動作慢、動作笨拙、動作不協調、動作不流暢",
    "動作表現會在生活中怎麼出現、動作問題的生活樣貌",
    "孩子動作慢跟不上同學怎麼辦？動作問題在教室裡的表現",
    "粗大動作對學校生活的影響、體能課需要的協助",
]

# Patch 2: 情緒行為與社會適應功能 — 加抗拒 / 動機層
EMOTION_ADD = [
    "孩子說不要做、孩子說累、孩子拒絕配合練習",
    "孩子做一下就不要、孩子練不下去、孩子抗拒練習活動",
    "孩子覺得無聊、孩子失去動機、孩子缺乏興趣",
    "孩子害怕失敗、不敢嘗試、退縮、逃避困難",
    "孩子做一下就喊累、孩子不耐煩、孩子缺乏耐心",
    "孩子鬧脾氣不想做練習、家長該怎麼調整",
]

# Patch 3: 整體概況 — 加跨專業 / 整合層
OVERVIEW_ADD = [
    "跨專業整合、各領域整理重點、不同治療師建議的關聯",
    "報告整理成幾個重點、學校相關的整體重點",
    "整體建議的優先順序、家裡時間有限該先做哪一類",
    "報告裡每個專業都有建議、各專業建議的整合與取捨",
    "整理這份聯評報告中和學校生活最有關的重點",
    "下次回診或療育時可以帶哪些問題、整體回診準備",
    "這份報告整體可以怎麼用、報告的整體應用方向",
    "報告主要在說什麼、整體要怎麼看",
]

# Patch 4: 口語表達 — 把過於泛化的 anchor 改具體
SPEECH_REPLACE = [
    (
        "孩子不會表達需求、不知道怎麼說，怎麼辦？",
        "孩子用單字代替句子、不會用完整句子說明事情",
    ),
    (
        "孩子表達能力不好、詞彙量少，需要做語言治療嗎？",
        "孩子詞彙量很少、講話只會用幾個詞彙，是表達能力不足嗎？",
    ),
]


def patch():
    data = json.load(open(CFG, encoding="utf-8"))

    # Backup
    if not BAK.exists():
        shutil.copy(CFG, BAK)
        print(f"Backup: {BAK}")
    else:
        print(f"(Backup already exists at {BAK})")

    # Apply patches
    anchors = data["domain_anchors"]

    def append_unique(domain_key, additions):
        existing = anchors[domain_key]
        added = []
        for a in additions:
            if a not in existing:
                existing.append(a)
                added.append(a)
        print(f"  {domain_key}: +{len(added)}/{len(additions)} (total {len(existing)})")
        return added

    print("\n=== Append anchors ===")
    append_unique("粗大動作", GROSS_MOTOR_ADD)
    append_unique("情緒行為與社會適應功能", EMOTION_ADD)
    overview_added = append_unique("整體概況", OVERVIEW_ADD)
    # 同步更新 overview_anchors (top-level field, 與 domain_anchors["整體概況"] 一致)
    for a in overview_added:
        if a not in data["overview_anchors"]:
            data["overview_anchors"].append(a)

    print("\n=== Replace anchors (口語表達) ===")
    speech_list = anchors["口語表達"]
    for old, new in SPEECH_REPLACE:
        if old in speech_list:
            idx = speech_list.index(old)
            speech_list[idx] = new
            print(f"  replaced: {old[:30]}...  ->  {new[:30]}...")
        else:
            print(f"  (skip, not found): {old[:30]}...")

    # Write back
    with open(CFG, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {CFG}")
    print(f"\n=== New anchor counts ===")
    for d in data["domains"]:
        print(f"  {d:<28} {len(anchors[d])}")


if __name__ == "__main__":
    patch()
