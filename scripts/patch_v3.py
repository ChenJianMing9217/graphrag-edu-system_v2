"""
Patch v3 — 解決 v2 的「說話」連鎖反應 bug。

Patch 5: 限縮「說話」anchor (移除廣義 meta 詞)
Patch 6: 「整體概況」補「發展面向 / 需要被注意」類 meta anchor
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "dialogue_state_module" / "config" / "domain_anchors.json"
BAK_V2 = CFG.with_suffix(".json.v2.bak")

# Patch 5: 「說話」要替換的廣義句
SPEECH_REPLACE = [
    (
        "發音問題、構音問題、語音問題、說話困難、說話遲緩",
        "發音不正確、構音錯誤、語音清晰度低、口齒不清",
    ),
    (
        "說話能力評估結果、說話能力訓練建議、說話能力治療計畫、說話問題、說話改善目標",
        "構音治療計畫、構音治療目標、構音評估結果、發音矯正",
    ),
]

# Patch 6: 「整體概況」補 meta query
OVERVIEW_ADD = [
    "發展面向的整體說明、孩子目前需要被注意的能力面向",
    "孩子整體應該注意哪些面向、整體要重點關注什麼",
    "報告中需要重點關注的能力領域、整體需要關注的重點",
    "目前最需要注意的幾個發展面向、需要被注意的重點",
]


def patch():
    data = json.load(open(CFG, encoding="utf-8"))

    if not BAK_V2.exists():
        shutil.copy(CFG, BAK_V2)
        print(f"Backup v2: {BAK_V2}")
    else:
        print(f"(v2 backup exists: {BAK_V2})")

    anchors = data["domain_anchors"]

    print("\n=== Patch 5: 限縮「說話」anchor ===")
    speech = anchors["說話"]
    for old, new in SPEECH_REPLACE:
        if old in speech:
            idx = speech.index(old)
            speech[idx] = new
            print(f"  ✓ replaced: {old[:30]}... -> {new[:30]}...")
        else:
            print(f"  ✗ skip (not found): {old[:30]}...")

    print("\n=== Patch 6: 補「整體概況」meta anchor ===")
    overview = anchors["整體概況"]
    added = []
    for a in OVERVIEW_ADD:
        if a not in overview:
            overview.append(a)
            added.append(a)
            print(f"  ✓ add: {a[:40]}...")
    # 同步 top-level overview_anchors
    for a in added:
        if a not in data["overview_anchors"]:
            data["overview_anchors"].append(a)

    with open(CFG, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {CFG}")
    print(f"\n=== Anchor 總數 ===")
    for d in data["domains"]:
        print(f"  {d:<24} {len(anchors[d])}")


if __name__ == "__main__":
    patch()
