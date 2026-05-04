"""
Held-out 測試集標註工具（互動式選單）

用法：
    python rl_pipeline/scripts/held_out_review.py \
        --input  raw_held_out.json \
        --output held_out_eval_v1.json

操作說明：
    每個 turn 會逐欄詢問，輸入「選項編號」後 Enter 即可。
    - 直接 Enter = 跳過該欄位（None / 不確定）
    - 輸入 b 退到上一個 turn
    - 輸入 s 跳過整個 scenario
    - 輸入 q 暫停（已標註的會自動存檔，下次跑會繼續）

輸入格式（raw_held_out.json）：
    [
      {
        "name": "...",
        "expected_task": "C",            # LLM 自己給的猜測（你可改）
        "expected_secondary": ["D"],     # LLM 自己給的猜測（你可改）
        "steps": ["問題1", "問題2", ...]
      },
      ...
    ]

輸出格式（held_out_eval_v1.json）：
    每個 scenario 額外加一個 "annotations" 陣列，逐 turn 標註：
    [
      {
        "name": "...",
        "steps": [...],
        "annotations": [
          {
            "turn_idx": 0,
            "expected_task": "C",
            "expected_secondary": [],
            "expected_domains": ["粗大動作"],
            "expected_memory": "REFRESH",
            "expected_clarify": null,
            "expected_sections": ["assessment", "observation"],
            "notes": ""
          },
          ...
        ]
      }
    ]
"""

import argparse
import json
import os
import sys
from typing import Optional, List, Dict, Any

# ── 系統定義（與 dialogue_state_module 對齊） ──────────────────────────────

TASK_LABELS = [
    ("A", "報告總覽與閱讀順序"),
    ("B", "分數/量表/百分位解讀"),
    ("C", "臨床觀察與表現解讀"),
    ("D", "能力剖面（優勢/需求/優先順序）"),
    ("E", "在家訓練怎麼做"),
    ("F", "融入日常作息的練習"),
    ("G", "是否需要早療/成效追蹤"),
    ("H", "轉介與在地資源"),
    ("I", "報告分享/隱私與安全"),
    ("J", "與學校合作"),
    ("K", "補助/福利/申請"),
    ("L", "後續追蹤/再評估"),
    ("M", "家長情緒支持與家庭協作"),
    ("N", "進步查詢"),
    ("OOD", "離題（非早療相關）"),
]

DOMAIN_LIST = [
    "整體概況",
    "粗大動作",
    "精細動作",
    "感覺統合",
    "口腔動作",
    "情緒行為與社會適應功能",
    "吞嚥功能",
    "口語理解",
    "口語表達",
    "說話",
    "認知功能",
]

MEMORY_OPTIONS = [
    ("STAY", "延續上一輪（同主題深入）"),
    ("REFRESH", "切換主題（換 task 或換 domain）"),
]

CLARIFY_OPTIONS = [
    (None, "不需追問"),
    ("DOMAIN_HARD", "極模糊，無法判斷主題"),
    ("CONTEXT_MISSING", "T0 接續語但無歷史"),
    ("SLOT_REGION", "缺地區（H/K 任務）"),
    ("TASK_SOFT", "多任務並存"),
    ("OUT_OF_DOMAIN", "離題"),
]

SECTION_OPTIONS = [
    "assessment",      # 個案：評估分數
    "observation",     # 個案：臨床觀察
    "training",        # 個案：訓練方向
    "suggestion",      # 個案：建議
    "community_resources",  # SQL：在地資源
    "external_gpt",    # 通用早療知識
]

# ── UI 工具 ──────────────────────────────────────────────────────────────

def _print_header(text: str, char: str = "="):
    print()
    print(char * 60)
    print(text)
    print(char * 60)


def _ask_single(prompt: str, options: List, allow_skip: bool = True,
                allow_back: bool = True, allow_quit: bool = True) -> Any:
    """
    單選：列出選項，使用者輸入編號
    options: list of (value, label) tuples
    回傳: value 或 "__BACK__" / "__SKIP__" / "__QUIT__" / "__NONE__"
    """
    while True:
        print(f"\n{prompt}")
        for i, (val, label) in enumerate(options, 1):
            val_str = "(None)" if val is None else str(val)
            print(f"  [{i}] {val_str:14} {label}")
        flags = []
        if allow_skip: flags.append("Enter=跳過")
        if allow_back: flags.append("b=上一步")
        if allow_quit: flags.append("q=暫停存檔")
        print(f"  ({'  '.join(flags)})")
        ans = input("  > ").strip().lower()

        if ans == "" and allow_skip:
            return "__NONE__"
        if ans == "b" and allow_back:
            return "__BACK__"
        if ans == "q" and allow_quit:
            return "__QUIT__"
        if ans == "s":
            return "__SKIP__"

        if ans.isdigit():
            idx = int(ans)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        print(f"  [錯誤] 請輸入 1-{len(options)} 之間的數字")


def _ask_multi(prompt: str, options: List, allow_skip: bool = True) -> List:
    """
    多選：使用者輸入逗號分隔的編號（如 1,3,5）
    options: list of (value, label) tuples
    回傳: list of values 或 "__BACK__" / "__SKIP__" / "__QUIT__"
    """
    while True:
        print(f"\n{prompt}")
        for i, item in enumerate(options, 1):
            if isinstance(item, tuple):
                val, label = item
                print(f"  [{i:2}] {str(val):14} {label}")
            else:
                print(f"  [{i:2}] {item}")
        print(f"  (多選用逗號分隔，例如 1,3,5  Enter=跳過  b=上一步  q=暫停)")
        ans = input("  > ").strip().lower()

        if ans == "" and allow_skip:
            return []
        if ans == "b":
            return "__BACK__"
        if ans == "q":
            return "__QUIT__"
        if ans == "s":
            return "__SKIP__"

        try:
            indices = [int(x.strip()) for x in ans.split(",") if x.strip()]
            picked = []
            for idx in indices:
                if 1 <= idx <= len(options):
                    item = options[idx - 1]
                    val = item[0] if isinstance(item, tuple) else item
                    picked.append(val)
                else:
                    raise ValueError(f"超出範圍：{idx}")
            return picked
        except ValueError as e:
            print(f"  [錯誤] {e}")


# ── 主流程 ───────────────────────────────────────────────────────────────

def annotate_turn(scenario: Dict, turn_idx: int, prev_annotation: Optional[Dict] = None) -> Optional[Dict]:
    """
    標註單一 turn，回傳 annotation dict 或 None（表 BACK/QUIT）
    """
    name = scenario.get("name", "?")
    steps = scenario["steps"]
    user_query = steps[turn_idx]
    suggested_task = scenario.get("expected_task", "")
    suggested_secondary = scenario.get("expected_secondary", [])

    _print_header(f"[{name}]  Turn {turn_idx + 1} / {len(steps)}", "─")
    # 顯示前後 context
    for i, q in enumerate(steps):
        marker = ">>>" if i == turn_idx else "   "
        print(f"  {marker} T{i}: {q}")
    if suggested_task:
        sec_str = f" + {suggested_secondary}" if suggested_secondary else ""
        print(f"\n  💡 LLM 提示：主任務={suggested_task}{sec_str}（僅供參考，你決定）")

    annotation = {"turn_idx": turn_idx}

    # 1. expected_task
    ans = _ask_single(
        "❶ 預期主要 task（這個 turn 該被分到哪個 task）",
        TASK_LABELS,
    )
    if ans == "__BACK__":
        return "__BACK__"
    if ans == "__QUIT__":
        return "__QUIT__"
    if ans == "__SKIP__":
        return "__SKIP__"
    annotation["expected_task"] = None if ans == "__NONE__" else ans

    # 2. expected_secondary（多選）
    ans = _ask_multi(
        "❷ 預期 secondary task（混合任務時的次要 task，可多選；單一任務直接 Enter）",
        TASK_LABELS,
    )
    if ans == "__BACK__":
        return "__BACK__"
    if ans == "__QUIT__":
        return "__QUIT__"
    annotation["expected_secondary"] = ans if isinstance(ans, list) else []

    # 3. expected_domains（多選）
    ans = _ask_multi(
        "❸ 預期領域 domain（這個 turn 涉及哪個能力面向，可多選）",
        DOMAIN_LIST,
    )
    if ans == "__BACK__":
        return "__BACK__"
    if ans == "__QUIT__":
        return "__QUIT__"
    annotation["expected_domains"] = ans if isinstance(ans, list) else []

    # 4. expected_memory
    if turn_idx == 0:
        # T0 一定 REFRESH，自動填
        annotation["expected_memory"] = "REFRESH"
        print(f"\n❹ Memory 動作：[自動] T0 → REFRESH")
    else:
        ans = _ask_single(
            "❹ 預期 Memory 動作（相對於上一輪）",
            MEMORY_OPTIONS,
        )
        if ans == "__BACK__":
            return "__BACK__"
        if ans == "__QUIT__":
            return "__QUIT__"
        annotation["expected_memory"] = None if ans == "__NONE__" else ans

    # 5. expected_clarify
    ans = _ask_single(
        "❺ 預期 Clarify 類型（系統是否該追問）",
        CLARIFY_OPTIONS,
    )
    if ans == "__BACK__":
        return "__BACK__"
    if ans == "__QUIT__":
        return "__QUIT__"
    annotation["expected_clarify"] = None if ans == "__NONE__" else ans

    # 6. expected_sections（多選）
    sec_options_with_labels = [
        ("assessment", "個案-評估分數"),
        ("observation", "個案-臨床觀察"),
        ("training", "個案-訓練方向"),
        ("suggestion", "個案-建議"),
        ("community_resources", "SQL-在地資源/補助"),
        ("external_gpt", "通用早療知識"),
    ]
    ans = _ask_multi(
        "❻ 預期應勾選的 sections（系統該檢索哪些資料源，可多選）",
        sec_options_with_labels,
    )
    if ans == "__BACK__":
        return "__BACK__"
    if ans == "__QUIT__":
        return "__QUIT__"
    annotation["expected_sections"] = ans if isinstance(ans, list) else []

    # 7. notes（自由輸入）
    notes = input("\n❼ 備註（選填，Enter 跳過）：").strip()
    annotation["notes"] = notes

    return annotation


DEFAULT_INPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "held_out",
    "raw_held_out_combined.json",
)
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "held_out",
    "held_out_eval_v1.json",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help=f"LLM 生成的原始 scenarios JSON（預設：{DEFAULT_INPUT}）")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"標註後輸出 JSON，也用於斷點續做（預設：{DEFAULT_OUTPUT}）")
    args = parser.parse_args()

    # 載入原始輸入
    with open(args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 載入既有進度（若有）
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            current = json.load(f)
        print(f"[續做] 從 {args.output} 載入既有進度（{len(current)} 個 scenario）")
    else:
        current = []

    # 對齊：raw 沒標的接著 current 後面
    annotated_names = {sc["name"] for sc in current}
    todo = [sc for sc in raw if sc["name"] not in annotated_names]

    print(f"\n[總覽] 共 {len(raw)} 個 scenario，已標 {len(current)}，待標 {len(todo)}")
    if not todo:
        print("[完成] 全部已標註！")
        return

    # 開始標註
    quit_flag = False
    for sc_idx, scenario in enumerate(todo):
        if quit_flag:
            break

        sc_copy = dict(scenario)
        sc_copy["annotations"] = []
        steps = scenario["steps"]

        _print_header(f"[Scenario {sc_idx + 1} / {len(todo)}]  {scenario.get('name', '?')}")
        print(f"共 {len(steps)} turn")
        print(f"name: {scenario.get('name', '?')}")
        print(f"LLM 提示主 task: {scenario.get('expected_task', '?')}")
        print(f"LLM 提示副 task: {scenario.get('expected_secondary', [])}")

        turn_idx = 0
        while turn_idx < len(steps):
            ann = annotate_turn(scenario, turn_idx, sc_copy["annotations"][-1] if sc_copy["annotations"] else None)

            if ann == "__BACK__":
                if turn_idx > 0:
                    sc_copy["annotations"].pop()
                    turn_idx -= 1
                else:
                    print("  [已是第一個 turn，無法後退]")
            elif ann == "__SKIP__":
                print(f"  [跳過 scenario {scenario.get('name', '?')}]")
                sc_copy = None
                break
            elif ann == "__QUIT__":
                quit_flag = True
                break
            else:
                sc_copy["annotations"].append(ann)
                turn_idx += 1

        if sc_copy and not quit_flag:
            current.append(sc_copy)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(current, f, ensure_ascii=False, indent=2)
            print(f"\n  ✓ Scenario {sc_idx + 1} 已存檔")
        elif quit_flag:
            # 暫存進度
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(current, f, ensure_ascii=False, indent=2)
            print(f"\n  💾 進度已存到 {args.output}")
            print(f"  下次跑同樣指令會自動續做")

    if not quit_flag:
        print(f"\n[完成] 全部 {len(current)} 個 scenario 標註完畢！")
        print(f"輸出：{args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Ctrl+C] 已中斷，下次跑同樣指令會自動續做")
        sys.exit(0)
