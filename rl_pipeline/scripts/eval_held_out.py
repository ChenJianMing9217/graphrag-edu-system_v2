"""
Held-out 測試集評估腳本

讀 held_out_eval_v1.json（含 annotations），跑 API 拿 flow_state，
與你親自標註的 ground truth 對比，輸出 6 軸指標 + markdown 報告。

跟 test_e2e_scenarios.py 的差別：
- test_e2e_scenarios.py：用 category 級別 expected_task（粗粒度）
- eval_held_out.py：用 per-turn annotation（細粒度，且非訓練分布）

用法：
    python rl_pipeline/scripts/eval_held_out.py
    python rl_pipeline/scripts/eval_held_out.py --code TESTCODE --report report.md

指標：
    1. Task 嚴格命中率 / 相容命中率
    2. Domain 命中率（active_domains ∩ expected_domains 是否非空）
    3. Memory 動作正確率（STAY/REFRESH）
    4. Clarify 觸發 precision / recall（per type）
    5. Sections 命中率（planning_active ∩ expected_sections 是否非空）
    6. OOD 偵測 precision / recall
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import requests

# 加 sys.path 才能 import 現有工具
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_e2e_scenarios import (  # noqa
    TASK_COMPATIBILITY, OOD_CATEGORY,
    login, new_chat, send_message, extract_flow,
    BASE_URL, ACCESS_CODE,
)

DEFAULT_INPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "held_out", "held_out_eval_v1.json",
)
DEFAULT_REPORT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "held_out", f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
)
TURN_DELAY = 0.6

# ── 評估工具函數 ─────────────────────────────────────────────────────────

def task_strict_hit(predicted_tasks: List[str], expected_task: Optional[str]) -> bool:
    """嚴格：預測的 task（含 secondary）是否包含 expected_task"""
    if expected_task is None:
        return True  # 未標註不評估
    return expected_task in predicted_tasks


def task_compat_hit(predicted_tasks: List[str], expected_task: Optional[str]) -> bool:
    """相容：預測 task 是否落在 expected 的相容群內"""
    if expected_task is None:
        return True
    if expected_task == "OOD":
        return "OOD" in predicted_tasks  # OOD 必須完全命中
    compat = TASK_COMPATIBILITY.get(expected_task, {expected_task})
    return any(t in compat for t in predicted_tasks)


def domain_hit(active_domains: List[str], expected_domains: List[str]) -> Optional[bool]:
    """domain 命中：交集非空算對；未標註返回 None"""
    if not expected_domains:
        return None
    return bool(set(active_domains) & set(expected_domains))


def memory_correct(predicted: Optional[str], expected: Optional[str]) -> Optional[bool]:
    """memory 動作對比"""
    if expected is None:
        return None
    return predicted == expected


def clarify_match(predicted: Optional[str], expected: Optional[str]) -> Tuple[bool, str]:
    """
    clarify 對比，回傳 (是否完全匹配, 細分類型)
    細分類型：tp (預期=預測=非None), tn (兩者皆None), fp (預期None預測有), fn (預期有預測None), wrong_type (兩者皆有但不同)
    """
    if expected is None and predicted is None:
        return True, "tn"
    if expected is None and predicted is not None:
        return False, "fp"
    if expected is not None and predicted is None:
        return False, "fn"
    if expected == predicted:
        return True, "tp"
    return False, "wrong_type"


def sections_hit(planning_active: List[str], expected_sections: List[str]) -> Optional[bool]:
    """sections 命中：planning agent 勾的 sections 跟 expected 有交集"""
    if not expected_sections:
        return None
    return bool(set(planning_active) & set(expected_sections))


# ── 主流程 ─────────────────────────────────────────────────────────────

def run_eval(scenarios: List[Dict], delay: float = TURN_DELAY) -> Dict:
    """跑 API、對比 annotation，回傳統計"""
    stats = {
        "n_scenarios": 0,
        "n_turns": 0,
        "task_strict": 0,
        "task_compat": 0,
        "task_strict_total": 0,
        "task_compat_total": 0,
        "domain_hit": 0,
        "domain_total": 0,
        "memory_correct": 0,
        "memory_total": 0,
        "clarify_buckets": Counter(),  # tn/tp/fp/fn/wrong_type
        "clarify_per_type": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
        "sections_hit": 0,
        "sections_total": 0,
        "ood_tp": 0, "ood_fp": 0, "ood_fn": 0, "ood_tn": 0,
        "scenario_details": [],
    }

    for sc_idx, sc in enumerate(scenarios):
        new_chat()
        time.sleep(delay)

        sc_detail = {
            "name": sc.get("name", f"sc_{sc_idx}"),
            "turns": [],
        }

        annotations = sc.get("annotations", [])
        if not annotations:
            print(f"  [SKIP] {sc['name']}：無 annotations")
            continue

        stats["n_scenarios"] += 1

        for turn_idx, step in enumerate(sc["steps"]):
            ann = annotations[turn_idx] if turn_idx < len(annotations) else None
            if ann is None:
                continue

            # 跑 API
            resp = send_message(step)
            flow = extract_flow(resp)
            time.sleep(delay)

            stats["n_turns"] += 1

            predicted_tasks = [flow["task"]] + (flow.get("secondary") or [])

            # 1. Task strict / compat
            exp_task = ann.get("expected_task")
            if exp_task is not None:
                stats["task_strict_total"] += 1
                stats["task_compat_total"] += 1
                if task_strict_hit(predicted_tasks, exp_task):
                    stats["task_strict"] += 1
                if task_compat_hit(predicted_tasks, exp_task):
                    stats["task_compat"] += 1

            # 2. Domain
            d_hit = domain_hit(flow["active_domains"], ann.get("expected_domains") or [])
            if d_hit is not None:
                stats["domain_total"] += 1
                if d_hit:
                    stats["domain_hit"] += 1

            # 3. Memory
            m_cor = memory_correct(flow["memory_action"], ann.get("expected_memory"))
            if m_cor is not None:
                stats["memory_total"] += 1
                if m_cor:
                    stats["memory_correct"] += 1

            # 4. Clarify
            exp_cl = ann.get("expected_clarify")
            pred_cl = flow.get("clarify_type")
            ok, bucket = clarify_match(pred_cl, exp_cl)
            stats["clarify_buckets"][bucket] += 1
            if exp_cl:
                if pred_cl == exp_cl:
                    stats["clarify_per_type"][exp_cl]["tp"] += 1
                else:
                    stats["clarify_per_type"][exp_cl]["fn"] += 1
            if pred_cl and pred_cl != exp_cl:
                stats["clarify_per_type"][pred_cl]["fp"] += 1

            # 5. Sections
            s_hit = sections_hit(flow.get("planning_active", []), ann.get("expected_sections") or [])
            if s_hit is not None:
                stats["sections_total"] += 1
                if s_hit:
                    stats["sections_hit"] += 1

            # 6. OOD
            if exp_task == "OOD":
                if "OOD" in predicted_tasks or pred_cl == "OUT_OF_DOMAIN":
                    stats["ood_tp"] += 1
                else:
                    stats["ood_fn"] += 1
            else:
                if "OOD" in predicted_tasks or pred_cl == "OUT_OF_DOMAIN":
                    stats["ood_fp"] += 1
                else:
                    stats["ood_tn"] += 1

            sc_detail["turns"].append({
                "turn_idx": turn_idx,
                "query": step,
                "predicted": {
                    "task": flow["task"],
                    "secondary": flow["secondary"],
                    "memory": flow["memory_action"],
                    "clarify": pred_cl,
                    "domains": flow["active_domains"],
                    "sections": flow.get("planning_active", []),
                },
                "expected": {
                    "task": exp_task,
                    "secondary": ann.get("expected_secondary", []),
                    "memory": ann.get("expected_memory"),
                    "clarify": exp_cl,
                    "domains": ann.get("expected_domains", []),
                    "sections": ann.get("expected_sections", []),
                },
                "task_strict_hit": task_strict_hit(predicted_tasks, exp_task) if exp_task else None,
                "task_compat_hit": task_compat_hit(predicted_tasks, exp_task) if exp_task else None,
            })

        stats["scenario_details"].append(sc_detail)
        print(f"  [{sc_idx + 1}/{len(scenarios)}] {sc['name']}: {len(sc_detail['turns'])} turns")

    return stats


def safe_div(a, b):
    return a / b if b > 0 else 0.0


def to_markdown_report(stats: Dict, total_scenarios: int) -> str:
    """轉成 markdown 報告（論文可直接 copy）"""
    lines = []
    lines.append("# Held-out Evaluation Report\n")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")
    lines.append(f"_Scenarios evaluated: {stats['n_scenarios']} / {total_scenarios}_  ")
    lines.append(f"_Total turns: {stats['n_turns']}_\n")

    lines.append("## 1. Task Classification\n")
    lines.append("| Metric | Hits / Total | Rate |")
    lines.append("|---|---|---|")
    lines.append(f"| Task strict hit (預測必須含 expected) | {stats['task_strict']}/{stats['task_strict_total']} | {safe_div(stats['task_strict'], stats['task_strict_total']) * 100:.1f}% |")
    lines.append(f"| Task compat hit (落在相容群內) | {stats['task_compat']}/{stats['task_compat_total']} | {safe_div(stats['task_compat'], stats['task_compat_total']) * 100:.1f}% |\n")

    lines.append("## 2. Domain Classification\n")
    lines.append(f"- Domain hit rate: **{stats['domain_hit']}/{stats['domain_total']} = {safe_div(stats['domain_hit'], stats['domain_total']) * 100:.1f}%**")
    lines.append("  > active_domains ∩ expected_domains 非空算命中\n")

    lines.append("## 3. Memory Action (STAY / REFRESH)\n")
    lines.append(f"- Memory accuracy: **{stats['memory_correct']}/{stats['memory_total']} = {safe_div(stats['memory_correct'], stats['memory_total']) * 100:.1f}%**\n")

    lines.append("## 4. Clarify Detection\n")
    lines.append("### 4.1 Confusion buckets\n")
    cb = stats["clarify_buckets"]
    total_cl = sum(cb.values())
    lines.append("| Bucket | Count | % |")
    lines.append("|---|---|---|")
    for k in ["tn", "tp", "fp", "fn", "wrong_type"]:
        c = cb.get(k, 0)
        lines.append(f"| {k} | {c} | {safe_div(c, total_cl) * 100:.1f}% |")
    lines.append("")
    lines.append("### 4.2 Per-type precision / recall\n")
    lines.append("| Type | TP | FP | FN | Precision | Recall |")
    lines.append("|---|---|---|---|---|---|")
    for ct, s in sorted(stats["clarify_per_type"].items()):
        prec = safe_div(s["tp"], s["tp"] + s["fp"]) * 100
        rec = safe_div(s["tp"], s["tp"] + s["fn"]) * 100
        lines.append(f"| {ct} | {s['tp']} | {s['fp']} | {s['fn']} | {prec:.1f}% | {rec:.1f}% |")
    lines.append("")

    lines.append("## 5. Retrieval Sections (Planning Agent)\n")
    lines.append(f"- Section hit rate: **{stats['sections_hit']}/{stats['sections_total']} = {safe_div(stats['sections_hit'], stats['sections_total']) * 100:.1f}%**")
    lines.append("  > planning agent 勾的 sections ∩ 標註預期 sections 非空算命中\n")

    lines.append("## 6. OOD Detection\n")
    ood_prec = safe_div(stats["ood_tp"], stats["ood_tp"] + stats["ood_fp"]) * 100
    ood_rec = safe_div(stats["ood_tp"], stats["ood_tp"] + stats["ood_fn"]) * 100
    lines.append("| Metric | Count |")
    lines.append("|---|---|")
    lines.append(f"| TP (預期 OOD 且預測 OOD) | {stats['ood_tp']} |")
    lines.append(f"| FP (預期非 OOD 卻預測 OOD) | {stats['ood_fp']} |")
    lines.append(f"| FN (預期 OOD 卻預測非 OOD) | {stats['ood_fn']} |")
    lines.append(f"| TN (預期非 OOD 且預測非 OOD) | {stats['ood_tn']} |")
    lines.append(f"\n- **Precision: {ood_prec:.1f}%  /  Recall: {ood_rec:.1f}%**\n")

    lines.append("## 7. Summary（給論文直接抄）\n")
    lines.append(f"On a held-out test set of {stats['n_scenarios']} scenarios "
                f"({stats['n_turns']} turns) generated by ChatGPT and Gemini "
                f"(distinct from training prototypes), our system achieves:")
    lines.append("")
    lines.append(f"- **Task strict accuracy**: {safe_div(stats['task_strict'], stats['task_strict_total']) * 100:.1f}%")
    lines.append(f"- **Task compat accuracy**: {safe_div(stats['task_compat'], stats['task_compat_total']) * 100:.1f}%")
    lines.append(f"- **Domain accuracy**: {safe_div(stats['domain_hit'], stats['domain_total']) * 100:.1f}%")
    lines.append(f"- **Memory action accuracy**: {safe_div(stats['memory_correct'], stats['memory_total']) * 100:.1f}%")
    lines.append(f"- **Section hit rate**: {safe_div(stats['sections_hit'], stats['sections_total']) * 100:.1f}%")
    lines.append(f"- **OOD precision / recall**: {ood_prec:.1f}% / {ood_rec:.1f}%")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help=f"標註後的 held-out JSON（預設：{DEFAULT_INPUT}）")
    parser.add_argument("--report", default=DEFAULT_REPORT,
                        help=f"Markdown 報告輸出路徑（預設：{DEFAULT_REPORT}）")
    parser.add_argument("--code", default=ACCESS_CODE)
    parser.add_argument("--delay", type=float, default=TURN_DELAY)
    parser.add_argument("--max-scenarios", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 找不到標註檔：{args.input}")
        print(f"[提示] 請先跑 held_out_review.py 完成標註")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    # 過濾出有 annotations 的
    scenarios = [s for s in scenarios if s.get("annotations")]
    if args.max_scenarios:
        scenarios = scenarios[:args.max_scenarios]

    print(f"[INFO] 載入 {len(scenarios)} 個已標註 scenario")

    if not login(args.code):
        print(f"[ERROR] 登入失敗")
        sys.exit(1)

    print(f"[INFO] 開始評估...")
    t0 = time.time()
    stats = run_eval(scenarios, delay=args.delay)
    elapsed = time.time() - t0

    # 輸出報告
    report = to_markdown_report(stats, len(scenarios))
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report)

    # 同時存原始 stats
    json_path = args.report.replace(".md", "_raw.json")
    with open(json_path, "w", encoding="utf-8") as f:
        # Counter / defaultdict 要先轉
        stats_copy = dict(stats)
        stats_copy["clarify_buckets"] = dict(stats["clarify_buckets"])
        stats_copy["clarify_per_type"] = {k: dict(v) for k, v in stats["clarify_per_type"].items()}
        json.dump(stats_copy, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[完成] 耗時 {elapsed:.1f} 秒")
    print(f"[報告] {args.report}")
    print(f"[原始] {json_path}")
    print()
    # 印簡短摘要
    print("=" * 60)
    print("Held-out Evaluation Summary")
    print("=" * 60)
    print(f"  Task strict   : {safe_div(stats['task_strict'], stats['task_strict_total']) * 100:.1f}%")
    print(f"  Task compat   : {safe_div(stats['task_compat'], stats['task_compat_total']) * 100:.1f}%")
    print(f"  Domain        : {safe_div(stats['domain_hit'], stats['domain_total']) * 100:.1f}%")
    print(f"  Memory        : {safe_div(stats['memory_correct'], stats['memory_total']) * 100:.1f}%")
    print(f"  Sections      : {safe_div(stats['sections_hit'], stats['sections_total']) * 100:.1f}%")
    ood_prec = safe_div(stats["ood_tp"], stats["ood_tp"] + stats["ood_fp"]) * 100
    ood_rec = safe_div(stats["ood_tp"], stats["ood_tp"] + stats["ood_fn"]) * 100
    print(f"  OOD prec/rec  : {ood_prec:.1f}% / {ood_rec:.1f}%")


if __name__ == "__main__":
    main()
