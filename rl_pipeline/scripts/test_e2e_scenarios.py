"""
test_e2e_scenarios.py — 大量 Scenario 批量測試 + 統計分析

功能：
  - 從 generated_scenarios.json 讀取所有 scenario，對 app.py 發送對話
  - 記錄每輪 Memory Agent / Planning Agent / Task 分類的決策
  - 輸出正確率、混淆矩陣、錯誤分析報告
  - 可選：儲存完整結果至 JSON 供後續分析

用法：
  python rl_pipeline/scripts/test_e2e_scenarios.py
  python rl_pipeline/scripts/test_e2e_scenarios.py --max-scenarios 20
  python rl_pipeline/scripts/test_e2e_scenarios.py --save-report report.json
  python rl_pipeline/scripts/test_e2e_scenarios.py --code XXXXXXXX  # 換另一組 access code
"""

import requests
import json
import time
import sys
import os
import argparse
import random
from collections import defaultdict, Counter
from datetime import datetime

# ── 設定 ─────────────────────────────────────────────────────────────────────
BASE_URL = "http://127.0.0.1:5001"
ACCESS_CODE = "IX4UZSXJ"          # 預設測試帳號
SCENARIOS_JSON = os.path.join(os.path.dirname(__file__), "generated_scenarios.json")
TURN_DELAY = 0.6                  # 每輪發送間隔（秒）

# Memory Agent 決策規則（根據 turn index 與對話脈絡定義期望值）
# 規則：
#   - Turn 0 永遠應 REFRESH（全新對話）
#   - 同 category（intra_category）後續 turns 應 STAY
#   - cross_category 且問題明顯跳 domain 時應 REFRESH（由啟發式判斷）
# 新設計 (Phase B)：Memory Agent 2 分類（CLARIFY 移至 clarify_type 屬性）
MEMORY_LABELS = ["STAY", "REFRESH"]
PLANNING_SECTIONS = ["assessment", "observation", "training", "suggestion",
                     "community_resources", "external_gpt"]

# Task category → 預期主要 task label 映射（用於 Task 分類正確率評估）
CATEGORY_TASK_MAP = {
    "A": ["A"],
    "B": ["B"],
    "C": ["C"],
    "D": ["D"],
    "E": ["E"],
    "F": ["F"],
    "G": ["G"],
    "H": ["H"],
    "I": ["I"],
    "J": ["J"],
    "K": ["K"],
    "L": ["L"],
    "M": ["M"],
    "N": ["N"],
    # 跨類別：接受主或副任務命中
    "A→B": ["A", "B"],
    "B→E": ["B", "E"],
    "C→D": ["C", "D"],
    "C→E": ["C", "E"],
    "C→G": ["C", "G"],
    "D→E": ["D", "E"],
    "D→F": ["D", "F"],
    "F→M": ["F", "M"],
    "H→K": ["H", "K"],
    "I→J": ["I", "J"],
    "J→K": ["J", "K"],
    "L→B": ["L", "B"],
    "M→G": ["M", "G"],
    # 新增（標籤修正後產生的新 cross_category）
    "A→C": ["A", "C"],
    "A→E": ["A", "E"],
    "A→G": ["A", "G"],
    "A→L": ["A", "L"],
    "E→F": ["E", "F"],
    "J→E": ["J", "E"],
    "M→E": ["M", "E"],
    # 新增（Layer 1 修改後：H/K 個人化情境）
    "H→D": ["H", "D"],
    "D→H": ["D", "H"],
    "K→D": ["K", "D"],
    "H→C": ["H", "C"],
}

# Task 相容群 — 接受 task drift 為正確（解決測試假設「同 scenario = 同 task」的問題）
# 原則：看報告時聊到影響、居家練習、追蹤等相關話題是自然對話流程
TASK_COMPATIBILITY = {
    "A": {"A", "B", "D", "L", "N"},          # 報告總覽 ↔ 分數/剖面/追蹤/進步
    "B": {"A", "B", "D", "N"},               # 分數解讀 ↔ 總覽/剖面/進步
    "C": {"C", "D", "E", "G", "H"},          # 臨床觀察 ↔ 剖面/居家/評估需求/轉介
    "D": {"A", "C", "D", "E", "G", "H"},     # 能力剖面 ↔ 總覽/觀察/居家/評估/轉介（依狀況決定機構）
    "E": {"C", "E", "F"},                    # 居家訓練 ↔ 觀察/作息
    "F": {"E", "F", "M"},                    # 作息融入 ↔ 居家/情緒
    "G": {"C", "D", "G", "L"},               # 評估需求 ↔ 觀察/剖面/追蹤
    "H": {"H", "I", "K", "G", "D", "C"},     # 轉介 ↔ 隱私/補助/評估/能力剖面/觀察（個人化機構查詢）
    "I": {"H", "I", "A"},                    # 隱私 ↔ 轉介/總覽
    "J": {"C", "E", "F", "J", "M"},          # 學校合作 ↔ 觀察/居家/作息/情緒
    "K": {"H", "K", "D"},                    # 補助 ↔ 轉介/能力剖面（補助資格常依能力）
    "L": {"B", "G", "L", "N"},               # 追蹤 ↔ 分數/評估/進步
    "M": {"E", "F", "G", "J", "M"},          # 情緒支持 ↔ 居家/作息/評估/學校
    "N": {"B", "L", "N"},                    # 進步查詢 ↔ 分數/追蹤
}

# OOD scenarios 的特殊驗證：期望至少有一輪觸發 clarify_type="OUT_OF_DOMAIN"
OOD_CATEGORY = "OOD"


def _build_compat_set(category: str) -> set:
    """取得此 category 下可接受的所有 task（聯集 CATEGORY_TASK_MAP 中每個期望 task 的相容群）"""
    expected = CATEGORY_TASK_MAP.get(category, [])
    compat = set()
    for t in expected:
        compat |= TASK_COMPATIBILITY.get(t, {t})
    return compat or set(expected)

# ── HTTP 工具 ─────────────────────────────────────────────────────────────────
http = requests.Session()


def login(code: str) -> bool:
    try:
        r = http.post(f"{BASE_URL}/api/login_with_code", json={"code": code}, timeout=120)
        data = r.json()
        if data.get("status") == "success":
            print(f"[LOGIN] {data.get('child_name', '?')} ({code})")
            return True
        else:
            print(f"[LOGIN FAILED] {data}")
            return False
    except Exception as e:
        print(f"[LOGIN ERROR] {e}")
        return False


def new_chat():
    try:
        http.post(f"{BASE_URL}/api/new_chat", timeout=120)
    except Exception:
        pass


def send_message(message: str) -> dict:
    try:
        r = http.post(f"{BASE_URL}/api/chat", json={"message": message}, timeout=120)
        return r.json()
    except Exception as e:
        return {"error": str(e), "flow_state": {}, "message": ""}


def extract_flow(resp: dict) -> dict:
    """從 API response 提取關鍵欄位"""
    flow = resp.get("flow_state", {}) or {}
    planning = flow.get("planning_info", {}) or {}
    return {
        "task":           flow.get("task_pred", "?"),
        "secondary":      flow.get("secondary_tasks", []),
        "memory_action":  flow.get("memory_action", "?"),
        "retrieval_action": flow.get("retrieval_action", "?"),
        "semantic_flow":  flow.get("semantic_flow", "?"),
        "top_domain":     flow.get("top_domain", "?"),
        "active_domains": flow.get("active_domains", []),
        "planning_active": planning.get("active", []),
        "planning_probs":  planning.get("probs", {}),
        "num_candidates": flow.get("num_candidates", 0),
        "is_ambiguous":   flow.get("is_ambiguous", False),
        "clarify_type":   flow.get("clarify_type"),        # 新設計
        "anchor_turn":    flow.get("anchor_turn"),         # 新設計
    }


# ── 正確性判斷 ─────────────────────────────────────────────────────────────────

def infer_expected_memory(
    turn_idx: int,
    scenario_type: str,
    prev_tasks: list,
    current_task: str = None,
):
    """
    根據情境規則推斷期望的 Memory 決策（修正版：允許 task drift）。

    回傳：
      - 字串 "REFRESH" / "STAY"  → 硬期望（必須匹配）
      - 集合 {"STAY", "REFRESH"} → 軟期望（兩者皆可，視為正確）
      - None                     → 不設期望，不計入統計
    """
    # OOD 情境：不對 Memory 設期望（OOD 是 clarify_type 的責任）
    if scenario_type == "out_of_domain":
        return None

    if turn_idx == 0:
        return "REFRESH"  # 第一輪永遠應重置

    # intra_category：若 current_task 跳出前一輪 task 的相容群 → 接受 STAY 或 REFRESH
    if scenario_type == "intra_category":
        if prev_tasks and current_task:
            prev_task = prev_tasks[-1] if prev_tasks else None
            if prev_task:
                prev_compat = TASK_COMPATIBILITY.get(prev_task, {prev_task})
                if current_task not in prev_compat:
                    return {"STAY", "REFRESH"}  # task drift → 兩者皆可
        return "STAY"  # 正常延續

    # cross_category：本來就跨類，兩者皆可
    return {"STAY", "REFRESH"}


def task_hit(predicted_tasks: list, category: str) -> bool:
    """[嚴格] 判斷預測的 task 是否命中此 category 期望的任何 task"""
    if category == OOD_CATEGORY:
        return True  # OOD 情境不評估 task 命中（task 必為某個閉集標籤，沒意義）
    expected = CATEGORY_TASK_MAP.get(category, [])
    if not expected:
        return True             # 未知 category → 跳過
    return any(t in predicted_tasks for t in expected)


def task_hit_compat(predicted_tasks: list, category: str) -> bool:
    """[相容] 判斷預測的 task 是否落在此 category 的相容群內（允許自然 task drift）"""
    if category == OOD_CATEGORY:
        return True  # OOD 情境不評估 task 相容
    compat = _build_compat_set(category)
    if not compat:
        return True
    return any(t in compat for t in predicted_tasks)


# ── 混淆矩陣工具 ──────────────────────────────────────────────────────────────

class ConfusionMatrix:
    def __init__(self, labels: list):
        self.labels = labels
        self.matrix = defaultdict(lambda: defaultdict(int))  # [true][pred]
        self.total = 0

    def add(self, true_label: str, pred_label: str):
        self.matrix[true_label][pred_label] += 1
        self.total += 1

    def accuracy(self) -> float:
        correct = sum(self.matrix[l][l] for l in self.labels)
        return correct / self.total if self.total else 0.0

    def per_class_recall(self) -> dict:
        result = {}
        for true_l in self.labels:
            row_total = sum(self.matrix[true_l].values())
            correct = self.matrix[true_l][true_l]
            result[true_l] = correct / row_total if row_total else 0.0
        return result

    def per_class_precision(self) -> dict:
        result = {}
        for pred_l in self.labels:
            col_total = sum(self.matrix[t][pred_l] for t in self.labels)
            correct = self.matrix[pred_l][pred_l]
            result[pred_l] = correct / col_total if col_total else 0.0
        return result

    def print_matrix(self, title: str):
        col_w = 12
        print(f"\n{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}")
        # 標題列
        header = "True\\Pred".ljust(col_w) + "".join(l[:9].ljust(col_w) for l in self.labels)
        print(f"  {header}")
        print(f"  {'─' * (col_w * (len(self.labels) + 1))}")
        for true_l in self.labels:
            row_total = sum(self.matrix[true_l].values())
            if row_total == 0:
                continue
            row = true_l[:9].ljust(col_w)
            for pred_l in self.labels:
                val = self.matrix[true_l][pred_l]
                cell = f"{val}".ljust(col_w)
                row += cell
            row += f"  (n={row_total})"
            print(f"  {row}")

    def to_dict(self) -> dict:
        return {
            "labels": self.labels,
            "matrix": {t: dict(self.matrix[t]) for t in self.labels},
            "accuracy": self.accuracy(),
            "recall": self.per_class_recall(),
            "precision": self.per_class_precision(),
        }


# ── 主測試邏輯 ────────────────────────────────────────────────────────────────

def run_scenario(scenario: dict) -> dict:
    """
    執行單一 scenario，回傳每輪的記錄與錯誤摘要。
    """
    name = scenario.get("name", "unnamed")
    steps = scenario.get("steps", [])
    meta = scenario.get("metadata", {})
    category = meta.get("category", "?")
    sc_type = meta.get("type", "intra_category")

    new_chat()
    time.sleep(0.2)

    turn_records = []
    errors = []
    prev_predicted_tasks = []

    for turn_idx, query in enumerate(steps):
        resp = send_message(query)
        time.sleep(TURN_DELAY)

        if "error" in resp and not resp.get("flow_state"):
            turn_records.append({
                "turn": turn_idx, "query": query,
                "error": resp["error"],
                "memory": "?", "task": "?", "planning_active": []
            })
            errors.append(f"T{turn_idx}: API 錯誤 — {resp['error']}")
            continue

        flow = extract_flow(resp)
        memory = flow["memory_action"]
        task = flow["task"]
        secondary = flow["secondary"]
        planning_active = flow["planning_active"]

        # Task 命中判斷：嚴格（原期望）+ 相容（接受 task drift）
        all_predicted_tasks = [task] + secondary
        task_correct = task_hit(all_predicted_tasks, category)
        task_compat_correct = task_hit_compat(all_predicted_tasks, category)

        # 期望 Memory 決策（傳入 current_task 以判斷是否跨相容群）
        expected_memory = infer_expected_memory(
            turn_idx, sc_type, prev_predicted_tasks,
            current_task=task,
        )

        # Memory 正確性（支援硬期望字串 / 軟期望集合）
        memory_correct = None
        if expected_memory is not None:
            if isinstance(expected_memory, set):
                memory_correct = (memory in expected_memory)
                _expected_display = "/".join(sorted(expected_memory))
            else:
                memory_correct = (memory == expected_memory)
                _expected_display = expected_memory
            if not memory_correct:
                errors.append(
                    f"T{turn_idx}: Memory 期望={_expected_display}, 實際={memory} "
                    f"| query=「{query[:30]}...」"
                )

        # Task 錯誤記錄：只在「相容群也不中」時報錯（真正的誤判，不是 task drift）
        if not task_compat_correct:
            errors.append(
                f"T{turn_idx}: Task 相容群{sorted(_build_compat_set(category))} 未命中 "
                f"實際={task}+{secondary} | query=「{query[:30]}...」"
            )

        record = {
            "turn": turn_idx,
            "query": query,
            "memory": memory,
            "expected_memory": expected_memory,
            "memory_correct": memory_correct,
            "task": task,
            "secondary": secondary,
            "task_correct": task_correct,
            "task_compat_correct": task_compat_correct,
            "clarify_type": flow.get("clarify_type"),     # 新設計：clarify 屬性
            "anchor_turn": flow.get("anchor_turn"),       # 新設計：錨點輪
            "planning_active": planning_active,
            "top_domain": flow["top_domain"],
            "num_candidates": flow["num_candidates"],
            "semantic_flow": flow["semantic_flow"],
            "reply_preview": resp.get("message", "")[:80],
        }
        turn_records.append(record)
        prev_predicted_tasks = all_predicted_tasks

    # OOD 情境特殊驗證：期望至少一輪觸發 OUT_OF_DOMAIN
    if category == OOD_CATEGORY:
        triggered_ood = any(
            r.get("clarify_type") == "OUT_OF_DOMAIN"
            for r in turn_records
        )
        if not triggered_ood:
            errors.append(
                f"OOD 期望觸發 clarify_type=OUT_OF_DOMAIN，但 {len(turn_records)} 輪皆未觸發"
            )

    return {
        "name": name,
        "category": category,
        "type": sc_type,
        "turns": turn_records,
        "errors": errors,
    }


def run_all_scenarios(
    scenarios: list,
    max_scenarios: int = None,
    shuffle: bool = False,
) -> dict:
    """
    批量執行所有 scenarios，累積統計。
    """
    if shuffle:
        scenarios = scenarios.copy()
        random.shuffle(scenarios)
    if max_scenarios:
        scenarios = scenarios[:max_scenarios]

    total = len(scenarios)
    print(f"\n[BATCH] 共 {total} 個 Scenario 準備開始測試")
    print("="*70)

    # 統計累積器
    memory_cm = ConfusionMatrix(MEMORY_LABELS)      # Memory 混淆矩陣（有期望值的）
    task_results = []                                 # (category, correct)
    scenario_results = []
    failed_scenarios = []                             # 有任何錯誤的情境

    # 每個 category 的正確率（嚴格 + 相容）
    category_task_hits = defaultdict(lambda: {"hit": 0, "total": 0})
    category_task_compat_hits = defaultdict(lambda: {"hit": 0, "total": 0})
    # Memory 按情境類型分
    memory_by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    # Clarify 類型統計（新設計）
    clarify_type_counts = defaultdict(int)

    for i, sc in enumerate(scenarios):
        name = sc.get("name", f"scenario_{i}")
        print(f"\n[{i+1}/{total}] {name}", flush=True)

        result = run_scenario(sc)
        scenario_results.append(result)

        category = result["category"]
        sc_type = result["type"]
        has_error = bool(result["errors"])

        if has_error:
            failed_scenarios.append({
                "name": name,
                "category": category,
                "errors": result["errors"],
            })

        for rec in result["turns"]:
            # Clarify 類型分布統計
            ct = rec.get("clarify_type")
            clarify_type_counts[ct or "None"] += 1

            # Memory 統計（只在有期望值的 turn 計算）
            if rec.get("expected_memory") is not None:
                true_m = rec["expected_memory"]
                pred_m = rec["memory"] if rec["memory"] in MEMORY_LABELS else "STAY"
                # 軟期望（集合）：跳過混淆矩陣累積（無單一 ground truth 可記）
                # 但仍計入 memory_by_type 統計
                if not isinstance(true_m, set):
                    memory_cm.add(true_m, pred_m)
                m_correct = rec.get("memory_correct", False)
                memory_by_type[sc_type]["total"] += 1
                if m_correct:
                    memory_by_type[sc_type]["correct"] += 1

            # Task 統計（嚴格 + 相容雙軌）
            if rec.get("task_correct") is not None:
                category_task_hits[category]["total"] += 1
                category_task_compat_hits[category]["total"] += 1
                if rec["task_correct"]:
                    category_task_hits[category]["hit"] += 1
                if rec.get("task_compat_correct"):
                    category_task_compat_hits[category]["hit"] += 1

        # 即時進度印出
        if has_error:
            for e in result["errors"]:
                print(f"   ✗ {e}")
        else:
            print(f"   ✓ 全通過 ({len(result['turns'])} turns)")

    return {
        "total_scenarios": total,
        "failed_count": len(failed_scenarios),
        "memory_cm": memory_cm,
        "category_task_hits": dict(category_task_hits),
        "category_task_compat_hits": dict(category_task_compat_hits),
        "memory_by_type": dict(memory_by_type),
        "clarify_type_counts": dict(clarify_type_counts),
        "scenario_results": scenario_results,
        "failed_scenarios": failed_scenarios,
    }


def print_summary(stats: dict):
    """印出完整統計摘要"""
    total = stats["total_scenarios"]
    failed = stats["failed_count"]
    memory_cm: ConfusionMatrix = stats["memory_cm"]
    cat_hits: dict = stats["category_task_hits"]
    cat_compat_hits: dict = stats.get("category_task_compat_hits", {})
    mem_type: dict = stats["memory_by_type"]
    failed_list: list = stats["failed_scenarios"]

    print("\n" + "="*70)
    print("  📊 測試結果摘要")
    print("="*70)

    # ── 1. Scenario 整體通過率 ────────────────────────────────────────────
    pass_rate = (total - failed) / total * 100 if total else 0
    print(f"\n  ① Scenario 通過率")
    print(f"     總共: {total} | 通過: {total - failed} | 有錯誤: {failed}")
    print(f"     通過率: {pass_rate:.1f}%")

    # ── 2. Memory Agent 混淆矩陣 ──────────────────────────────────────────
    if memory_cm.total > 0:
        memory_cm.print_matrix("② Memory Agent 決策 混淆矩陣 (True\\ Pred)")
        mem_acc = memory_cm.accuracy()
        recall = memory_cm.per_class_recall()
        precision = memory_cm.per_class_precision()

        print(f"\n  Memory Agent 整體準確率: {mem_acc*100:.1f}% (n={memory_cm.total})")
        print(f"\n  各類別 Recall (召回率):")
        for lbl in MEMORY_LABELS:
            r = recall.get(lbl, 0)
            truth_count = sum(memory_cm.matrix[lbl].values())
            print(f"    {lbl:<10}: {r*100:5.1f}%  (n={truth_count})")
        print(f"\n  各類別 Precision (精確率):")
        for lbl in MEMORY_LABELS:
            p = precision.get(lbl, 0)
            print(f"    {lbl:<10}: {p*100:5.1f}%")

        print(f"\n  Memory 正確率 by 情境類型:")
        for sc_type, d in sorted(mem_type.items()):
            t = d["total"]
            c = d["correct"]
            pct = c / t * 100 if t else 0
            print(f"    {sc_type:<20}: {c}/{t} = {pct:.1f}%")
    else:
        print("\n  ② Memory Agent: 無有效評估資料")

    # ── 3. Task 分類正確率（by category）────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  ③ Task 分類正確率 (by Category)")
    print(f"{'─'*60}")

    if cat_hits:
        # 按 category 排序
        all_cats = sorted(cat_hits.keys())
        total_task_hit = sum(d["hit"] for d in cat_hits.values())
        total_task_all = sum(d["total"] for d in cat_hits.values())
        overall_task_acc = total_task_hit / total_task_all * 100 if total_task_all else 0

        # 相容群整體命中率
        total_compat_hit = sum(d["hit"] for d in cat_compat_hits.values()) if cat_compat_hits else 0
        total_compat_all = sum(d["total"] for d in cat_compat_hits.values()) if cat_compat_hits else 0
        overall_compat_acc = total_compat_hit / total_compat_all * 100 if total_compat_all else 0

        print(f"\n  整體 Task 命中率（嚴格）: {total_task_hit}/{total_task_all} = {overall_task_acc:.1f}%")
        if cat_compat_hits:
            print(f"  整體 Task 命中率（相容群）: {total_compat_hit}/{total_compat_all} = {overall_compat_acc:.1f}%")
        print()
        print(f"  {'Category':<20} {'嚴格':>8}  {'相容':>8}  {'Hit/Total':>12}")
        print(f"  {'─'*20} {'─'*8}  {'─'*8}  {'─'*12}")

        # 分兩組：intra / cross
        for group, prefix in [("── 單一 Category ──", ""), ("── 跨 Category ──", "→")]:
            group_cats = [c for c in all_cats if ("→" in c) == (prefix == "→")]
            if not group_cats:
                continue
            print(f"  {group}")
            for cat in group_cats:
                d = cat_hits[cat]
                t = d["total"]
                h = d["hit"]
                pct = h / t * 100 if t else 0
                # 相容率
                cd = cat_compat_hits.get(cat, {"hit": 0, "total": 0})
                cpct = cd["hit"] / cd["total"] * 100 if cd["total"] else 0
                bar = "█" * int(cpct / 10) + "░" * (10 - int(cpct / 10))
                print(f"    {cat:<18} {pct:5.1f}%  {cpct:5.1f}%  [{bar}]  ({h}/{t} | compat {cd['hit']}/{cd['total']})")
    else:
        print("  無 Task 評估資料")

    # ── 4. 失敗 Scenario 清單 + 錯誤詳情 ────────────────────────────────
    if failed_list:
        print(f"\n{'─'*60}")
        print(f"  ④ 錯誤 Scenario 清單 ({len(failed_list)} 個)")
        print(f"{'─'*60}")

        # 按 category 分組顯示
        by_cat = defaultdict(list)
        for fs in failed_list:
            by_cat[fs["category"]].append(fs)

        for cat, items in sorted(by_cat.items()):
            print(f"\n  [{cat}]")
            for item in items:
                print(f"    ● {item['name']}")
                for err in item["errors"][:5]:  # 最多顯示 5 個錯誤
                    print(f"        → {err}")
                if len(item["errors"]) > 5:
                    print(f"        → ...（共 {len(item['errors'])} 個錯誤）")
    else:
        print(f"\n  ④ 所有 Scenario 均無錯誤！🎉")

    # ── 5. Clarify 類型分布（新設計）──────────────────────────────────────
    ct_counts = stats.get("clarify_type_counts", {})
    if ct_counts:
        total_turns = sum(ct_counts.values())
        print(f"\n{'─'*60}")
        print(f"  ⑤ Clarify 類型分布（新設計：clarify 屬性）")
        print(f"{'─'*60}")
        for ct_name in ["None", "DOMAIN_HARD", "CONTEXT_MISSING", "SLOT_REGION", "TASK_SOFT", "OUT_OF_DOMAIN"]:
            cnt = ct_counts.get(ct_name, 0)
            pct = cnt / total_turns * 100 if total_turns else 0
            print(f"    {ct_name:<18}: {cnt:4d} ({pct:5.1f}%)")

    print("\n" + "="*70)


def save_report(stats: dict, path: str):
    """將詳細結果儲存至 JSON（memory_cm 需序列化）"""
    memory_cm: ConfusionMatrix = stats["memory_cm"]
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_scenarios": stats["total_scenarios"],
        "failed_count": stats["failed_count"],
        "memory_confusion_matrix": memory_cm.to_dict(),
        "category_task_hits": stats["category_task_hits"],
        "category_task_compat_hits": stats.get("category_task_compat_hits", {}),
        "memory_by_type": stats["memory_by_type"],
        "failed_scenarios": stats["failed_scenarios"],
        # 每個 scenario 的詳細資料
        "scenario_details": [
            {
                "name": r["name"],
                "category": r["category"],
                "type": r["type"],
                "has_error": bool(r["errors"]),
                "errors": r["errors"],
                "turns": [
                    {k: v for k, v in t.items() if k != "reply_preview"}
                    for t in r["turns"]
                ],
            }
            for r in stats["scenario_results"]
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  [REPORT] 詳細結果已儲存至：{path}")


# ── CLI 入口 ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="EduBot E2E 批量 Scenario 測試腳本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--code", default=ACCESS_CODE,
                        help=f"Access code（預設: {ACCESS_CODE}）")
    parser.add_argument("--max-scenarios", type=int, default=None,
                        help="最多測試幾個 Scenario（不指定 = 全部）")
    parser.add_argument("--shuffle", action="store_true",
                        help="隨機打亂 Scenario 執行順序")
    parser.add_argument("--scenarios-file", default=SCENARIOS_JSON,
                        help=f"Scenario JSONL/JSON 檔案路徑（預設: {SCENARIOS_JSON}）")
    parser.add_argument("--save-report", default=None,
                        help="儲存完整 JSON 報告至指定路徑")
    parser.add_argument("--delay", type=float, default=TURN_DELAY,
                        help=f"每輪等待秒數（預設: {TURN_DELAY}）")
    parser.add_argument("--category", default=None,
                        help="只測試指定 category（例如 --category M）")
    return parser.parse_args()


def main():
    global TURN_DELAY
    args = parse_args()
    TURN_DELAY = args.delay

    print("="*70)
    print("  EduBot E2E Batch Scenario Tester")
    print(f"  時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # 1. 載入 Scenarios
    scenarios_path = args.scenarios_file
    if not os.path.exists(scenarios_path):
        print(f"[ERROR] 找不到 Scenario 檔案：{scenarios_path}")
        sys.exit(1)

    with open(scenarios_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    print(f"[LOAD] 載入 {len(scenarios)} 個 Scenario from {scenarios_path}")

    # 依 category 篩選
    if args.category:
        scenarios = [
            s for s in scenarios
            if s.get("metadata", {}).get("category", "") == args.category
        ]
        print(f"[FILTER] 篩選 category={args.category}，剩餘 {len(scenarios)} 個")

    if not scenarios:
        print("[ERROR] 無可用 Scenario")
        sys.exit(1)

    # 2. 登入
    if not login(args.code):
        sys.exit(1)

    # 3. 批量執行
    stats = run_all_scenarios(
        scenarios=scenarios,
        max_scenarios=args.max_scenarios,
        shuffle=args.shuffle,
    )

    # 4. 輸出摘要
    print_summary(stats)

    # 5. 儲存 JSON 報告
    if args.save_report:
        save_report(stats, args.save_report)

    # 回傳 exit code（有錯誤則為 1）
    sys.exit(1 if stats["failed_count"] > 0 else 0)


if __name__ == "__main__":
    main()
