"""
check_all.py — 訓練後模型完整驗證腳本

涵蓋：
  1. Memory Agent  — STAY / REFRESH / CLARIFY 決策正確性（12 情境）
  2. Planning Agent — 14 個 Task 的 Section 選擇正確性（A-N）
  3. 邊界情境     — 極端輸入、低信心 fallback、連續 STAY 等
  4. 信心校準檢查 — 是否所有輸出都是 100%（過擬合警報）

用法：
  cd app_v7
  python rl_pipeline/scripts/check_all.py

結果解讀：
  ✅ 通過：決策符合預期
  ⚠️  可接受：決策在允許範圍內（多選一）
  ❌ 失敗：決策偏差，建議繼續訓練
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rl_pipeline.agents.memory.memory_agent import MemoryAgent
from rl_pipeline.agents.planner.planning_agent import PlanningAgent

# ─────────────────────────────────────────────
# 常數
# ─────────────────────────────────────────────
SECTION_LABELS = ["assessment", "observation", "training", "suggestion", "community_resources", "external_gpt"]

# 預訓練的 Ground Truth（與 pretrain_agents.py 的 TASK_SECTION_MAP 一致）
TASK_SECTION_MAP = {
    "A": [1, 1, 0, 0, 0, 0],
    "B": [1, 1, 0, 0, 0, 0],
    "C": [0, 1, 0, 1, 0, 0],
    "D": [1, 1, 0, 1, 0, 0],
    "E": [0, 0, 1, 1, 0, 0],
    "F": [0, 0, 1, 1, 0, 0],
    "G": [1, 0, 0, 1, 0, 0],
    "H": [0, 0, 0, 0, 1, 1],
    "I": [1, 0, 0, 0, 0, 0],
    "J": [0, 0, 1, 1, 0, 1],
    "K": [0, 0, 0, 0, 1, 1],
    "L": [1, 1, 0, 0, 0, 0],
    "M": [0, 0, 0, 1, 1, 0],
    "N": [1, 1, 0, 0, 0, 0],
}

TASK_NAME = {
    "A": "報告總覽",        "B": "分數/百分位解讀",  "C": "臨床觀察",
    "D": "能力剖面",        "E": "在家訓練",          "F": "融入作息",
    "G": "早療追蹤/必要性", "H": "轉介與在地資源",   "I": "報告隱私分享",
    "J": "學校合作",        "K": "補助/福利/申請",   "L": "後續追蹤/再評估",
    "M": "家長情緒支持",    "N": "進步查詢",
}


# ─────────────────────────────────────────────
# 輔助：色彩與格線
# ─────────────────────────────────────────────
def bar(p: float, width: int = 14) -> str:
    return "█" * int(p * width) + "░" * (width - int(p * width))

def prob_row(labels, probs) -> str:
    return "  ".join(f"{l[:4]}={p:.0%}" for l, p in zip(labels, probs))

def section_row(labels, probs, threshold=0.5) -> str:
    parts = []
    for l, p in zip(labels, probs):
        mark = "●" if p > threshold else "○"
        parts.append(f"{mark}{l[:6]}={p:.0%}")
    return "  ".join(parts)

def divider(char="─", width=68):
    print(char * width)


# ═══════════════════════════════════════════════════════════════
# PART 1：Memory Agent 測試
# ═══════════════════════════════════════════════════════════════
MEMORY_SCENARIOS = [
    # ── STAY 預期情境 ──────────────────────────────────────────
    {
        "name": "追問同一主題（應 STAY）",
        "desc": "同話題低熵延續，context_sim 高，TV 距離小",
        "state": {
            "entropy": 0.12, "tv_distance": 0.06, "topic_overlap": 0.88,
            "context_sim": 0.91, "turn_index_norm": 0.3, "query_len_norm": 0.55,
            "is_multi_domain": False, "prev_action_norm": 0.0, "consecutive_stay_count": 0.2,
        },
        "expect": ["STAY"],
    },
    {
        "name": "前一輪 REFRESH 後，用戶深入追問新話題（應 STAY）",
        "desc": "剛切換話題後，現在繼續問同個新話題",
        "state": {
            "entropy": 0.20, "tv_distance": 0.10, "topic_overlap": 0.80,
            "context_sim": 0.85, "turn_index_norm": 0.5, "query_len_norm": 0.60,
            "is_multi_domain": False, "prev_action_norm": 0.5, "consecutive_stay_count": 0.0,
        },
        "expect": ["STAY"],
    },
    {
        "name": "高信心單一領域延續（應 STAY）",
        "desc": "單領域低熵，overlap 極高",
        "state": {
            "entropy": 0.08, "tv_distance": 0.04, "topic_overlap": 0.95,
            "context_sim": 0.93, "turn_index_norm": 0.4, "query_len_norm": 0.70,
            "is_multi_domain": False, "prev_action_norm": 0.0, "consecutive_stay_count": 0.4,
        },
        "expect": ["STAY"],
    },
    # ── REFRESH 預期情境 ────────────────────────────────────────
    {
        "name": "明確切換話題（應 REFRESH）",
        "desc": "TV 距離大，overlap 極低，句子清晰完整",
        "state": {
            "entropy": 0.10, "tv_distance": 0.85, "topic_overlap": 0.05,
            "context_sim": 0.22, "turn_index_norm": 0.4, "query_len_norm": 0.70,
            "is_multi_domain": False, "prev_action_norm": 0.0, "consecutive_stay_count": 0.0,
        },
        "expect": ["REFRESH"],
    },
    {
        "name": "連續 STAY 過多，新問題出現（應 REFRESH）",
        "desc": "Stay 次數已達上限，TV 距離中等",
        "state": {
            "entropy": 0.30, "tv_distance": 0.40, "topic_overlap": 0.50,
            "context_sim": 0.55, "turn_index_norm": 0.9, "query_len_norm": 0.65,
            "is_multi_domain": False, "prev_action_norm": 0.0, "consecutive_stay_count": 1.0,
        },
        "expect": ["REFRESH", "CLARIFY"],
    },
    {
        "name": "跨領域明確切換（應 REFRESH）",
        "desc": "從粗大動作跳到補助福利，意圖清晰",
        "state": {
            "entropy": 0.15, "tv_distance": 0.90, "topic_overlap": 0.02,
            "context_sim": 0.18, "turn_index_norm": 0.6, "query_len_norm": 0.75,
            "is_multi_domain": True, "prev_action_norm": 0.0, "consecutive_stay_count": 0.2,
        },
        "expect": ["REFRESH"],
    },
    # ── CLARIFY 預期情境 ───────────────────────────────────────
    {
        "name": "模糊短句（應 CLARIFY）",
        "desc": "高熵、TV 距離大、句子極短",
        "state": {
            "entropy": 0.92, "tv_distance": 0.88, "topic_overlap": 0.03,
            "context_sim": 0.08, "turn_index_norm": 0.2, "query_len_norm": 0.10,
            "is_multi_domain": True, "prev_action_norm": 0.0, "consecutive_stay_count": 0.0,
        },
        "expect": ["CLARIFY"],
    },
    {
        "name": "多領域高不確定性（應 CLARIFY）",
        "desc": "跨多領域，entropy 高，句子不完整",
        "state": {
            "entropy": 0.85, "tv_distance": 0.70, "topic_overlap": 0.15,
            "context_sim": 0.12, "turn_index_norm": 0.3, "query_len_norm": 0.18,
            "is_multi_domain": True, "prev_action_norm": 0.0, "consecutive_stay_count": 0.0,
        },
        "expect": ["CLARIFY"],
    },
    {
        "name": "前一輪 CLARIFY 但用戶仍模糊（應 CLARIFY）",
        "desc": "連續不確定，需再次澄清",
        "state": {
            "entropy": 0.80, "tv_distance": 0.60, "topic_overlap": 0.10,
            "context_sim": 0.10, "turn_index_norm": 0.4, "query_len_norm": 0.15,
            "is_multi_domain": True, "prev_action_norm": 1.0, "consecutive_stay_count": 0.0,
        },
        "expect": ["CLARIFY"],
    },
    # ── 多選可接受情境 ─────────────────────────────────────────
    {
        "name": "中等信心多領域（可 STAY 或 CLARIFY）",
        "desc": "中等熵值，多領域，overlap 中等",
        "state": {
            "entropy": 0.55, "tv_distance": 0.45, "topic_overlap": 0.42,
            "context_sim": 0.58, "turn_index_norm": 0.5, "query_len_norm": 0.45,
            "is_multi_domain": True, "prev_action_norm": 0.0, "consecutive_stay_count": 0.4,
        },
        "expect": ["STAY", "CLARIFY"],
    },
    {
        "name": "弱切換（可 STAY 或 REFRESH）",
        "desc": "TV 距離中等，句子較長，overlap 略低",
        "state": {
            "entropy": 0.35, "tv_distance": 0.48, "topic_overlap": 0.35,
            "context_sim": 0.50, "turn_index_norm": 0.6, "query_len_norm": 0.60,
            "is_multi_domain": False, "prev_action_norm": 0.0, "consecutive_stay_count": 0.2,
        },
        "expect": ["STAY", "REFRESH"],
    },
    {
        "name": "高熵但句子完整（可 REFRESH 或 CLARIFY）",
        "desc": "句子長、清晰，但跨多領域熵值高",
        "state": {
            "entropy": 0.75, "tv_distance": 0.55, "topic_overlap": 0.20,
            "context_sim": 0.30, "turn_index_norm": 0.5, "query_len_norm": 0.80,
            "is_multi_domain": True, "prev_action_norm": 0.5, "consecutive_stay_count": 0.0,
        },
        "expect": ["REFRESH", "CLARIFY"],
    },
]


def check_memory_agent(agent: MemoryAgent) -> tuple:
    print("\n" + "═" * 68)
    print("  PART 1：Memory Agent 決策測試（12 情境）")
    print("═" * 68)

    pass_count = 0
    total = len(MEMORY_SCENARIOS)
    confidence_sum = 0.0
    all_100pct = True

    for s in MEMORY_SCENARIOS:
        state_tensor = agent._extract_features(s["state"])
        with torch.no_grad():
            logits = agent.policy_net(state_tensor)
            probs = F.softmax(logits, dim=1).squeeze().tolist()

        action_idx = int(np.argmax(probs))
        action_str = agent.action_space[action_idx]
        confidence = max(probs)
        confidence_sum += confidence

        if confidence < 1.0 - 1e-4:
            all_100pct = False

        ok = action_str in s["expect"]
        mark = "✅" if ok else ("⚠️ " if len(s["expect"]) > 1 else "❌")
        if ok:
            pass_count += 1

        print(f"\n{mark} {s['name']}")
        print(f"   {s['desc']}")
        print(f"   預期：{' 或 '.join(s['expect'])}　　決策：{action_str}（{confidence:.0%}）")
        print(f"   ", end="")
        for act, p in zip(agent.action_space, probs):
            print(f"[{bar(p,10)}] {act} {p:.1%}  ", end="")
        print()

    avg_conf = confidence_sum / total
    print(f"\n{'─'*68}")
    print(f"  Memory Agent 結果：{pass_count}/{total} 通過")
    print(f"  平均信心值：{avg_conf:.1%}  {'⚠️  所有輸出皆 100%，疑似過擬合！建議重新預訓練。' if all_100pct else '✅ 信心分布正常'}")
    return pass_count, total, all_100pct


# ═══════════════════════════════════════════════════════════════
# PART 2：Planning Agent 測試（14 Tasks）
# ═══════════════════════════════════════════════════════════════

def _build_planning_state(task: str, mask: list, entropy: float = 0.3) -> dict:
    """
    根據 mask 建立合理的 Planning Agent 輸入 state。
    mask 為 1 的 section 給高語義分數（0.75~0.90），為 0 的給低分（0.05~0.15）。
    """
    import random
    random.seed(42)
    keys = ["sem_assessment", "sem_observation", "sem_training",
            "sem_suggestion", "sem_community_resources", "sem_external_gpt"]
    state = {}
    for i, k in enumerate(keys):
        if mask[i] == 1:
            state[k] = round(random.uniform(0.72, 0.92), 3)
        else:
            state[k] = round(random.uniform(0.04, 0.18), 3)
    state["domain_entropy"] = entropy
    state["task_label"] = task
    return state


def check_planning_agent(agent: PlanningAgent) -> tuple:
    print("\n" + "═" * 68)
    print("  PART 2：Planning Agent Section 選擇測試（14 Tasks A-N）")
    print("═" * 68)

    pass_count = 0
    partial_count = 0
    total = len(TASK_SECTION_MAP)
    confidence_vals = []
    all_100pct = True

    for task, expected_mask in TASK_SECTION_MAP.items():
        state = _build_planning_state(task, expected_mask)
        state_tensor = agent._extract_features(state)

        with torch.no_grad():
            probs = agent.policy_net(state_tensor).squeeze().tolist()

        active = [SECTION_LABELS[i] for i, p in enumerate(probs) if p > 0.5]
        expected_secs = [SECTION_LABELS[i] for i, m in enumerate(expected_mask) if m == 1]

        # 信心 = 各預期 section 的平均機率
        expected_probs = [probs[i] for i, m in enumerate(expected_mask) if m == 1]
        avg_p = sum(expected_probs) / len(expected_probs) if expected_probs else 0.0
        confidence_vals.append(avg_p)

        # 計算信心分布是否全部趨近 0/1
        if any(0.05 < p < 0.95 for p in probs):
            all_100pct = False

        # 判斷通過條件
        hits = sum(1 for s in expected_secs if s in active)
        total_expected = len(expected_secs)
        exact = (sorted(active) == sorted(expected_secs))
        partial = (hits >= max(1, total_expected - 1))  # 允許差 1 個

        if exact:
            mark = "✅"
            pass_count += 1
        elif partial:
            mark = "⚠️ "
            partial_count += 1
        else:
            mark = "❌"

        print(f"\n{mark} Task {task} — {TASK_NAME[task]}")
        print(f"   預期：{expected_secs}")
        print(f"   啟動：{active if active else ['（無）']}")
        print(f"   ", end="")
        for lbl, p in zip(SECTION_LABELS, probs):
            exp = "↑" if SECTION_LABELS.index(lbl) < len(expected_mask) and expected_mask[SECTION_LABELS.index(lbl)] == 1 else " "
            print(f"{exp}[{bar(p,8)}]{lbl[:5]}={p:.0%}  ", end="")
        print()

    avg_conf = sum(confidence_vals) / len(confidence_vals) if confidence_vals else 0.0
    print(f"\n{'─'*68}")
    print(f"  Planning Agent 結果：{pass_count}/{total} 完全通過，{partial_count}/{total} 部分通過")
    print(f"  預期 section 平均信心：{avg_conf:.1%}  {'⚠️  輸出趨近 0/1，疑似飽和' if all_100pct else '✅ 分布合理'}")
    return pass_count, partial_count, total, all_100pct


# ═══════════════════════════════════════════════════════════════
# PART 3：邊界情境測試
# ═══════════════════════════════════════════════════════════════

EDGE_CASES = [
    {
        "name": "Memory — 全部特徵為 0（冷啟動）",
        "type": "memory",
        "state": {k: 0.0 for k in ["entropy", "tv_distance", "topic_overlap", "context_sim",
                                    "turn_index_norm", "query_len_norm", "is_multi_domain",
                                    "prev_action_norm", "consecutive_stay_count"]},
        "check": lambda action, probs: max(probs) < 0.99,
        "desc": "全零輸入下信心不應趨近 100%（否則梯度不流動）",
    },
    {
        "name": "Memory — entropy=1.0, tv=1.0（極端模糊）",
        "type": "memory",
        "state": {
            "entropy": 1.0, "tv_distance": 1.0, "topic_overlap": 0.0,
            "context_sim": 0.0, "turn_index_norm": 0.0, "query_len_norm": 0.0,
            "is_multi_domain": 1.0, "prev_action_norm": 0.0, "consecutive_stay_count": 0.0,
        },
        "check": lambda action, probs: action == "CLARIFY",
        "desc": "極端模糊輸入應輸出 CLARIFY",
    },
    {
        "name": "Memory — 全部特徵為 1（極端延續）",
        "type": "memory",
        "state": {
            "entropy": 0.0, "tv_distance": 0.0, "topic_overlap": 1.0,
            "context_sim": 1.0, "turn_index_norm": 1.0, "query_len_norm": 1.0,
            "is_multi_domain": 0.0, "prev_action_norm": 0.0, "consecutive_stay_count": 0.0,
        },
        "check": lambda action, probs: action == "STAY",
        "desc": "完美延續信號應輸出 STAY",
    },
    {
        "name": "Planning — Task H，all sem=0（規則 fallback 下）",
        "type": "planning",
        "state": {
            "sem_assessment": 0.0, "sem_observation": 0.0, "sem_training": 0.0,
            "sem_suggestion": 0.0, "sem_community_resources": 0.01, "sem_external_gpt": 0.01,
            "domain_entropy": 0.5, "task_label": "H",
        },
        "check": lambda active, probs: "community_resources" in active or "external_gpt" in active,
        "desc": "Task H 即使語義分數為 0，task one-hot 也應引導輸出 community/external",
    },
    {
        "name": "Planning — Task K，all sem=0（補助福利）",
        "type": "planning",
        "state": {
            "sem_assessment": 0.0, "sem_observation": 0.0, "sem_training": 0.0,
            "sem_suggestion": 0.0, "sem_community_resources": 0.01, "sem_external_gpt": 0.01,
            "domain_entropy": 0.5, "task_label": "K",
        },
        "check": lambda active, probs: "community_resources" in active or "external_gpt" in active,
        "desc": "Task K 應至少啟動 community_resources 或 external_gpt",
    },
    {
        "name": "Planning — Task N，all sem=0（進步查詢）",
        "type": "planning",
        "state": {
            "sem_assessment": 0.01, "sem_observation": 0.01, "sem_training": 0.0,
            "sem_suggestion": 0.0, "sem_community_resources": 0.0, "sem_external_gpt": 0.0,
            "domain_entropy": 0.3, "task_label": "N",
        },
        "check": lambda active, probs: "assessment" in active,
        "desc": "Task N（進步查詢）必須啟動 assessment",
    },
    {
        "name": "Planning — 多任務 H+K（轉介+補助同時問）",
        "type": "planning",
        "state": {
            "sem_assessment": 0.05, "sem_observation": 0.05, "sem_training": 0.05,
            "sem_suggestion": 0.05, "sem_community_resources": 0.75, "sem_external_gpt": 0.70,
            "domain_entropy": 0.4,
            "task_label": "H",
            "secondary_tasks": ["K"],
            "task_dist": {"H": 0.45, "K": 0.35, "A": 0.05},
        },
        "check": lambda active, probs: "community_resources" in active or "external_gpt" in active,
        "desc": "多任務 H+K 輸入應確保啟動 community_resources 或 external_gpt",
    },
    {
        "name": "Planning — 多任務 E+J（居家訓練+學校合作）",
        "type": "planning",
        "state": {
            "sem_assessment": 0.10, "sem_observation": 0.30, "sem_training": 0.75,
            "sem_suggestion": 0.70, "sem_community_resources": 0.05, "sem_external_gpt": 0.40,
            "domain_entropy": 0.35,
            "task_label": "E",
            "secondary_tasks": ["J"],
            "task_dist": {"E": 0.40, "J": 0.30, "F": 0.10},
        },
        "check": lambda active, probs: "training" in active and "suggestion" in active,
        "desc": "多任務 E+J 應至少啟動 training + suggestion",
    },
]


def check_edge_cases(mem_agent: MemoryAgent, plan_agent: PlanningAgent) -> tuple:
    print("\n" + "═" * 68)
    print("  PART 3：邊界情境測試（6 案例）")
    print("═" * 68)

    pass_count = 0
    total = len(EDGE_CASES)

    for ec in EDGE_CASES:
        if ec["type"] == "memory":
            state_tensor = mem_agent._extract_features(ec["state"])
            with torch.no_grad():
                logits = mem_agent.policy_net(state_tensor)
                probs = F.softmax(logits, dim=1).squeeze().tolist()
            action_idx = int(np.argmax(probs))
            action = mem_agent.action_space[action_idx]
            ok = ec["check"](action, probs)
            detail = f"決策：{action}（{max(probs):.0%}）"
        else:
            state_tensor = plan_agent._extract_features(ec["state"])
            with torch.no_grad():
                probs = plan_agent.policy_net(state_tensor).squeeze().tolist()
            active = [SECTION_LABELS[i] for i, p in enumerate(probs) if p > 0.5]
            ok = ec["check"](active, probs)
            detail = f"啟動：{active if active else ['（無）']}"

        mark = "✅" if ok else "❌"
        if ok:
            pass_count += 1
        print(f"\n{mark} {ec['name']}")
        print(f"   {ec['desc']}")
        print(f"   {detail}")

    return pass_count, total


# ═══════════════════════════════════════════════════════════════
# PART 4：信心校準整體報告
# ═══════════════════════════════════════════════════════════════

def calibration_report(mem_agent: MemoryAgent, plan_agent: PlanningAgent):
    """
    輸出所有預訓練情境的平均信心分布，快速判斷飽和程度。
    """
    print("\n" + "═" * 68)
    print("  PART 4：信心校準報告")
    print("═" * 68)

    # Memory
    mem_confs = []
    for s in MEMORY_SCENARIOS:
        t = mem_agent._extract_features(s["state"])
        with torch.no_grad():
            p = F.softmax(mem_agent.policy_net(t), dim=1).squeeze().tolist()
        mem_confs.append(max(p))

    mem_avg = sum(mem_confs) / len(mem_confs)
    mem_all100 = all(c > 0.999 for c in mem_confs)
    print(f"\n  Memory Agent — 最高信心平均：{mem_avg:.1%}")
    print(f"  {'❌ 所有情境皆 100%（Dropout/weight_decay 未生效或預訓練 epoch 過多）' if mem_all100 else '✅ 信心分布合理（未飽和）'}")

    # Planning
    plan_confs_hi = []  # 預期為 1 的 section
    plan_confs_lo = []  # 預期為 0 的 section
    for task, mask in TASK_SECTION_MAP.items():
        state = _build_planning_state(task, mask)
        t = plan_agent._extract_features(state)
        with torch.no_grad():
            p = plan_agent.policy_net(t).squeeze().tolist()
        for i, m in enumerate(mask):
            if m == 1:
                plan_confs_hi.append(p[i])
            else:
                plan_confs_lo.append(p[i])

    avg_hi = sum(plan_confs_hi) / len(plan_confs_hi) if plan_confs_hi else 0.0
    avg_lo = sum(plan_confs_lo) / len(plan_confs_lo) if plan_confs_lo else 0.0
    sep = avg_hi - avg_lo
    print(f"\n  Planning Agent —")
    print(f"    預期啟動 section 平均機率：{avg_hi:.1%}（目標 >0.65）")
    print(f"    預期關閉 section 平均機率：{avg_lo:.1%}（目標 <0.25）")
    print(f"    分離度（差值）：{sep:.1%}（目標 >0.40）")
    if sep > 0.40:
        print(f"  ✅ 分離度良好，模型已學到有效的 Task→Section 映射")
    elif sep > 0.20:
        print(f"  ⚠️  分離度尚可，建議繼續 RL 訓練以強化決策邊界")
    else:
        print(f"  ❌ 分離度過低，模型尚未學到有效映射，建議重新預訓練")


# ═══════════════════════════════════════════════════════════════
# 主程式
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "╔" + "═" * 66 + "╗")
    print("║" + "  早療 AI 系統 — 訓練後模型完整驗證報告".center(64) + "  ║")
    print("╚" + "═" * 66 + "╝")

    # 載入模型
    mem_path  = "rl_pipeline/agents/memory/models/memory_agent.pth"
    plan_path = "rl_pipeline/agents/planner/models/planning_agent.pth"

    print(f"\n  載入 Memory Agent：{mem_path}")
    mem_agent = MemoryAgent(model_path=mem_path)
    mem_agent.policy_net.eval()

    print(f"  載入 Planning Agent：{plan_path}")
    plan_agent = PlanningAgent(model_path=plan_path)
    plan_agent.policy_net.eval()

    # 執行各部分測試
    m_pass, m_total, m_saturated         = check_memory_agent(mem_agent)
    p_pass, p_partial, p_total, p_sat    = check_planning_agent(plan_agent)
    e_pass, e_total                      = check_edge_cases(mem_agent, plan_agent)
    calibration_report(mem_agent, plan_agent)

    # 最終總結
    print("\n" + "╔" + "═" * 66 + "╗")
    print("║" + "  最終總結".center(64) + "  ║")
    print("╠" + "═" * 66 + "╣")

    total_pass = m_pass + p_pass + e_pass
    total_all  = m_total + p_total + e_total
    pct = total_pass / total_all * 100 if total_all else 0

    print(f"║  Memory Agent   ：{m_pass:2d}/{m_total} 通過{'  ⚠️ 飽和' if m_saturated else ''}".ljust(66) + "  ║")
    print(f"║  Planning Agent ：{p_pass:2d}/{p_total} 完全通過，{p_partial} 部分通過{'  ⚠️ 飽和' if p_sat else ''}".ljust(66) + "  ║")
    print(f"║  邊界情境       ：{e_pass:2d}/{e_total} 通過".ljust(66) + "  ║")
    print(f"║  整體通過率     ：{total_pass}/{total_all}（{pct:.0f}%）".ljust(66) + "  ║")
    print("╠" + "═" * 66 + "╣")

    if pct >= 85 and not m_saturated and not p_sat:
        verdict = "✅  模型決策符合預期，可進行 RL 訓練。"
    elif pct >= 65:
        verdict = "⚠️   模型大致合理，建議先跑 1~2 輪 RL 後再評估。"
    else:
        verdict = "❌  模型偏差較大，建議重新預訓練後再 RL 訓練。"

    saturation_warning = ""
    if m_saturated or p_sat:
        saturation_warning = "\n║  ⚠️  偵測到飽和輸出（100%），請確認 Dropout + weight_decay 已啟用。"

    print(f"║  {verdict}".ljust(66) + "  ║")
    if saturation_warning:
        print(saturation_warning.ljust(66) + "  ║")
    print("╚" + "═" * 66 + "╝\n")

    # 建議後續步驟
    print("【建議後續步驟】")
    if m_saturated or p_sat:
        print("  1. 刪除舊模型，重新執行預訓練：")
        print("       python rl_pipeline/scripts/pretrain_agents.py")
        print("  2. 重新執行本腳本確認飽和問題已解決")
    elif pct >= 65:
        print("  1. 收集真實對話資料：")
        print("       python rl_pipeline/scripts/auto_query_bot.py")
        print("  2. 執行 RL 訓練：")
        print("       python rl_pipeline/scripts/unified_train_db.py")
        print("  3. 再次執行本腳本評估改善程度")
    else:
        print("  1. 檢查 pretrain_data.json 資料品質與數量")
        print("  2. 考慮增加預訓練資料或調整 epoch 數")
        print("  3. 重新執行預訓練後再測試")
    print()


if __name__ == "__main__":
    main()
