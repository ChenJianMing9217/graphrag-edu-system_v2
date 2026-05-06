"""
Step 1: 評估 v1/v3 在 turn>=2 子集（去掉首輪 base-rate 灌水）的真實表現。

不重訓、只重新評估。讀:
  - rl_pipeline/dataset/memory_v3_full_signals.jsonl
  - rl_pipeline/agents/memory/models/memory_agent.pth (v1)
  - rl_pipeline/agents/memory/models/memory_agent_v3_19d.pth (v3)

輸出指標：
  - 全 test (turn 1-5)：與既有報告比對
  - turn=1 only：基線（多半全 REFRESH）
  - turn>=2 only：真實 memory 智慧分數
  - turn=2/3/4/5 各別分布

使用：
  python rl_pipeline/scripts/evaluate_turn2plus.py
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rl_pipeline.agents.memory.memory_agent import MemoryAgent

DATA_PATH = ROOT / "rl_pipeline/dataset/memory_v3_full_signals.jsonl"
SEED = 42


def load_test():
    samples = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    rng = random.Random(SEED)
    sessions = sorted({s["session_id"] for s in samples})
    rng.shuffle(sessions)
    n_test = max(1, int(len(sessions) * 0.2))
    test_sids = set(sessions[:n_test])
    return [s for s in samples if s["session_id"] in test_sids]


def merged_state(sample):
    st = {}
    st.update(sample.get("features_7d_legacy", {}))
    st.update(sample.get("features_v2", {}))
    return st


def metrics_binary(y_true, y_pred):
    if len(y_true) == 0:
        return None
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    acc = (tp + tn) / max(1, len(y_true))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return {
        "n": len(y_true),
        "acc": float(acc),
        "P_R": float(prec),
        "R_R": float(rec),
        "F1_R": float(f1),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "support_STAY": int((y_true == 0).sum()),
        "support_REFRESH": int((y_true == 1).sum()),
    }


def evaluate(agent, samples, label):
    y_true = np.array([s["label_idx"] for s in samples], dtype=np.int64)
    y_pred = []
    for s in samples:
        st = merged_state(s)
        r = agent.select_action(st, deterministic=True)
        y_pred.append(r["action_idx"])
    y_pred = np.asarray(y_pred, dtype=np.int64)
    m = metrics_binary(y_true, y_pred)
    return m


def main():
    test = load_test()
    print(f"[Test] total={len(test)}")
    print(f"  turn_id 分布: {sorted(Counter(s['turn_id'] for s in test).items())}")

    print("\n--- Loading v1 ---")
    agent_v1 = MemoryAgent(version="v1")
    print("\n--- Loading v3 ---")
    agent_v3 = MemoryAgent(version="v3")

    subsets = {
        "all_turns":  test,
        "turn_1_only": [s for s in test if s["turn_id"] == 1],
        "turn_>=2":   [s for s in test if s["turn_id"] >= 2],
        "turn_2":     [s for s in test if s["turn_id"] == 2],
        "turn_3":     [s for s in test if s["turn_id"] == 3],
        "turn_4plus": [s for s in test if s["turn_id"] >= 4],
    }

    print()
    print("=" * 80)
    print(f"{'Subset':<14} {'Model':<5} {'n':>4} {'Acc':>7} {'P_R':>7} {'R_R':>7} {'F1_R':>7} "
          f"{'TN':>4} {'FP':>4} {'FN':>4} {'TP':>4}")
    print("-" * 80)
    summary = {}
    for name, subset in subsets.items():
        if not subset:
            continue
        m1 = evaluate(agent_v1, subset, "v1")
        m3 = evaluate(agent_v3, subset, "v3")
        summary[name] = {"v1": m1, "v3": m3}
        print(f"{name:<14} {'v1':<5} {m1['n']:>4} {m1['acc']:>7.4f} {m1['P_R']:>7.4f} "
              f"{m1['R_R']:>7.4f} {m1['F1_R']:>7.4f} {m1['TN']:>4} {m1['FP']:>4} "
              f"{m1['FN']:>4} {m1['TP']:>4}")
        print(f"{name:<14} {'v3':<5} {m3['n']:>4} {m3['acc']:>7.4f} {m3['P_R']:>7.4f} "
              f"{m3['R_R']:>7.4f} {m3['F1_R']:>7.4f} {m3['TN']:>4} {m3['FP']:>4} "
              f"{m3['FN']:>4} {m3['TP']:>4}")
        print()

    out = ROOT / "rl_pipeline/dataset/evaluate_turn2plus.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Out] {out}")

    # Decision hint
    print("\n=== Decision ===")
    real = summary.get("turn_>=2", {}).get("v3", {}).get("acc")
    if real is not None:
        if real >= 0.86:
            print(f"  v3 turn>=2 acc={real:.4f} >= 0.86  → 不需要重訓，直接做 Step 3 (first-turn hard rule)")
        else:
            print(f"  v3 turn>=2 acc={real:.4f} <  0.86  → 建議 Step 2 重訓（過濾 turn=1 訓練資料）")


if __name__ == "__main__":
    main()
