"""
真實 test 樣本對打：v1 (7d) vs v3 (18d) on memory_v3_full_signals.jsonl test set.

直接走 MemoryAgent 完整推論路徑（含 _extract_features + load）→ 確認上線時表現
與離線 train_memory_agent_v3.py 一致。

使用：
  python rl_pipeline/scripts/compare_agents_realtest.py
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
    """把 features_7d_legacy + features_v2 合併成 18-key dict"""
    st = {}
    st.update(sample.get("features_7d_legacy", {}))
    st.update(sample.get("features_v2", {}))
    return st


def metrics(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    acc = (tp + tn) / max(1, len(y_true))
    return {
        "acc": float(acc),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def main():
    test = load_test()
    print(f"[Test] n={len(test)}  REFRESH={sum(1 for s in test if s['label_idx']==1)} "
          f"STAY={sum(1 for s in test if s['label_idx']==0)}")

    # 兩個 agent
    print("\n--- Loading v1 ---")
    agent_v1 = MemoryAgent(version="v1")
    print("\n--- Loading v3 ---")
    agent_v3 = MemoryAgent(version="v3")

    y_true = [s["label_idx"] for s in test]
    pred_v1 = []
    pred_v3 = []
    for s in test:
        st = merged_state(s)
        r1 = agent_v1.select_action(st, deterministic=True)
        r3 = agent_v3.select_action(st, deterministic=True)
        pred_v1.append(r1["action_idx"])
        pred_v3.append(r3["action_idx"])

    m1 = metrics(np.array(y_true), np.array(pred_v1))
    m3 = metrics(np.array(y_true), np.array(pred_v3))

    print()
    print("=" * 60)
    print(f"v1 (7d):  acc={m1['acc']:.4f}  TN={m1['TN']} FP={m1['FP']} "
          f"FN={m1['FN']} TP={m1['TP']}")
    print(f"v3 (18d): acc={m3['acc']:.4f}  TN={m3['TN']} FP={m3['FP']} "
          f"FN={m3['FN']} TP={m3['TP']}")
    print()

    # disagreement
    agree = sum(1 for a, b in zip(pred_v1, pred_v3) if a == b)
    disagree_idx = [i for i, (a, b) in enumerate(zip(pred_v1, pred_v3)) if a != b]
    v3_right = sum(1 for i in disagree_idx if pred_v3[i] == y_true[i])
    v1_right = sum(1 for i in disagree_idx if pred_v1[i] == y_true[i])
    print(f"Agreement   : {agree}/{len(test)}")
    print(f"Disagreement: {len(disagree_idx)}")
    print(f"  v3 對 v1 錯: {v3_right}")
    print(f"  v1 對 v3 錯: {v1_right}")

    # 印 5 筆 v3 改進的 case
    print()
    print("--- Sample v3-improvement cases (5筆) ---")
    LABEL = {0: "STAY", 1: "REFRESH"}
    shown = 0
    for i in disagree_idx:
        if pred_v3[i] == y_true[i] and pred_v1[i] != y_true[i]:
            s = test[i]
            print(f"  [GT={LABEL[y_true[i]]}] turn={s['turn_id']}  v1→{LABEL[pred_v1[i]]}  v3→{LABEL[pred_v3[i]]}")
            print(f"    query: {s['user_query']}")
            print(f"    topic_overlap={s['features_7d_legacy']['topic_overlap']:.3f}  "
                  f"tv={s['features_7d_legacy']['tv_distance']:.3f}  "
                  f"context_sim={s['features_7d_legacy']['context_sim']:.3f}")
            shown += 1
            if shown >= 5:
                break


if __name__ == "__main__":
    main()
