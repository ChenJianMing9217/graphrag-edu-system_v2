"""
離線測試：7d MemoryAgent vs 12d MLP_v2 在同一份 test 上的對打。

讀:
  - rl_pipeline/dataset/memory_v2_dataset_12d.jsonl  (12d 含 ctx 與 features_v2)
  - rl_pipeline/agents/memory/models/memory_agent.pth  (上線中的 7d)
  - rl_pipeline/agents/memory/models/memory_agent_v2.pth (新訓 12d)

輸出:
  - rl_pipeline/dataset/compare_7d_vs_12d.json  (整體統計)
  - rl_pipeline/dataset/compare_7d_vs_12d_disagreements.jsonl  (兩者意見不同的 case)

注意：
  - 7d MemoryAgent 需要 entropy/tv_distance/topic_overlap/context_sim/turn_index_norm/query_len_norm/is_multi_domain
  - 我們離線資料只有 entropy/tv_distance（task）/query_len_norm
  - topic_overlap / context_sim / turn_index_norm / is_multi_domain 缺 → 用 0 / 推估值代入
  - 因此 7d 預測會比線上實際略差，但這份對打主要看 12d 的相對改善
"""
from __future__ import annotations
import os
import sys
import json
import random
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_PATH = ROOT / "rl_pipeline/dataset/memory_v2_dataset_12d.jsonl"
PATH_7D = ROOT / "rl_pipeline/agents/memory/models/memory_agent.pth"
PATH_12D = ROOT / "rl_pipeline/agents/memory/models/memory_agent_v2.pth"
OUT_SUMMARY = ROOT / "rl_pipeline/dataset/compare_7d_vs_12d.json"
OUT_DISAGREE = ROOT / "rl_pipeline/dataset/compare_7d_vs_12d_disagreements.jsonl"

SEED = 42

FEATURE_KEYS_12D = [
    "prev_top_eq_current_raw_top",
    "prev_top_in_current_top3",
    "ambiguous_followup_score",
    "followup_kw_present",
    "switch_kw_present",
    "tv_distance_raw",
    "task_top1_drop",
    "query_len_norm",
    "domain_kw_present",
    "query_len_chars",
    "has_question_mark",
    "domain_entropy_raw",
]


# ------------------- model defs -------------------
class Net7d(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=32, output_dim=2, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x)); x = self.dropout(x)
        x = F.relu(self.fc2(x)); x = self.dropout(x)
        return self.fc3(x)


class MLP12d(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, output_dim=2, dropout=0.25):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x)); x = self.dropout(x)
        x = F.relu(self.fc2(x)); x = self.dropout(x)
        return self.fc3(x)


def load_samples():
    samples = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def session_split(samples, test_frac=0.2, seed=SEED):
    rng = random.Random(seed)
    sessions = sorted({s["session_id"] for s in samples})
    rng.shuffle(sessions)
    n_test = max(1, int(len(sessions) * test_frac))
    test_sids = set(sessions[:n_test])
    train = [s for s in samples if s["session_id"] not in test_sids]
    test = [s for s in samples if s["session_id"] in test_sids]
    return train, test


def to_x12(samples):
    X = []
    for s in samples:
        f = s["features_v2"]
        feat = []
        for k in FEATURE_KEYS_12D:
            v = float(f.get(k, 0.0))
            if k == "query_len_chars":
                v = min(v / 50.0, 1.0)
            feat.append(v)
        X.append(feat)
    return np.asarray(X, dtype=np.float32)


def to_x7(samples):
    """
    7d 鍵：entropy / tv_distance / topic_overlap / context_sim / turn_index_norm / query_len_norm / is_multi_domain
    我們手上有：domain_entropy / tv_distance(task) / query_len_norm
    缺：topic_overlap / context_sim / turn_index_norm / is_multi_domain
    用零值代入（會讓 7d 表現偏低，這是已知限制）。
    """
    X = []
    for s in samples:
        f = s["features_v2"]
        ctx = s.get("ctx", {})
        # turn_index_norm 用 turn_id/10
        turn_idx_norm = min(float(s.get("turn_id", 1)) / 10.0, 1.0)
        feat = [
            float(f.get("domain_entropy_raw", 0.0)),  # entropy
            float(f.get("tv_distance_raw", 0.0)),     # tv_distance
            0.0,                                      # topic_overlap (缺)
            0.0,                                      # context_sim (缺)
            float(turn_idx_norm),                     # turn_index_norm
            float(f.get("query_len_norm", 0.0)),      # query_len_norm
            0.0,                                      # is_multi_domain (缺)
        ]
        X.append(feat)
    return np.asarray(X, dtype=np.float32)


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(xt)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    return preds, probs


def confmat(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def metrics(y_true, y_pred):
    cm = confmat(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    acc = (tp + tn) / max(1, len(y_true))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return {
        "accuracy": float(acc),
        "precision_REFRESH": float(prec),
        "recall_REFRESH": float(rec),
        "f1_REFRESH": float(f1),
        "cm": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }


def main():
    if not DATA_PATH.exists():
        print(f"[Error] missing {DATA_PATH}; run build_v2_dataset.py first")
        sys.exit(1)
    if not PATH_7D.exists():
        print(f"[Warn] 7d weights missing: {PATH_7D}; will compare against random-init 7d")

    samples = load_samples()
    _, test = session_split(samples)
    y_true = np.array([s["label_idx"] for s in test], dtype=np.int64)
    print(f"[Load] test={len(test)} samples, REFRESH={int((y_true==1).sum())} STAY={int((y_true==0).sum())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 7d
    net7 = Net7d().to(device)
    if PATH_7D.exists():
        try:
            net7.load_state_dict(torch.load(PATH_7D, map_location=device))
            loaded_7d = True
        except Exception as e:
            print(f"[Warn] 7d load failed: {e}")
            loaded_7d = False
    else:
        loaded_7d = False
    X7 = to_x7(test)
    pred7, prob7 = predict(net7, X7, device)
    m7 = metrics(y_true, pred7)
    m7["model"] = "7d_legacy"
    m7["loaded_weights"] = bool(loaded_7d)
    m7["note"] = "topic_overlap/context_sim/is_multi_domain 缺值用 0 代入 → 比線上實際略差"

    # 12d
    net12 = MLP12d().to(device)
    net12.load_state_dict(torch.load(PATH_12D, map_location=device))
    X12 = to_x12(test)
    pred12, prob12 = predict(net12, X12, device)
    m12 = metrics(y_true, pred12)
    m12["model"] = "12d_v2"
    m12["loaded_weights"] = True

    # 對打統計
    agree = (pred7 == pred12).sum()
    disagree = (pred7 != pred12).sum()
    # disagreement 中誰對誰錯
    d_idx = np.where(pred7 != pred12)[0]
    cnt_12d_right_7d_wrong = int(((pred12[d_idx] == y_true[d_idx]) & (pred7[d_idx] != y_true[d_idx])).sum())
    cnt_7d_right_12d_wrong = int(((pred7[d_idx] == y_true[d_idx]) & (pred12[d_idx] != y_true[d_idx])).sum())

    summary = {
        "n_test": len(test),
        "agreement": int(agree),
        "disagreement": int(disagree),
        "d_12d_right_7d_wrong": cnt_12d_right_7d_wrong,
        "d_7d_right_12d_wrong": cnt_7d_right_12d_wrong,
        "models": {"7d": m7, "12d": m12},
    }

    # disagreement detail
    LABEL_NAME = {0: "STAY", 1: "REFRESH"}
    OUT_DISAGREE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_DISAGREE, "w", encoding="utf-8") as fout:
        for i in d_idx:
            s = test[int(i)]
            obj = {
                "session_id": s["session_id"],
                "turn_id": s["turn_id"],
                "user_query": s["user_query"],
                "ground_truth": LABEL_NAME[int(y_true[i])],
                "label_orig": s.get("label_orig"),
                "pred_7d": LABEL_NAME[int(pred7[i])],
                "pred_12d": LABEL_NAME[int(pred12[i])],
                "prob_7d_REFRESH": float(prob7[i, 1]),
                "prob_12d_REFRESH": float(prob12[i, 1]),
                "ctx": s.get("ctx", {}),
                "verdict": (
                    "12d_correct" if pred12[i] == y_true[i] and pred7[i] != y_true[i] else
                    ("7d_correct" if pred7[i] == y_true[i] and pred12[i] != y_true[i] else "both_wrong")
                ),
                "features_v2": s.get("features_v2", {}),
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 印
    print("\n" + "=" * 60)
    print("Summary (test=%d)" % len(test))
    print(f"  {'Model':<14} {'Acc':>7} {'P_R':>7} {'R_R':>7} {'F1_R':>7}")
    print(f"  {'7d_legacy':<14} {m7['accuracy']:>7.4f} {m7['precision_REFRESH']:>7.4f} "
          f"{m7['recall_REFRESH']:>7.4f} {m7['f1_REFRESH']:>7.4f}  (weights={loaded_7d})")
    print(f"  {'12d_v2':<14} {m12['accuracy']:>7.4f} {m12['precision_REFRESH']:>7.4f} "
          f"{m12['recall_REFRESH']:>7.4f} {m12['f1_REFRESH']:>7.4f}")
    print()
    print(f"  Agreement: {agree}/{len(test)} ({agree/len(test):.1%})")
    print(f"  Disagreement: {disagree}")
    print(f"    12d 對、7d 錯: {cnt_12d_right_7d_wrong}")
    print(f"    7d 對、12d 錯: {cnt_7d_right_12d_wrong}")
    print(f"    兩者都錯:     {disagree - cnt_12d_right_7d_wrong - cnt_7d_right_12d_wrong}")
    print()
    print(f"[Out] {OUT_SUMMARY}")
    print(f"[Out] {OUT_DISAGREE}")


if __name__ == "__main__":
    main()
