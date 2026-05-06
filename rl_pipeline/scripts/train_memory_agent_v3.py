"""
Train Memory Agent on full-signal dataset (v3).

讀: rl_pipeline/dataset/memory_v3_full_signals.jsonl

訓練 3 組特徵 × 3 種模型 = 9 個對照（論文 Table 1 用）：
  Feature sets:
    A) 7d_legacy: entropy, tv_distance(domain), topic_overlap, context_sim,
                  turn_index_norm, query_len_norm, is_multi_domain
    B) 12d_v2: 前 8 主特徵 + 4 屬性
    C) 19d_combined: A + B 去重（query_len_norm 重複，所以是 7+12 = 19 個欄位）

  Models:
    1) LogisticRegression
    2) GradientBoostingClassifier
    3) MLP (PyTorch)

Session-level split, 80/20, fixed seed.

輸出:
  rl_pipeline/dataset/memory_v3_results.json
  rl_pipeline/agents/memory/models/memory_agent_v3_19d.pth (最強組合 MLP 權重)
"""
from __future__ import annotations
import os
import sys
import json
import random
from pathlib import Path
from collections import Counter, OrderedDict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "rl_pipeline/dataset/memory_v3_full_signals.jsonl"
RESULT_PATH = ROOT / "rl_pipeline/dataset/memory_v3_results.json"
MODEL_PATH_19D = ROOT / "rl_pipeline/agents/memory/models/memory_agent_v3_19d.pth"

SEED = 42

KEYS_7D = [
    "entropy", "tv_distance", "topic_overlap", "context_sim",
    "turn_index_norm", "query_len_norm", "is_multi_domain",
]
KEYS_12D = [
    "prev_top_eq_current_raw_top", "prev_top_in_current_top3",
    "ambiguous_followup_score", "followup_kw_present",
    "switch_kw_present", "tv_distance_raw", "task_top1_drop",
    "query_len_norm",
    "domain_kw_present", "query_len_chars",
    "has_question_mark", "domain_entropy_raw",
]
# 19d combined: 7d 原樣 + 12d 中扣掉 query_len_norm（重複）
KEYS_12D_NO_DUP = [k for k in KEYS_12D if k != "query_len_norm"]
KEYS_19D = KEYS_7D + KEYS_12D_NO_DUP


def load_dataset():
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


def get_feat(s, key):
    if key in s.get("features_7d_legacy", {}):
        return float(s["features_7d_legacy"][key])
    if key in s.get("features_v2", {}):
        v = float(s["features_v2"][key])
        if key == "query_len_chars":
            v = min(v / 50.0, 1.0)
        return v
    return 0.0


def to_xy(samples, feature_keys):
    X, y = [], []
    for s in samples:
        feat = [get_feat(s, k) for k in feature_keys]
        X.append(feat)
        y.append(int(s["label_idx"]))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def confmat(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def metrics_binary(y_true, y_pred):
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


def train_logreg(X_tr, y_tr, X_te, y_te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Xs_tr = sc.fit_transform(X_tr)
    Xs_te = sc.transform(X_te)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    clf.fit(Xs_tr, y_tr)
    pred = clf.predict(Xs_te)
    return metrics_binary(y_te, pred), clf


def train_gbdt(X_tr, y_tr, X_te, y_te):
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=SEED
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    return metrics_binary(y_te, pred), clf


def train_mlp(X_tr, y_tr, X_te, y_te, input_dim, hidden_dim=64, epochs=150,
              batch_size=32, save_path=None):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    torch.manual_seed(SEED)

    class MLP(nn.Module):
        def __init__(self, d, h=hidden_dim, o=2, drop=0.25):
            super().__init__()
            self.fc1 = nn.Linear(d, h)
            self.fc2 = nn.Linear(h, h)
            self.fc3 = nn.Linear(h, o)
            self.drop = nn.Dropout(drop)
        def forward(self, x):
            x = F.relu(self.fc1(x)); x = self.drop(x)
            x = F.relu(self.fc2(x)); x = self.drop(x)
            return self.fc3(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim).to(device)
    counts = Counter(y_tr.tolist())
    n0, n1 = counts[0], counts[1]
    w = torch.tensor([1.0/max(1,n0), 1.0/max(1,n1)], dtype=torch.float32, device=device)
    w = w / w.sum() * 2.0
    crit = nn.CrossEntropyLoss(weight=w)
    opt = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)

    n = X_tr_t.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_tr_t[idx], y_tr_t[idx]
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(X_te_t).argmax(dim=1).cpu().numpy()
    m = metrics_binary(y_te, pred)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        m["model_path"] = str(save_path)
    return m, model


def main():
    if not DATA_PATH.exists():
        print(f"[Error] {DATA_PATH} missing")
        sys.exit(1)

    samples = load_dataset()
    print(f"[Load] {len(samples)} samples")
    train_s, test_s = session_split(samples)
    print(f"[Split] train={len(train_s)} test={len(test_s)}")
    print(f"        train labels={Counter(s['label'] for s in train_s)}")
    print(f"        test  labels={Counter(s['label'] for s in test_s)}")

    feature_sets = OrderedDict([
        ("7d_legacy",   KEYS_7D),
        ("12d_v2",      KEYS_12D),
        ("19d_combined", KEYS_19D),
    ])

    results = {
        "dataset": str(DATA_PATH),
        "n_train": len(train_s),
        "n_test": len(test_s),
        "feature_sets": {k: v for k, v in feature_sets.items()},
        "table": [],
    }

    for set_name, keys in feature_sets.items():
        print(f"\n=== Feature set: {set_name} ({len(keys)}d) ===")
        X_tr, y_tr = to_xy(train_s, keys)
        X_te, y_te = to_xy(test_s, keys)
        print(f"  X_tr={X_tr.shape}  X_te={X_te.shape}")

        # LogReg
        m_lr, clf_lr = train_logreg(X_tr, y_tr, X_te, y_te)
        m_lr["model"] = "LogReg"; m_lr["features"] = set_name
        # 加入 coefficient
        m_lr["coefficients"] = {keys[i]: float(clf_lr.coef_[0, i]) for i in range(len(keys))}
        results["table"].append(m_lr)
        print(f"  LR    acc={m_lr['accuracy']:.4f} f1_R={m_lr['f1_REFRESH']:.4f}")

        # GBDT
        m_gb, clf_gb = train_gbdt(X_tr, y_tr, X_te, y_te)
        m_gb["model"] = "GBDT"; m_gb["features"] = set_name
        m_gb["feature_importance"] = dict(sorted(
            {keys[i]: float(clf_gb.feature_importances_[i]) for i in range(len(keys))}.items(),
            key=lambda kv: -kv[1]
        ))
        results["table"].append(m_gb)
        print(f"  GBDT  acc={m_gb['accuracy']:.4f} f1_R={m_gb['f1_REFRESH']:.4f}")

        # MLP
        save_path = MODEL_PATH_19D if set_name == "19d_combined" else None
        m_mlp, _ = train_mlp(X_tr, y_tr, X_te, y_te, input_dim=len(keys), save_path=save_path)
        m_mlp["model"] = "MLP"; m_mlp["features"] = set_name
        results["table"].append(m_mlp)
        print(f"  MLP   acc={m_mlp['accuracy']:.4f} f1_R={m_mlp['f1_REFRESH']:.4f}")

    RESULT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 最終 9 列對照表
    print()
    print("=" * 78)
    print("Table 1: 3 feature sets x 3 models (test set, n=%d)" % len(test_s))
    print(f"  {'Features':<14} {'Model':<8} {'Acc':>7} {'P_R':>7} {'R_R':>7} {'F1_R':>7}")
    print("  " + "-" * 60)
    for row in results["table"]:
        print(f"  {row['features']:<14} {row['model']:<8} "
              f"{row['accuracy']:>7.4f} {row['precision_REFRESH']:>7.4f} "
              f"{row['recall_REFRESH']:>7.4f} {row['f1_REFRESH']:>7.4f}")
    print()
    print(f"[Out] {RESULT_PATH.name}")
    if MODEL_PATH_19D.exists():
        print(f"[Model] {MODEL_PATH_19D.name}")


if __name__ == "__main__":
    main()
