"""
Train Memory Agent v2 (12d features) — first baseline table for thesis Phase 1.

讀: rl_pipeline/dataset/memory_v2_dataset_12d.jsonl
輸出:
  - rl_pipeline/agents/memory/models/memory_agent_v2.pth (12d MLP)
  - rl_pipeline/dataset/memory_v2_results.json (Linear / GBDT / MLP 三模型 metrics)

訓練 3 個 baseline 比較:
  1. Logistic Regression (linear, 顯示「線性可分性」上限)
  2. Gradient Boosting Decision Tree (sklearn, 表格資料強 baseline)
  3. 12d MLP (3 層 FC + dropout, 與線上推論架構一致)

評估:
  - 80/20 random split, fixed seed
  - Stratified by session（同 session 不會同時出現在 train+test，避免洩漏）
  - 報告 accuracy / precision / recall / F1 / confusion matrix

使用：
  python rl_pipeline/scripts/train_memory_agent_v2.py
"""
from __future__ import annotations
import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "rl_pipeline/dataset/memory_v2_dataset_12d.jsonl"
RESULT_PATH = ROOT / "rl_pipeline/dataset/memory_v2_results.json"
MODEL_PATH = ROOT / "rl_pipeline/agents/memory/models/memory_agent_v2.pth"

FEATURE_KEYS_12D = [
    # 8d main
    "prev_top_eq_current_raw_top",
    "prev_top_in_current_top3",
    "ambiguous_followup_score",
    "followup_kw_present",
    "switch_kw_present",
    "tv_distance_raw",
    "task_top1_drop",
    "query_len_norm",
    # 4 attributes
    "domain_kw_present",
    "query_len_chars",
    "has_question_mark",
    "domain_entropy_raw",
]

SEED = 42


def load_dataset() -> List[dict]:
    samples = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def to_xy(samples: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    for s in samples:
        f = s["features_v2"]
        # query_len_chars 是 0~~50+ 的整數，做歸一化避免主導梯度
        feat = []
        for k in FEATURE_KEYS_12D:
            v = float(f.get(k, 0.0))
            if k == "query_len_chars":
                v = min(v / 50.0, 1.0)
            feat.append(v)
        X.append(feat)
        y.append(int(s["label_idx"]))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def session_split(samples: List[dict], test_frac: float = 0.2, seed: int = SEED):
    """以 session_id 做 split，避免同 session 跨 train/test 洩漏。"""
    rng = random.Random(seed)
    sessions = sorted({s["session_id"] for s in samples})
    rng.shuffle(sessions)
    n_test = max(1, int(len(sessions) * test_frac))
    test_sids = set(sessions[:n_test])
    train = [s for s in samples if s["session_id"] not in test_sids]
    test = [s for s in samples if s["session_id"] in test_sids]
    return train, test


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 2):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    acc = (tp + tn) / max(1, len(y_true))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return {
        "accuracy": float(acc),
        "precision_REFRESH": float(prec),
        "recall_REFRESH": float(rec),
        "f1_REFRESH": float(f1),
        "confusion_matrix": {
            "true_STAY_pred_STAY": int(tn),
            "true_STAY_pred_REFRESH": int(fp),
            "true_REFRESH_pred_STAY": int(fn),
            "true_REFRESH_pred_REFRESH": int(tp),
        },
        "support": {"STAY": int((y_true == 0).sum()), "REFRESH": int((y_true == 1).sum())},
    }


# ------------------- Logistic Regression -------------------
def train_logreg(X_tr, y_tr, X_te, y_te) -> Dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    clf.fit(X_tr_s, y_tr)
    y_pred = clf.predict(X_te_s)
    m = metrics_binary(y_te, y_pred)
    coef_dict = {FEATURE_KEYS_12D[i]: float(clf.coef_[0, i]) for i in range(len(FEATURE_KEYS_12D))}
    m["model"] = "LogisticRegression"
    m["coefficients"] = coef_dict
    m["intercept"] = float(clf.intercept_[0])
    return m


# ------------------- Gradient Boosting -------------------
def train_gbdt(X_tr, y_tr, X_te, y_te) -> Dict:
    from sklearn.ensemble import GradientBoostingClassifier

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=SEED
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    m = metrics_binary(y_te, y_pred)
    imp = {FEATURE_KEYS_12D[i]: float(clf.feature_importances_[i]) for i in range(len(FEATURE_KEYS_12D))}
    m["model"] = "GradientBoosting"
    m["feature_importance"] = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
    return m


# ------------------- 12d MLP -------------------
def train_mlp(X_tr, y_tr, X_te, y_te, epochs: int = 150, batch_size: int = 32) -> Dict:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F

    torch.manual_seed(SEED)

    class MLP12d(nn.Module):
        def __init__(self, input_dim=12, hidden_dim=64, output_dim=2, dropout=0.25):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            return self.fc3(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP12d().to(device)

    # 類別不平衡：用 weight balance
    class_counts = Counter(y_tr.tolist())
    n0, n1 = class_counts[0], class_counts[1]
    w = torch.tensor([1.0 / max(1, n0), 1.0 / max(1, n1)], dtype=torch.float32, device=device)
    w = w / w.sum() * 2.0
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)

    n_tr = X_tr_t.shape[0]
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        total_loss = 0.0
        correct = 0
        for i in range(0, n_tr, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_tr_t[idx]
            yb = y_tr_t[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
        if (ep + 1) % 25 == 0:
            print(f"  [MLP] epoch {ep+1}/{epochs}  loss={total_loss/n_tr:.4f}  train_acc={correct/n_tr:.4f}")
        history.append({"epoch": ep + 1, "loss": total_loss / n_tr, "train_acc": correct / n_tr})

    # Test eval
    model.eval()
    with torch.no_grad():
        logits = model(X_te_t)
        y_pred = logits.argmax(dim=1).cpu().numpy()

    m = metrics_binary(y_te, y_pred)
    m["model"] = "MLP_12d"
    m["epochs"] = epochs

    # Save weights
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    m["model_path"] = str(MODEL_PATH)
    return m


def main():
    if not DATA_PATH.exists():
        print(f"[Error] Dataset missing: {DATA_PATH}")
        print("先跑 build_v2_dataset.py")
        sys.exit(1)

    samples = load_dataset()
    print(f"[Load] {len(samples)} samples")
    print(f"       label dist: {Counter(s['label'] for s in samples)}")

    train_samples, test_samples = session_split(samples, test_frac=0.2, seed=SEED)
    print(f"[Split] train={len(train_samples)}  test={len(test_samples)}")
    print(f"        train sessions={len({s['session_id'] for s in train_samples})}")
    print(f"        test  sessions={len({s['session_id'] for s in test_samples})}")
    print(f"        train labels={Counter(s['label'] for s in train_samples)}")
    print(f"        test  labels={Counter(s['label'] for s in test_samples)}")

    X_tr, y_tr = to_xy(train_samples)
    X_te, y_te = to_xy(test_samples)
    print(f"[Shape] X_tr={X_tr.shape}  X_te={X_te.shape}")

    results = {
        "dataset": str(DATA_PATH),
        "feature_keys_12d": FEATURE_KEYS_12D,
        "n_train": len(train_samples),
        "n_test": len(test_samples),
        "train_label_dist": dict(Counter(s["label"] for s in train_samples)),
        "test_label_dist": dict(Counter(s["label"] for s in test_samples)),
        "models": {},
    }

    print("\n[Train] LogisticRegression ...")
    m_lr = train_logreg(X_tr, y_tr, X_te, y_te)
    results["models"]["logreg"] = m_lr
    print(f"  LR test acc={m_lr['accuracy']:.4f}  f1_REFRESH={m_lr['f1_REFRESH']:.4f}")

    print("\n[Train] GradientBoosting ...")
    m_gb = train_gbdt(X_tr, y_tr, X_te, y_te)
    results["models"]["gbdt"] = m_gb
    print(f"  GBDT test acc={m_gb['accuracy']:.4f}  f1_REFRESH={m_gb['f1_REFRESH']:.4f}")

    print("\n[Train] MLP 12d ...")
    m_mlp = train_mlp(X_tr, y_tr, X_te, y_te)
    results["models"]["mlp_12d"] = m_mlp
    print(f"  MLP test acc={m_mlp['accuracy']:.4f}  f1_REFRESH={m_mlp['f1_REFRESH']:.4f}")

    RESULT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Done] {RESULT_PATH.name}")
    print(f"       MLP weights: {MODEL_PATH}")

    # Summary table
    print("\n" + "=" * 60)
    print("Summary (test set):")
    print(f"  {'Model':<22} {'Acc':>7} {'P_R':>7} {'R_R':>7} {'F1_R':>7}")
    for name, m in results["models"].items():
        print(f"  {m['model']:<22} {m['accuracy']:>7.4f} "
              f"{m['precision_REFRESH']:>7.4f} {m['recall_REFRESH']:>7.4f} {m['f1_REFRESH']:>7.4f}")


if __name__ == "__main__":
    main()
