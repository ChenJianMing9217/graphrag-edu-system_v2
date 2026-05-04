"""
pretrain_agents.py — Memory Agent + Planning Agent 行為克隆預訓練腳本

目的：
  在冷啟動前給兩個 Agent 一個合理的初始策略，
  避免 RL 訓練初期用隨機決策傷害使用者體驗。

資料來源（優先順序）：
  1. sft_dataset_v4_final.jsonl — 真實對話記錄，包含完整 DST 特徵與人工標註
  2. 合成資料 — 若 SFT 資料集不存在，回退到手寫規則 + 隨機生成

方法：Behavioral Cloning（監督式學習）

用法：
  python rl_pipeline/scripts/pretrain_agents.py
  python rl_pipeline/scripts/pretrain_agents.py --use-synthetic  # 強制使用合成資料

執行完畢後再跑：
  1. python rl_pipeline/scripts/auto_query_bot.py  （收集資料）
  2. python rl_pipeline/scripts/unified_train_db.py  （RL 訓練）
"""

import os
import sys
import json
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rl_pipeline.agents.memory.memory_agent import MemoryAgent
from rl_pipeline.agents.planner.planning_agent import PlanningAgent

# ============================================================
# 共用常數
# ============================================================
SFT_DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "../dataset/sft_dataset_v4_final.jsonl"
)

SECTION_LABELS = ["assessment", "observation", "training", "suggestion",
                  "community_resources", "external_gpt"]

# 新設計 (Phase B)：Memory Agent 2 分類，CLARIFY 已移至 clarify_type 屬性
MEMORY_ACTION_MAP = {"STAY": 0, "REFRESH": 1}


def _relabel_clarify(record: dict) -> int:
    """
    舊 SFT 中的 CLARIFY 樣本重標（保守版）。

    修正背景：
      初版規則 (tv > 0.6 OR overlap < 0.3 → REFRESH) 過於激進，
      導致中間地帶樣本被過度標為 REFRESH，STAY recall 降低 8%。

    保守規則：
      - T0 → REFRESH（新話題）
      - tv > 0.75 AND overlap < 0.3 → REFRESH（同時滿足才算明顯切換）
      - 其他 → STAY（保留「話題延續中模糊」的預設判斷）

    設計理念：CLARIFY 樣本本質上「還在延續但意圖不確定」，
    用 STAY 作為預設更符合其語意，只有極端切換訊號才改標 REFRESH。
    """
    rm = record.get("retrieval_metadata", {})
    tv = float(rm.get("tv_distance", 0.5))
    overlap = float(rm.get("topic_overlap", 0.5))
    turn_idx = int(record.get("turn_index", 0))

    # T0 通常是新話題 → REFRESH
    if turn_idx == 0:
        return 1  # REFRESH

    # 雙重門檻：tv 極高且 overlap 極低才算明顯切換
    if tv > 0.75 and overlap < 0.30:
        return 1  # REFRESH

    # 其他都保守標 STAY
    return 0  # STAY

# ============================================================
# Planning Agent 預訓練：Task → Section 標準對照表（合成資料用）
# sections: [assessment, observation, training, suggestion, community_resources, external_gpt]
# ============================================================
TASK_SECTION_MAP = {
    "A": [1, 1, 0, 0, 0, 0],  # 報告總覽：看評量 + 觀察記錄
    "B": [1, 1, 0, 0, 0, 0],  # 分數解讀：評量數據 + 臨床觀察
    "C": [0, 1, 0, 1, 0, 0],  # 臨床觀察：觀察 + 具體建議
    "D": [1, 1, 0, 1, 0, 0],  # 能力剖面：評量 + 觀察 + 建議
    "E": [0, 0, 1, 1, 0, 0],  # 在家訓練：訓練方式 + 建議
    "F": [0, 0, 1, 1, 0, 0],  # 融入作息：訓練方式 + 建議
    "G": [1, 0, 0, 1, 0, 0],  # 早療追蹤：評量結果 + 建議
    "H": [0, 0, 0, 0, 1, 1],  # 轉介資源：社區資源 + 外部通用知識（不拉報告）
    "I": [0, 0, 0, 0, 0, 1],  # 隱私安全：外部通用知識（不需要報告內容，需要隱私指引）
    "J": [0, 0, 1, 1, 0, 1],  # 學校合作：訓練策略 + 建議 + 外部通用知識
    "K": [0, 0, 0, 0, 1, 1],  # 補助福利：社區資源 + 外部通用知識（不拉報告）
    "L": [1, 1, 0, 0, 0, 0],  # 後續追蹤：再評估 + 觀察
    "M": [0, 0, 0, 1, 1, 1],  # 情緒支持：建議 + 社區資源 + 外部通用知識
    "N": [1, 1, 0, 0, 0, 0],  # 進步查詢：兩份評量比對
}


# ============================================================
# 從 SFT 資料集載入真實對話資料
# ============================================================

def _load_sft_records(sft_path: str = None) -> list:
    """載入 SFT JSONL，回傳 list of dict。"""
    path = sft_path or SFT_DATASET_PATH
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_memory_data_from_sft(sft_path: str = None) -> list:
    """
    從 SFT 資料集提取 Memory Agent 訓練資料（7 維特徵 + 2 分類 action label）。

    新設計 (Phase B)：
      - Memory Agent 為 2 分類 {STAY, REFRESH}
      - 舊 SFT 中的 CLARIFY 樣本透過 _relabel_clarify 依實際 tv/overlap 分配到 STAY 或 REFRESH
      - 移除 CLARIFY oversampling 和 boundary augmentation
    """
    records = _load_sft_records(sft_path)

    # 按 session 分組，每組按 turn_index 排序
    sessions = defaultdict(list)
    for r in records:
        sessions[r["session_id"]].append(r)
    for sid in sessions:
        sessions[sid].sort(key=lambda x: x["turn_index"])

    data = []
    relabel_count = 0
    for sid, turns in sessions.items():
        for r in turns:
            rm = r["retrieval_metadata"]
            sem = rm.get("semantic_section_scores", {})
            query = r.get("user_query", "")
            action_str = r["memory_action"]

            # 舊 CLARIFY 樣本：用 _relabel_clarify 重新分類
            if action_str == "CLARIFY":
                action_idx = _relabel_clarify(r)
                relabel_count += 1
            else:
                action_idx = MEMORY_ACTION_MAP.get(action_str, 0)

            q_len_norm = min(len(query) / 50.0, 1.0)
            entropy = float(sem.get("domain_entropy", 0.5))

            state = {
                "entropy":                entropy,
                "tv_distance":            float(rm.get("tv_distance", 0.0)),
                "topic_overlap":          float(rm.get("topic_overlap", 0.0)),
                "context_sim":            float(rm.get("context_sim", 0.5)),
                "turn_index_norm":        min(float(r["turn_index"]) / 10.0, 1.0),
                "query_len_norm":         q_len_norm,
                "is_multi_domain":        bool(entropy > 0.65),
            }
            data.append((state, action_idx))

    if relabel_count > 0:
        print(f"  [Relabel] 舊 CLARIFY 樣本 {relabel_count} 筆已重標為 STAY/REFRESH")

    # ── 邊界增強：STAY vs REFRESH 對比樣本 ──────────────────
    # 三層設計（修正版）：
    #   1. 強 STAY：overlap 高 / tv 低（主題明確延續）
    #   2. 中間地帶 STAY：tv 0.3-0.5 + overlap 0.3-0.6（話題漂移但延續）★修正重點
    #   3. 強 REFRESH：tv 高 / overlap 低（明顯切換）
    boundary_samples = []
    n_boundary = 30

    for _ in range(n_boundary):
        turn_norm = round(random.uniform(0.1, 0.7), 3)

        # 強 STAY 樣本：主題延續信號強
        stay_strong = {
            "entropy":         round(random.uniform(0.10, 0.50), 3),
            "tv_distance":     round(random.uniform(0.05, 0.30), 3),
            "topic_overlap":   round(random.uniform(0.55, 0.95), 3),
            "context_sim":     round(random.uniform(0.55, 0.90), 3),
            "turn_index_norm": turn_norm,
            "query_len_norm":  round(random.uniform(0.20, 0.70), 3),
            "is_multi_domain": False,
        }
        boundary_samples.append((stay_strong, 0))  # STAY

        # 【新增】中間地帶 STAY：話題漂移但仍延續（修正 STAY recall 下降問題）
        # 對應「家裡怎麼練習」、「影響到什麼日常」等 task drift 情境
        stay_mid = {
            "entropy":         round(random.uniform(0.20, 0.60), 3),
            "tv_distance":     round(random.uniform(0.30, 0.55), 3),  # 中等 tv
            "topic_overlap":   round(random.uniform(0.35, 0.65), 3),  # 中等 overlap
            "context_sim":     round(random.uniform(0.35, 0.70), 3),  # 中等 ctx
            "turn_index_norm": round(random.uniform(0.15, 0.60), 3),  # 非 T0
            "query_len_norm":  round(random.uniform(0.25, 0.75), 3),
            "is_multi_domain": random.choice([False, False, True]),
        }
        boundary_samples.append((stay_mid, 0))  # STAY

        # 強 REFRESH 樣本：主題切換信號強
        refresh_state = {
            "entropy":         round(random.uniform(0.10, 0.45), 3),
            "tv_distance":     round(random.uniform(0.65, 0.95), 3),  # 提高下限
            "topic_overlap":   round(random.uniform(0.05, 0.25), 3),  # 降低上限
            "context_sim":     round(random.uniform(0.05, 0.30), 3),
            "turn_index_norm": turn_norm,
            "query_len_norm":  round(random.uniform(0.35, 0.85), 3),
            "is_multi_domain": False,
        }
        boundary_samples.append((refresh_state, 1))  # REFRESH

    # ── T0 樣本增強 ──────────────────────────────────────────
    # 對應「測驗精神狀況」、「家人意見不一」等 T0 明確意圖情境
    # 預設 T0 + query 明確（非極短）→ REFRESH
    n_t0 = 20
    for _ in range(n_t0):
        t0_state = {
            "entropy":         round(random.uniform(0.15, 0.70), 3),
            "tv_distance":     0.5,   # T0 時 tv 預設中性
            "topic_overlap":   0.5,   # T0 時 overlap 預設中性
            "context_sim":     0.5,   # T0 時 ctx_sim 預設中性（neutral_first_turn）
            "turn_index_norm": 0.0,   # T0
            "query_len_norm":  round(random.uniform(0.25, 0.90), 3),  # 正常長度 query
            "is_multi_domain": random.choice([False, True]),
        }
        boundary_samples.append((t0_state, 1))  # REFRESH

    data.extend(boundary_samples)
    return data


def load_planning_data_from_sft(sft_path: str = None) -> list:
    """
    從 SFT 資料集提取 Planning Agent 訓練資料（21 維特徵 + 6 維 section mask）。

    特徵組成：
      - 6 維 semantic section scores（真實 cosine similarity）
      - 14 維 task one-hot（可含 task_dist soft weight）
      - 1 維 domain_entropy
    標籤：active_sections → binary mask
    """
    records = _load_sft_records(sft_path)
    data = []

    for r in records:
        rm = r["retrieval_metadata"]
        sem = rm.get("semantic_section_scores", {})
        task_label = sem.get("task_label", rm.get("task_pred", "A"))
        task_dist = rm.get("task_dist", {})

        state = {
            "sem_assessment":           float(sem.get("assessment", 0.0)),
            "sem_observation":          float(sem.get("observation", 0.0)),
            "sem_training":             float(sem.get("training", 0.0)),
            "sem_suggestion":           float(sem.get("suggestion", 0.0)),
            "sem_community_resources":  float(sem.get("community_resources", 0.0)),
            "sem_external_gpt":         float(sem.get("external_gpt", 0.0)),
            "domain_entropy":           float(sem.get("domain_entropy", 0.0)),
            "task_label":               task_label,
            "task_dist":                task_dist,
        }

        # active_sections → binary mask
        active = set(r.get("active_sections", []))
        mask = [1 if s in active else 0 for s in SECTION_LABELS]

        # 至少要有一個 section 被啟用
        if sum(mask) == 0:
            mask[0] = 1  # fallback: assessment

        data.append((state, mask, 1.0))  # reward=1.0（正確行為）

    return data


def generate_planning_data_synthetic(samples_per_task: int = 80) -> list:
    """
    （Fallback）從 TASK_SECTION_MAP 生成合成訓練資料。
    僅在 SFT 資料集不可用時使用。
    """
    data = []
    section_keys = SECTION_LABELS

    for task, mask in TASK_SECTION_MAP.items():
        for _ in range(samples_per_task):
            sem = {}
            for i, key in enumerate(section_keys):
                if mask[i] == 1:
                    sem[f"sem_{key}"] = round(random.uniform(0.45, 0.82), 3)
                else:
                    sem[f"sem_{key}"] = round(random.uniform(0.12, 0.52), 3)
                sem[f"sem_{key}"] = float(np.clip(sem[f"sem_{key}"] + np.random.normal(0, 0.03), 0.0, 1.0))

            entropy = round(random.uniform(0.1, 0.9), 3)
            state = {**sem, "domain_entropy": entropy, "task_label": task}
            data.append((state, mask, 1.0))

    # 多任務合成資料
    multi_task_pairs = [
        ("D", "A"), ("E", "J"), ("H", "K"), ("G", "L"), ("B", "N"),
        ("C", "D"), ("F", "M"), ("G", "H"), ("J", "K"), ("L", "N"),
    ]
    for (t1, t2) in multi_task_pairs:
        mask1 = TASK_SECTION_MAP[t1]
        mask2 = TASK_SECTION_MAP[t2]
        merged = [max(m1, m2) for m1, m2 in zip(mask1, mask2)]
        for _ in range(samples_per_task // 5):
            sem = {}
            for i, key in enumerate(section_keys):
                if merged[i] == 1:
                    sem[f"sem_{key}"] = round(random.uniform(0.45, 0.82), 3)
                else:
                    sem[f"sem_{key}"] = round(random.uniform(0.12, 0.52), 3)
                sem[f"sem_{key}"] = float(np.clip(sem[f"sem_{key}"] + np.random.normal(0, 0.03), 0.0, 1.0))
            state = {
                **sem,
                "domain_entropy": round(random.uniform(0.1, 0.9), 3),
                "task_label": t1,
                "secondary_tasks": [t2],
                "task_dist": {t1: round(random.uniform(0.55, 0.75), 2),
                              t2: round(random.uniform(0.25, 0.45), 2)},
            }
            data.append((state, merged, 1.0))

    random.shuffle(data)
    return data


def pretrain_planning_agent(epochs: int = 40, samples_per_task: int = 80, use_sft: bool = True):
    """Planning Agent 行為克隆預訓練（多標籤 BCE）"""
    print("\n" + "=" * 55)
    print("  [Planning Agent] 行為克隆預訓練開始")
    print("=" * 55)

    agent = PlanningAgent(lr=0.005)
    device = agent.device
    net = agent.policy_net
    optimizer = optim.Adam(net.parameters(), lr=0.005, weight_decay=1e-4)

    # 資料來源選擇：SFT 模式 = 真實資料 + 合成 task-prior 混合
    if use_sft and os.path.exists(SFT_DATASET_PATH):
        sft_data = load_planning_data_from_sft()
        # 混入合成 task-prior 資料，強化 task → section 的硬性映射
        # 解決 SFT 資料中 suggestion / external_gpt 該開沒開的問題
        synthetic_prior = generate_planning_data_synthetic(samples_per_task=30)
        data = sft_data + synthetic_prior
        print(f"  [SFT+Prior] 真實={len(sft_data)} + 合成 task-prior={len(synthetic_prior)} = {len(data)} 筆")
        epochs = max(epochs, 60)
    else:
        data = generate_planning_data_synthetic(samples_per_task)
        print(f"  [Synthetic] 生成合成樣本：{len(data)} 筆")

    net.train()
    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0.0
        correct_bits = 0
        total_bits = 0

        for state_dict, mask, _ in data:
            state_tensor = agent._extract_features(state_dict)
            target = torch.tensor(mask, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            probs = net(state_tensor).squeeze(0)
            # Label smoothing：1→0.88, 0→0.06，防止 sigmoid 趨近 1.0/0.0 飽和
            smooth = 0.12
            soft_target = target * (1.0 - smooth) + smooth / 2.0
            loss = F.binary_cross_entropy(probs, soft_target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (probs.detach() > 0.5).float()
            correct_bits += (preds == target).sum().item()
            total_bits += len(mask)

        if (epoch + 1) % 10 == 0:
            bit_acc = correct_bits / total_bits
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {total_loss/len(data):.4f} | Bit-Acc: {bit_acc:.4f}")

    net.eval()
    agent.save()
    print(f"  模型已儲存：{agent.model_path}")


# ============================================================
# Memory Agent 預訓練：從 pretrain_data.json 補齊 9 維特徵（Fallback）
# ============================================================

def load_memory_pretrain_data_legacy(data_path: str) -> list:
    """
    （Fallback）讀取既有的 pretrain_data.json（4 維），補齊 9 維特徵後回傳。
    僅在 SFT 資料集不可用時使用。
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = []
    for item in raw:
        st = item["state"]
        action = int(item["action"])  # 0=STAY, 1=REFRESH, 2=CLARIFY

        # 舊 legacy 檔案 action=2 (CLARIFY) → 依 tv/overlap 降級為 STAY/REFRESH
        if action == 2:
            tv = float(st.get("tv_distance", 0.5))
            overlap = float(st.get("topic_overlap", 0.5))
            action = 1 if (tv > 0.6 or overlap < 0.3) else 0
            q_len_norm = round(random.uniform(0.1, 0.3), 3)
            turn_norm = round(random.uniform(0.1, 0.5), 3)
        elif action == 1:
            q_len_norm = round(random.uniform(0.4, 0.8), 3)
            turn_norm = round(random.uniform(0.2, 0.7), 3)
        else:
            q_len_norm = round(random.uniform(0.3, 0.7), 3)
            turn_norm = round(random.uniform(0.2, 0.6), 3)

        entropy = float(st.get("entropy", 0.5))
        full_state = {
            "entropy":               float(st.get("entropy", 0.5)),
            "tv_distance":           float(st.get("tv_distance", 0.5)),
            "topic_overlap":         float(st.get("topic_overlap", 0.5)),
            "context_sim":           float(st.get("context_sim", 0.5)),
            "turn_index_norm":       turn_norm,
            "query_len_norm":        q_len_norm,
            "is_multi_domain":       bool(entropy > 0.65),
        }
        data.append((full_state, action))

    return data


def pretrain_memory_agent(epochs: int = 80, use_sft: bool = True):
    """Memory Agent 行為克隆預訓練（CrossEntropy + class weight）"""
    print("\n" + "=" * 55)
    print("  [Memory Agent] 行為克隆預訓練開始")
    print("=" * 55)

    # 資料來源選擇
    if use_sft and os.path.exists(SFT_DATASET_PATH):
        data = load_memory_data_from_sft()
        from collections import Counter
        action_counts = Counter(a for _, a in data)
        print(f"  [SFT] 載入真實資料（2 分類）：{len(data)} 筆")
        print(f"         STAY={action_counts.get(0, 0)}, REFRESH={action_counts.get(1, 0)}")
    else:
        data_path = os.path.join(
            os.path.dirname(__file__), "../agents/memory/pretrain_data.json"
        )
        if not os.path.exists(data_path):
            print(f"  ⚠️  找不到任何訓練資料，跳過 Memory Agent 預訓練。")
            return
        data = load_memory_pretrain_data_legacy(data_path)
        print(f"  [Legacy] 載入手寫樣本：{len(data)} 筆")

    agent = MemoryAgent(lr=0.005)
    device = agent.device
    net = agent.policy_net
    optimizer = optim.Adam(net.parameters(), lr=0.005, weight_decay=1e-4)

    # 計算 class weight（2 分類）
    from collections import Counter
    action_counts = Counter(a for _, a in data)
    total = len(data)
    n_classes = 2
    weights = []
    for c in range(n_classes):
        cnt = action_counts.get(c, 1)
        weights.append(total / (n_classes * cnt))
    class_weight = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"  Class weights: STAY={weights[0]:.2f}, REFRESH={weights[1]:.2f}")

    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.1)

    net.train()
    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0.0
        correct = 0
        per_class_correct = [0, 0]
        per_class_total = [0, 0]

        for state_dict, action_idx in data:
            state_tensor = agent._extract_features(state_dict)
            target = torch.tensor([action_idx], dtype=torch.long).to(device)

            optimizer.zero_grad()
            logits = net(state_tensor)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1).item()
            correct += int(pred == action_idx)
            per_class_total[action_idx] += 1
            per_class_correct[action_idx] += int(pred == action_idx)

        if (epoch + 1) % 20 == 0:
            acc = correct / len(data)
            cls_acc = []
            for c, name in enumerate(["STAY", "REFRESH"]):
                ca = per_class_correct[c] / max(per_class_total[c], 1)
                cls_acc.append(f"{name}={ca:.3f}")
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {total_loss/len(data):.4f} | "
                  f"Acc: {acc:.4f} | {', '.join(cls_acc)}")

    net.eval()
    agent.save()
    print(f"  模型已儲存：{agent.model_path}")


# ============================================================
# 主程式
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pre-train Planning + Memory Agents via Behavioral Cloning")
    parser.add_argument("--skip-planning", action="store_true", help="跳過 Planning Agent 預訓練")
    parser.add_argument("--skip-memory",   action="store_true", help="跳過 Memory Agent 預訓練")
    parser.add_argument("--epochs-planning", type=int, default=40,  help="Planning Agent 訓練 epoch 數（預設 40）")
    parser.add_argument("--epochs-memory",   type=int, default=80,  help="Memory Agent 訓練 epoch 數（預設 80）")
    parser.add_argument("--samples-per-task", type=int, default=80, help="每個 task 生成的樣本數（僅合成模式，預設 80）")
    parser.add_argument("--use-synthetic", action="store_true",
                        help="強制使用合成資料（忽略 SFT 資料集）")
    args = parser.parse_args()

    use_sft = not args.use_synthetic

    if not args.skip_planning:
        pretrain_planning_agent(
            epochs=args.epochs_planning,
            samples_per_task=args.samples_per_task,
            use_sft=use_sft,
        )

    if not args.skip_memory:
        pretrain_memory_agent(epochs=args.epochs_memory, use_sft=use_sft)

    print("\n[DONE] 預訓練完成。接下來：")
    print("    1. python rl_pipeline/scripts/auto_query_bot.py")
    print("    2. python rl_pipeline/scripts/unified_train_db.py")
