import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import json

class DialoguePolicyNet(nn.Module):
    """
    Input: [entropy, tv_distance, topic_overlap, context_sim, turn_idx_norm, q_len_norm, is_multi]
    Output: Probabilities for [STAY, REFRESH]

    新設計 (Phase B)：CLARIFY 已從 action 變 attribute（見 semantic_flow_module_v2._decide_clarify），
    Memory Agent 專注在 STAY vs REFRESH 二元決策，訓練更穩定，消除 CLARIFY 回饋迴圈。
    """
    def __init__(self, input_dim=7, hidden_dim=32, output_dim=2, dropout=0.2):
        super(DialoguePolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class MemoryAgent:
    """
    Memory Agent using Policy Gradient (REINFORCE) logic for discrete actions.

    支援兩種版本（透過環境變數 MEMORY_AGENT_VERSION 切換）：
      v1（預設）: 7d MLP, hidden=32, dropout=0.2
                  特徵：entropy / tv_distance / topic_overlap / context_sim /
                       turn_index_norm / query_len_norm / is_multi_domain
      v3        : 18d MLP, hidden=64, dropout=0.25  (test acc 0.892)
                  = v1 7d 全部 + v2 11d (扣掉重複的 query_len_norm)

    切換：
      Linux/Mac: export MEMORY_AGENT_VERSION=v3   (回 v1: unset 或 =v1)
      Windows  : set MEMORY_AGENT_VERSION=v3
    """

    KEYS_7D = [
        "entropy", "tv_distance", "topic_overlap", "context_sim",
        "turn_index_norm", "query_len_norm", "is_multi_domain",
    ]
    # 與 train_memory_agent_v3.py KEYS_19D 完全一致（命名沿用，實際 18d）
    KEYS_18D = KEYS_7D + [
        "prev_top_eq_current_raw_top", "prev_top_in_current_top3",
        "ambiguous_followup_score", "followup_kw_present",
        "switch_kw_present", "tv_distance_raw", "task_top1_drop",
        "domain_kw_present", "query_len_chars",
        "has_question_mark", "domain_entropy_raw",
    ]

    def __init__(self, model_path=None, version=None, lr=0.001, gamma=0.95, epsilon=0.1):
        if version is None:
            version = os.environ.get("MEMORY_AGENT_VERSION", "v1").lower()
        self.version = version

        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        if version == "v3":
            self.input_dim = 18
            self.feature_keys = self.KEYS_18D
            self.policy_net = DialoguePolicyNet(
                input_dim=18, hidden_dim=64, output_dim=2, dropout=0.25
            )
            default_path = os.path.join(models_dir, "memory_agent_v3_19d.pth")
        else:
            self.input_dim = 7
            self.feature_keys = self.KEYS_7D
            self.policy_net = DialoguePolicyNet(
                input_dim=7, hidden_dim=32, output_dim=2, dropout=0.2
            )
            default_path = os.path.join(models_dir, "memory_agent.pth")

        # 若呼叫端傳入 model_path 但版本是 v3 → 強制改用 v3 預設路徑（避免維度不符）
        if version == "v3" and model_path and "memory_agent.pth" in str(model_path) \
                and "v3" not in str(model_path):
            model_path = default_path
        self.model_path = model_path or default_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = self.policy_net.to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon

        self.action_space = ["STAY", "REFRESH"]
        self.load()
        print(f"[Memory Agent] active version={self.version}, input_dim={self.input_dim}, model={os.path.basename(self.model_path)}")

    def _extract_features(self, state_dict: dict) -> torch.Tensor:
        """
        v1: 7 keys。v3: 18 keys（v3 多出來的鍵若缺則 0；query_len_chars 做歸一化）。
        state_dict 可同時包含兩版鍵，自動依 self.feature_keys 取。
        """
        feats = []
        for k in self.feature_keys:
            v = state_dict.get(k, 0.0)
            if k == "is_multi_domain":
                v = 1.0 if v else 0.0
            elif k == "query_len_chars":
                # 訓練時做了 v/50 歸一化
                v = min(float(v) / 50.0, 1.0)
            else:
                v = float(v) if v is not None else 0.0
            feats.append(v)
        return torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(self.device)

    def select_action(self, state_dict: dict, deterministic=False) -> dict:
        """
        Selects an Action based on epsilon-greedy or probability distribution.
        Returns a dict with the chosen action string and index.
        """
        state_tensor = self._extract_features(state_dict)
        
        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            probs = F.softmax(logits, dim=1)
            
        if not deterministic and np.random.rand() < self.epsilon:
            # Exploration
            action_idx = np.random.choice(len(self.action_space))
        else:
            # Exploitation
            action_idx = torch.argmax(probs).item()
            
        return {
            "action_idx": action_idx,
            "action_str": self.action_space[action_idx],
            "probs": probs.cpu().numpy().tolist()[0]
        }

    def update(self, memory_buffer: list):
        """
        Updates the policy network using REINFORCE algorithm.
        memory_buffer is a list of tuples: (state_dict, action_idx, reward)
        """
        if not memory_buffer:
            return
            
        self.policy_net.train()
        self.optimizer.zero_grad()
        
        loss = 0
        for state_dict, action_idx, reward in memory_buffer:
            state_tensor = self._extract_features(state_dict)
            logits = self.policy_net(state_tensor)
            
            # Use log_softmax for cross entropy style loss
            log_probs = F.log_softmax(logits, dim=1)
            
            # Select the log probability of the action that was taken
            action_log_prob = log_probs[0, action_idx]
            
            # 標準 Policy Gradient：最小化「負對數機率 * 獎勵」
            # 若獎勵為正，則增加該動作機率；若獎勵為負，則減少該動作機率。
            loss += -action_log_prob * reward
            
        # Average loss over batch
        loss = loss / len(memory_buffer)
        loss.backward()
        # 梯度裁剪：防止 REINFORCE 在極端獎勵樣本下梯度爆炸（loss=-13 的根本原因）
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.policy_net.eval()
        return loss.item()

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), self.model_path)

    def load(self):
        if os.path.exists(self.model_path):
            try:
                self.policy_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.policy_net.eval()
                print(f"[Memory Agent] Loaded model from {self.model_path}")
            except Exception as e:
                print(f"[Memory Agent] Load failed: {e}. Starting fresh.")

if __name__ == "__main__":
    # Test block for 7-dim Memory Agent
    agent = MemoryAgent(model_path="temp_memory_agent.pth")
    test_state = {
        "entropy": 0.3,
        "tv_distance": 0.7,
        "topic_overlap": 0.6,
        "context_sim": 0.8,
        "turn_index_norm": 0.2,
        "query_len_norm": 0.4,
        "is_multi_domain": True,
    }

    res = agent.select_action(test_state)
    print(f"Initial 7-dim Memory Decision: {res['action_str']} (probs: {res['probs']})")

    for _ in range(50):
        agent.update([(test_state, 0, 1.0)])

    res = agent.select_action(test_state)
    print(f"Updated 7-dim Memory Decision: {res['action_str']} (probs: {res['probs']})")

    if os.path.exists("temp_memory_agent.pth"):
        os.remove("temp_memory_agent.pth")
