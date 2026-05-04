import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import json

class PlanningPolicyNet(nn.Module):
    """
    A Policy Network for Retrieval Planning.
    Input: [sem_assessment, sem_observation, sem_training, sem_suggestion, sem_community, sem_external] (6 dims)
           + [Task_A ... Task_N] (14 dims one-hot)
           + [domain_entropy] (1 dim)
           Total: 21 dims
    Output: Probabilities for each section (6 dims, using Sigmoid)
    """
    def __init__(self, input_dim=21, hidden_dim=64, output_dim=6, dropout=0.2):
        super(PlanningPolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # Use Sigmoid for multi-label binary decision (on/off for each section)
        probs = torch.sigmoid(self.fc3(x))
        return probs

class PlanningAgent:
    """
    Planning Agent that decides which knowledge sections to fetch.
    """
    def __init__(self, model_path=None, lr=0.005, epsilon=0.1):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/planning_agent.pth")
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # [NEW] 擴充為 6 維動作空間 (標籤更新)
        self.section_labels = ["assessment", "observation", "training", "suggestion", "community_resources", "external_gpt"]
        self.task_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]

        # input_dim = 21 (6 sem + 14 task one-hot + 1 entropy), output_dim = 6
        self.policy_net = PlanningPolicyNet(input_dim=21, output_dim=6).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.epsilon = epsilon
        self.load()

    def _extract_features(self, state_dict: dict) -> torch.Tensor:
        # ... (保持不變) ...
        # 1. Semantic features (6 dims)
        features = [
            float(state_dict.get('sem_assessment', 0.0)),
            float(state_dict.get('sem_observation', 0.0)),
            float(state_dict.get('sem_training', 0.0)),
            float(state_dict.get('sem_suggestion', 0.0)),
            float(state_dict.get('sem_community_resources', 0.0)), # New
            float(state_dict.get('sem_external_gpt', 0.0))         # New
        ]
        
        # 2. Task One-hot（支援多任務：confidence-weighted，總和歸一為 1）
        task_label      = state_dict.get('task_label', 'A')
        secondary_tasks = state_dict.get('secondary_tasks', [])  # list of str
        task_dist       = state_dict.get('task_dist', {})

        all_active = [task_label] + [t for t in secondary_tasks if t != task_label]
        # 取各任務的軟機率（若無 task_dist 則均分）
        raw_weights = [task_dist.get(t, 1.0 / max(len(all_active), 1)) for t in all_active]
        total_w = sum(raw_weights) or 1.0
        norm_weights = [w / total_w for w in raw_weights]

        task_onehot = [0.0] * len(self.task_list)
        for t, w in zip(all_active, norm_weights):
            if t in self.task_list:
                task_onehot[self.task_list.index(t)] = w
        if all(v == 0.0 for v in task_onehot):
            task_onehot[0] = 1.0  # fallback to A
            
        # 3. Uncertainty feature (1 dim)
        entropy = [float(state_dict.get('domain_entropy', 0.0))]
            
        full_features = features + task_onehot + entropy
        return torch.tensor(full_features, dtype=torch.float32).unsqueeze(0).to(self.device)

    def select_sections(self, state_dict: dict, threshold=0.5, deterministic=False) -> dict:
        state_tensor = self._extract_features(state_dict)
        with torch.no_grad():
            probs = self.policy_net(state_tensor).squeeze(0).cpu().numpy()
            
        active_sections = []
        import random
        
        if deterministic:
            # 上線模式：使用硬門檻 (0.5) 確保決策穩定
            for i, section in enumerate(self.section_labels):
                if probs[i] > threshold:
                    active_sections.append(section)
        else:
            # 訓練/採樣模式：使用 Bernoulli 採樣提高探索多樣性
            # 機率為 0.5 代表有 50% 機率選中，解決「冷啟動時決策固定」的問題
            for i, section in enumerate(self.section_labels):
                if random.random() < probs[i]:
                    active_sections.append(section)
        
        # 確保至少有一個（保底機制）
        if not active_sections:
            max_idx = np.argmax(probs)
            active_sections.append(self.section_labels[max_idx])

        # 原有的 15% 強制隨機擾動保留，作為額外保險
        if not deterministic and random.random() < 0.15:
            forced_action = random.choice(self.section_labels)
            if forced_action not in active_sections:
                if len(active_sections) >= 3:
                    active_sections.pop(0) # 移除最舊的
                active_sections.append(forced_action)

        result = {
            "probs": {self.section_labels[i]: float(probs[i]) for i in range(len(self.section_labels))},
            "active": active_sections
        }
        return result

    def update(self, memory_buffer: list, entropy_beta=0.02):
        """
        Updates the policy network with Entropy Regularization.
        """
        if not memory_buffer:
            return
            
        self.policy_net.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        for state_dict, action_probs_mask, reward in memory_buffer:
            state_tensor = self._extract_features(state_dict)
            output_probs = self.policy_net(state_tensor).squeeze(0)
            
            # [移除懲罰] 效率處罰 (Efficiency Penalty) 暫時關閉，鼓勵探索
            num_active = sum(action_probs_mask)
            efficiency_penalty = 0.0 # 0.15 * num_active
            adjusted_reward = reward - efficiency_penalty if reward > 0 else reward
            
            target = torch.tensor(action_probs_mask, dtype=torch.float32).to(self.device)
            
            # 1. 常規 BCE Loss
            if adjusted_reward >= 0:
                bce_loss = F.binary_cross_entropy(output_probs, target, reduction='none')
                policy_loss = (bce_loss * adjusted_reward).mean()
            else:
                # 懲罰項：只對「被選中且拿負分」的動作進行處罰
                target_zeros = torch.zeros_like(target)
                bce_loss = F.binary_cross_entropy(output_probs, target_zeros, reduction='none')
                policy_loss = (bce_loss * target * abs(adjusted_reward)).mean()
            
            # 2. [NEW] Entropy Bonus (防止策略坍縮至全 0)
            # Entropy = -p log p - (1-p) log(1-p)
            eps = 1e-8
            entropy = -(output_probs * torch.log(output_probs + eps) + (1 - output_probs) * torch.log(1 - output_probs + eps))
            entropy_loss = -entropy_beta * entropy.mean() # 減去熵代表增加熵
            
            total_loss += (policy_loss + entropy_loss)
            
        avg_loss = total_loss / len(memory_buffer)
        avg_loss.backward()
        self.optimizer.step()
        
        self.policy_net.eval()
        return avg_loss.item()

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), self.model_path)

    def load(self):
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                # Check for input dimension mismatch (e.g. after adding task "N": 20→21)
                if "fc1.weight" in state_dict:
                    saved_dim = state_dict["fc1.weight"].shape[1]
                    expected_dim = 6 + len(self.task_list) + 1
                    if saved_dim != expected_dim:
                        print(f"[Planning Agent] Dimension mismatch: saved={saved_dim}, expected={expected_dim}. Starting fresh.")
                        return
                self.policy_net.load_state_dict(state_dict)
                self.policy_net.eval()
                print(f"[Planning Agent] Loaded model from {self.model_path}")
            except Exception as e:
                print(f"[Planning Agent] Load failed: {e}. Starting fresh.")

if __name__ == "__main__":
    # Test block for 20-dim Planning Agent
    agent = PlanningAgent(model_path="temp_planning_agent.pth")
    test_state = {
        'sem_assessment': 0.1,
        'sem_observation': 0.8,
        'sem_training': 0.2,
        'sem_suggestion': 0.4,
        'sem_community_resources': 0.1,
        'sem_external_gpt': 0.2,
        'domain_entropy': 0.5,
        'task_label': 'A'
    }
    res = agent.select_sections(test_state)
    print(f"Initial 20-dim Decision: {res}")
    
    # Simulate positive feedback
    mask = [0, 1, 0, 1, 0, 0] 
    for _ in range(50):
        agent.update([(test_state, mask, 1.0)])
    
    res = agent.select_sections(test_state)
    print(f"Updated 20-dim Decision: {res}")
    if os.path.exists("temp_planning_agent.pth"):
        os.remove("temp_planning_agent.pth")
