import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json

class RerankAgent(nn.Module):
    """
    Dirichlet Policy Network for Reranking weights.
    Outputs alpha parameters (>0) for a Dirichlet distribution.
    """
    def __init__(self, input_dim=33, hidden_dim=64, output_dim=3):
        super(RerankAgent, self).__init__()
        torch.manual_seed(42)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Alphas must be positive
        alphas = self.softplus(self.fc3(x)) + 1.0  # Add 1.0 for stability
        return alphas

class RLAgentManager:
    """
    Handles state encoding, prediction, and REINFORCE updates for the RerankAgent.
    """
    DEFAULT_CONTINUOUS = {
        "entropy": 0.5,
        "top_prob": 0.3,
        "context_sim": 0.5,
        "topic_overlap": 0.5,
        "turn_index_norm": 0.0,
    }
    N_CONTINUOUS = len(DEFAULT_CONTINUOUS)

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/rerank_agent.pth")
        self.model_path = model_path
        self.tasks = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "T_meta_query"]
        self.domains = [
            "整體概況",
            "粗大動作", "精細動作", "感覺統合", "口腔動作", 
            "情緒行為與社會適應功能", "吞嚥功能", "口語理解", 
            "口語表達", "說話", "認知功能"
        ]
        self.scopes = ["S_overview", "S_domain", "S_multi_domain"]
        
        self.onehot_dim = len(self.tasks) + len(self.domains) + len(self.scopes)
        self.input_dim = self.onehot_dim + self.N_CONTINUOUS
        self.model = RerankAgent(input_dim=self.input_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, weights_only=True)
                # Check for input dimension mismatch (e.g. after adding task "N": 33→34)
                if "fc1.weight" in state_dict:
                    saved_dim = state_dict["fc1.weight"].shape[1]
                    if saved_dim != self.input_dim:
                        print(f"[RLAgent] Dimension mismatch: saved={saved_dim}, expected={self.input_dim}. Using fresh model.")
                    else:
                        self.model.load_state_dict(state_dict)
                        print(f"[RLAgent] Loaded model from {self.model_path}")
                else:
                    self.model.load_state_dict(state_dict)
                    print(f"[RLAgent] Loaded model from {self.model_path}")
            except Exception as e:
                print(f"[RLAgent] Failed to load model ({e}), using fresh model")

    def encode_state(self, task, domain, scope, continuous=None):
        task_vec = [1 if t == task else 0 for t in self.tasks]
        domain_vec = [1 if d == domain else 0 for d in self.domains]
        scope_vec = [1 if s == scope else 0 for s in self.scopes]
        
        if sum(task_vec) == 0: task_vec[0] = 1
        if sum(domain_vec) == 0: domain_vec[0] = 1
        if sum(scope_vec) == 0: scope_vec[0] = 1
        
        cont = continuous or {}
        cont_vec = [
            float(cont.get("entropy", self.DEFAULT_CONTINUOUS["entropy"])),
            float(cont.get("top_prob", self.DEFAULT_CONTINUOUS["top_prob"])),
            float(cont.get("context_sim", self.DEFAULT_CONTINUOUS["context_sim"])),
            float(cont.get("topic_overlap", self.DEFAULT_CONTINUOUS["topic_overlap"])),
            min(float(cont.get("turn_index_norm", self.DEFAULT_CONTINUOUS["turn_index_norm"])), 1.0),
        ]
        return torch.FloatTensor(task_vec + domain_vec + scope_vec + cont_vec)

    def select_weights(self, task, domain, scope, continuous=None, deterministic=False):
        """
        Selects weights by sampling from Dirichlet distribution.
        """
        self.model.eval()
        state_tensor = self.encode_state(task, domain, scope, continuous).unsqueeze(0)
        with torch.no_grad():
            alphas = self.model(state_tensor)
            
        if deterministic:
            # Return mean
            weights = alphas / alphas.sum()
            return weights.squeeze().tolist()
        else:
            # Sample from Dirichlet
            from torch.distributions import Dirichlet
            dist = Dirichlet(alphas)
            weights = dist.sample()
            return weights.squeeze().tolist()

    def predict_weights(self, task, domain, scope, continuous=None):
        """ Alias for backward compatibility — uses stochastic sampling (deterministic=False) for online exploration """
        return self.select_weights(task, domain, scope, continuous, deterministic=False)

    def update(self, memory_buffer: list):
        """
        Update the model using REINFORCE on the continuous weight space.
        Memory buffer: list of (state_tuple, sampled_weights, reward, continuous_stats)
        """
        if not memory_buffer: return
        
        self.model.train()
        from torch.distributions import Dirichlet
        
        total_loss = 0
        for (task, domain, scope), sampled_w, reward, cont in memory_buffer:
            state_tensor = self.encode_state(task, domain, scope, cont).unsqueeze(0)
            target_weights = torch.FloatTensor(sampled_w).unsqueeze(0)
            
            alphas = self.model(state_tensor)
            dist = Dirichlet(alphas)
            
            # REINFORCE: loss = -log_prob * reward
            log_prob = dist.log_prob(target_weights)
            loss = -log_prob * (reward - 0.5) # Use 0.5 as baseline
            
            total_loss += loss.mean()
            
        avg_loss = total_loss / len(memory_buffer)
        self.optimizer.zero_grad()
        avg_loss.backward()
        self.optimizer.step()
        return avg_loss.item()

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)

if __name__ == "__main__":
    # Quick test for Dirichlet Policy Gradient
    manager = RLAgentManager()
    cont = {"entropy": 0.3, "top_prob": 0.7, "context_sim": 0.8, "topic_overlap": 0.6, "turn_index_norm": 0.2}
    
    # 預測 (Deterministic Mean)
    w_init = manager.predict_weights("B", "精細動作", "S_domain", continuous=cont)
    print(f"Initial Weights (Mean): {w_init}")
    
    # 模擬採樣訓練
    # 目標：學習較高的 Structural Weight [0.1, 0.8, 0.1]
    buffer = []
    for _ in range(100):
        # 採樣
        sampled_w = manager.select_weights("B", "精細動作", "S_domain", continuous=cont, deterministic=False)
        # 給予獎勵 (越接近 [0, 1, 0] 獎勵越高)
        reward = 1.0 if sampled_w[1] > 0.6 else 0.0
        buffer.append((("B", "精細動作", "S_domain"), sampled_w, reward, cont))
    
    loss = manager.update(buffer)
    print(f"Training Loss: {loss:.4f}")
    
    w_final = manager.predict_weights("B", "精細動作", "S_domain", continuous=cont)
    print(f"Updated Weights (Mean): {w_final}")
