import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rl_pipeline.agents.memory.memory_agent import MemoryAgent

def pretrain_behavioral_cloning(data_path="rl_pipeline/agents/memory/pretrain_data.json", epochs=100, batch_size=16):
    """
    Performs Behavioral Cloning on MemoryAgent's policy network using the offline dataset.
    This "warms up" the agent before deploying it online for RL.
    """
    print(f"--- Starting Behavioral Cloning for Memory Agent ---")
    
    # 1. Load Data
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"[Info] Loaded {len(data)} training samples.")
    
    # 2. Init Agent
    agent = MemoryAgent(lr=0.005) # slightly higher lr for supervised learning
    device = agent.device
    policy_net = agent.policy_net
    optimizer = optim.Adam(policy_net.parameters(), lr=0.005)
    
    # Standard Classification Loss for Behavioral Cloning
    criterion = nn.CrossEntropyLoss()
    
    # 3. Training Loop
    policy_net.train()
    
    for epoch in range(epochs):
        # Shuffle data
        import random
        random.shuffle(data)
        
        total_loss = 0.0
        correct_preds = 0
        
        # Batch processing
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            
            # Prepare tensors
            states = []
            targets = []
            
            for item in batch:
                st = item["state"]
                features = [
                    float(st.get('entropy', 0.0)),
                    float(st.get('tv_distance', 0.0)),
                    float(st.get('topic_overlap', 0.0)),
                    float(st.get('context_sim', 0.0))
                ]
                states.append(features)
                targets.append(item["action"])
                
            state_tensor = torch.tensor(states, dtype=torch.float32).to(device)
            target_tensor = torch.tensor(targets, dtype=torch.long).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = policy_net(state_tensor)
            
            # Compute loss
            loss = criterion(logits, target_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch)
            
            # Compute accuracy
            preds = torch.argmax(logits, dim=1)
            correct_preds += (preds == target_tensor).sum().item()
            
        avg_loss = total_loss / len(data)
        accuracy = correct_preds / len(data)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")
            
    # 4. Save the Warmed-up Model
    agent.save()
    print(f"--- Behavioral Cloning Complete. Model saved globally. ---")

if __name__ == "__main__":
    pretrain_behavioral_cloning(epochs=150)
