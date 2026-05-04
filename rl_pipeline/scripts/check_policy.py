"""
check_policy.py — Planning Agent 快速驗證工具

建議使用完整版（含 14 tasks + 邊界測試 + 信心校準）：
  python rl_pipeline/scripts/check_all.py
"""
import sys
import os
import torch
import json

# 設定路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rl_pipeline.agents.planner.planning_agent import PlanningAgent
from dialogue_state_module.embedding import TextEncoder

def check_policy():
    print(f"{'='*60}")
    print(" RL Planning Agent 決策分析工具 (v7)")
    print(f"{'='*60}")

    # 1. 載入 Agent 與 Encoder
    agent = PlanningAgent(model_path="rl_pipeline/agents/planner/models/planning_agent.pth")
    encoder = TextEncoder()
    
    # 2. 定義真實測試問題 (涵蓋不同意圖層次)
    test_queries = [
        "他的百分等級 (PR值) 是多少？",  # B: 數據分析
        "他在學校會搶別人的玩具，在家可以怎麼練習？", # C: 具體建議
        "關於自閉症的最新醫療診斷規範（DSM-5）是什麼？", # E: 臨床常模/學術
        "請問附近有沒有推薦的早療機構或是補助可以申請？", # F: 外部資源
        "這份報告的受測日期是哪一天？", # A: 概括性問題
        "哈囉，今天辛苦了，妳好嗎？" # L: 社交/閒聊
    ]
    
    # 3. 模擬決策
    for q in test_queries:
        print(f"\n[問題]: {q}")
        
        # 獲取語義特徵 (簡化模擬)
        # 在實際系統中，這會由 StrategyMapper 呼叫 Retrieval 得到
        # 這裡我們直接看 Agent 的 Policy Output
        
        # 構建 20 維特徵 (6語義 + 13任務 + 1領域熵)
        semantic_stats = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        task_idx = 0 # 預設 A (整體)
        entropy = 0.3 # 預設不確定度
        
        # 根據關鍵字精準對應 Task Label
        if "PR" in q or "百分等級" in q or "幾分" in q:
            semantic_stats[0] = 0.9 # Assessment
            task_idx = 1 # B (數據分析)
        elif "玩具" in q or "練習" in q or "教" in q:
            semantic_stats[3] = 0.8 # Suggestion
            task_idx = 2 # C (具體建議)
        elif "診斷" in q or "標準" in q or "DSM" in q:
            semantic_stats[5] = 0.9 # External GPT
            task_idx = 9 # J (學校合作/學術知識 → external_gpt)
        elif "機構" in q or "補助" in q or "申請" in q or "台北" in q or "附近" in q:
            semantic_stats[4] = 0.9 # Community Resources
            task_idx = 7 # H (轉介資源 → community_resources + external_gpt)
        elif "日期" in q or "簽名" in q:
            semantic_stats[1] = 0.7 # Observation (General Data)
            task_idx = 0 # A (概括)
        elif "妳好" in q or "辛苦" in q:
            # 社交閒聊在實際流程中由 retrieval 層早期短路（return []），不走 Planning Agent
            # 這裡用 task=L（後續追蹤）測試模型的最低干擾輸出
            semantic_stats = [0.05] * 6
            task_idx = 11 # L (後續追蹤，非社交；社交閒聊不進 Planning Agent)
            entropy = 0.1
            
        task_onehot = [0.0] * len(agent.task_list)
        if task_idx < len(agent.task_list):
            task_onehot[task_idx] = 1.0

        state = semantic_stats + task_onehot + [entropy]
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        
        with torch.no_grad():
            probs = agent.policy_net(state_tensor).squeeze().tolist()
        
        # 輸出機率分佈
        labels = agent.section_labels # ['assessment', 'observation', 'training', 'suggestion', 'community_resources', 'external_gpt']
        print(f"{'決策區塊':<15} | {'啟動機率':<10}")
        print("-" * 30)
        for label, prob in zip(labels, probs):
            bar = "█" * int(prob * 20)
            print(f"{label:<15} | {prob:>8.2%} {bar}")

if __name__ == "__main__":
    check_policy()
