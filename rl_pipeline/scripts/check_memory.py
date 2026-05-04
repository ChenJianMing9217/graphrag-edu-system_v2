"""
check_memory.py — Memory Agent 決策分析工具（快速版）

測試 Memory Agent 在不同對話情境下的 STAY/REFRESH/CLARIFY 決策是否合理。
用法：python rl_pipeline/scripts/check_memory.py

建議使用完整版：
  python rl_pipeline/scripts/check_all.py
"""
import sys
import os
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from rl_pipeline.agents.memory.memory_agent import MemoryAgent

SCENARIOS = [
    {
        "name": "明確追問（應 STAY）",
        "desc": "低 entropy，TV 距離小，overlap 高 → 繼續上一輪話題",
        "state": {"entropy": 0.15, "tv_distance": 0.08, "topic_overlap": 0.85,
                  "context_sim": 0.88, "turn_index_norm": 0.3, "query_len_norm": 0.5,
                  "is_multi_domain": False, "prev_action_norm": 0.0, "consecutive_stay_count": 0.2},
        "expect": "STAY",
    },
    {
        "name": "明確切換主題（應 REFRESH）",
        "desc": "低 entropy，TV 距離大，overlap 低 → 切換到新話題",
        "state": {"entropy": 0.12, "tv_distance": 0.82, "topic_overlap": 0.08,
                  "context_sim": 0.25, "turn_index_norm": 0.4, "query_len_norm": 0.65,
                  "is_multi_domain": False, "prev_action_norm": 0.0, "consecutive_stay_count": 0.0},
        "expect": "REFRESH",
    },
    {
        "name": "模糊短句（應 CLARIFY）",
        "desc": "高 entropy，TV 距離大，overlap 極低 → 需澄清",
        "state": {"entropy": 0.91, "tv_distance": 0.89, "topic_overlap": 0.04,
                  "context_sim": 0.07, "turn_index_norm": 0.2, "query_len_norm": 0.12,
                  "is_multi_domain": True, "prev_action_norm": 0.0, "consecutive_stay_count": 0.0},
        "expect": "CLARIFY",
    },
    {
        "name": "多領域中等信心（可 STAY 或 CLARIFY）",
        "desc": "中等 entropy，多領域，overlap 中等",
        "state": {"entropy": 0.55, "tv_distance": 0.45, "topic_overlap": 0.42,
                  "context_sim": 0.58, "turn_index_norm": 0.5, "query_len_norm": 0.45,
                  "is_multi_domain": True, "prev_action_norm": 0.0, "consecutive_stay_count": 0.4},
        "expect": "STAY or CLARIFY",
    },
    {
        "name": "連續 STAY 過多（應 REFRESH 或 CLARIFY）",
        "desc": "consecutive_stay_count 高，可能需要打破循環",
        "state": {"entropy": 0.35, "tv_distance": 0.25, "topic_overlap": 0.55,
                  "context_sim": 0.62, "turn_index_norm": 0.8, "query_len_norm": 0.5,
                  "is_multi_domain": False, "prev_action_norm": 0.0, "consecutive_stay_count": 1.0},
        "expect": "REFRESH or CLARIFY",
    },
]

def check_memory():
    print(f"{'='*65}")
    print("  Memory Agent 決策分析工具")
    print(f"{'='*65}")

    agent = MemoryAgent(model_path="rl_pipeline/agents/memory/models/memory_agent.pth")
    agent.policy_net.eval()

    pass_count = 0
    for s in SCENARIOS:
        state = s["state"]
        state_tensor = agent._extract_features(state)

        with torch.no_grad():
            logits = agent.policy_net(state_tensor)
            probs = F.softmax(logits, dim=1).squeeze().tolist()

        action_idx = probs.index(max(probs))
        action_str = agent.action_space[action_idx]
        confidence = max(probs)

        ok = action_str in s["expect"]
        mark = "✅" if ok else "❌"
        if ok:
            pass_count += 1

        print(f"\n{mark} {s['name']}")
        print(f"   說明：{s['desc']}")
        print(f"   預期：{s['expect']}")
        print(f"   決策：{action_str}（信心 {confidence:.1%}）")
        print(f"   分布：", end="")
        for i, (act, p) in enumerate(zip(agent.action_space, probs)):
            bar = "█" * int(p * 15)
            print(f"{act}={p:.2%}{bar}  ", end="")
        print()

    print(f"\n{'='*65}")
    print(f"  結果：{pass_count}/{len(SCENARIOS)} 通過")
    if pass_count == len(SCENARIOS):
        print("  Agent 決策符合預期 ✅")
    elif pass_count >= len(SCENARIOS) * 0.6:
        print("  Agent 決策大致合理，建議繼續訓練 ⚠️")
    else:
        print("  Agent 決策偏差較大，建議重新預訓練後再 RL ❌")
    print(f"{'='*65}")

if __name__ == "__main__":
    check_memory()
