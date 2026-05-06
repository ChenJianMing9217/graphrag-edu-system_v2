"""
本地 smoke test: 確認 MEMORY_AGENT_VERSION env var 切換能正常載入 v1 / v3。

不啟動完整 DialogueManager，只測試 MemoryAgent 本身（避免 Neo4j / 其他依賴）。

使用：
  # 預設 v1
  python rl_pipeline/scripts/smoke_test_memory_v3.py

  # 切 v3 (Linux/Mac)
  MEMORY_AGENT_VERSION=v3 python rl_pipeline/scripts/smoke_test_memory_v3.py

  # 切 v3 (Windows PowerShell)
  $env:MEMORY_AGENT_VERSION='v3'; python rl_pipeline/scripts/smoke_test_memory_v3.py

  # 切回 v1 (Windows PowerShell)
  Remove-Item Env:MEMORY_AGENT_VERSION; python rl_pipeline/scripts/smoke_test_memory_v3.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rl_pipeline.agents.memory.memory_agent import MemoryAgent

# 模擬一個 18-key state（v1 會自動忽略多餘鍵）
STATE_18D = {
    # 7d 基礎
    "entropy": 0.65,
    "tv_distance": 0.45,
    "topic_overlap": 0.30,
    "context_sim": 0.55,
    "turn_index_norm": 0.2,
    "query_len_norm": 0.3,
    "is_multi_domain": False,
    # v2 11d 補充
    "prev_top_eq_current_raw_top": 0,
    "prev_top_in_current_top3": 1,
    "ambiguous_followup_score": 0.85,
    "followup_kw_present": 1,
    "switch_kw_present": 0,
    "tv_distance_raw": 0.40,
    "task_top1_drop": 0.30,
    "domain_kw_present": 0,
    "query_len_chars": 8,
    "has_question_mark": 0,
    "domain_entropy_raw": 0.65,
}

# 三個典型場景
SCENARIOS = [
    ("明確新主題（high tv, switch kw 觸發）", {
        **STATE_18D,
        "tv_distance": 0.85, "tv_distance_raw": 0.85,
        "topic_overlap": 0.10,
        "switch_kw_present": 1, "followup_kw_present": 0,
        "ambiguous_followup_score": 0.0,
        "domain_kw_present": 1, "prev_top_eq_current_raw_top": 0,
        "prev_top_in_current_top3": 0,
    }),
    ("模糊 followup（短句、followup kw）", {
        **STATE_18D,
        "tv_distance": 0.40, "tv_distance_raw": 0.40,
        "topic_overlap": 0.50,
        "switch_kw_present": 0, "followup_kw_present": 1,
        "ambiguous_followup_score": 1.05,
        "query_len_chars": 6, "query_len_norm": 0.2,
        "domain_kw_present": 0, "prev_top_eq_current_raw_top": 1,
        "prev_top_in_current_top3": 1,
    }),
    ("明確同主題（high topic_overlap）", {
        **STATE_18D,
        "tv_distance": 0.10, "tv_distance_raw": 0.10,
        "topic_overlap": 0.85,
        "domain_kw_present": 1, "prev_top_eq_current_raw_top": 1,
        "prev_top_in_current_top3": 1,
    }),
]


def main():
    version = os.environ.get("MEMORY_AGENT_VERSION", "v1")
    print(f"[ENV] MEMORY_AGENT_VERSION = {version!r}")
    print()

    agent = MemoryAgent()
    print()
    print(f"Active version : {agent.version}")
    print(f"Input dim      : {agent.input_dim}")
    print(f"Model file     : {agent.model_path}")
    print(f"Feature keys   : {agent.feature_keys}")
    print()

    for name, st in SCENARIOS:
        res = agent.select_action(st, deterministic=True)
        probs = res["probs"]
        print(f"[Scenario] {name}")
        print(f"  decision = {res['action_str']:<8}  probs(STAY,REFRESH) = "
              f"({probs[0]:.3f}, {probs[1]:.3f})")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
