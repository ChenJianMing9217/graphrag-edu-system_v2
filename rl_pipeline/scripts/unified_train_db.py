"""
unified_train_db.py — 統一離線訓練腳本 (v2: Multi-Agent LLM-as-Judge)

從 SQL ChatMessage 讀取歷史對話，利用 LLM 分別為 3 個 RL Agent 評分並更新模型。

用法：
    python rl_pipeline/scripts/unified_train_db.py
    （可選）設定環境變數 OPENAI_API_KEY 以使用 OpenAI API 作為 Judge
"""
import sys
import os
import json
import torch
from tqdm import tqdm
from dotenv import load_dotenv
import random

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app import app, db, ChatMessage
from rl_pipeline.agents.reranker.rerank_agent import RLAgentManager
from rl_pipeline.agents.memory.memory_agent import MemoryAgent
from rl_pipeline.agents.planner.planning_agent import PlanningAgent
from rl_pipeline.shared.reward_judge import MultiAgentRewardJudge


# [Problem 8] 只取最近 N 筆 bot 訊息作為訓練窗口，避免用全量歷史導致 off-policy 偏移
TRAINING_WINDOW = 200


def compute_proxy_reward(flow_state: dict, retrieved_nodes: list) -> dict:
    """
    [Problem 10] 輕量代理獎勵（不呼叫 LLM），作為 30% 穩定補充。
    - planning/rerank proxy: 取前 3 筆節點平均向量分數
    - memory proxy: context_sim 與 topic_overlap 的平均
    """
    scores = [float(n.get("score", 0.5)) for n in retrieved_nodes[:3] if isinstance(n, dict)]
    retrieval_proxy = sum(scores) / len(scores) if scores else 0.5

    context_sim = float(flow_state.get("context_sim", 0.5))
    topic_overlap = float(flow_state.get("topic_overlap", 0.5))
    memory_proxy = (context_sim + topic_overlap) / 2.0

    return {
        "planning_proxy": retrieval_proxy,
        "rerank_proxy": retrieval_proxy,
        "memory_proxy": memory_proxy,
    }


def reward_from_feedback(actual_feedback: float, proxy_score: float) -> float:
    """
    有使用者明確回饋（按讚/踩）時的獎勵計算。
    - 使用者回饋為主（70%），代理獎勵補充（30%）
    - actual_feedback: +1（讚）/ -1（踩）→ 正規化到 [0, 1]
    """
    fb_norm = (actual_feedback + 1.0) / 2.0
    return 0.7 * fb_norm + 0.3 * proxy_score


def reward_from_llm(llm_score: float, proxy_score: float) -> float:
    """
    無使用者回饋時，改用 LLM Judge 作為獎勵來源。
    - LLM Judge 為主（70%），代理獎勵補充（30%）
    - llm_score: 已正規化到 [0, 1] 的 Judge 分數
    """
    return 0.7 * llm_score + 0.3 * proxy_score


def run_unified_training(openai_api_key=None, gamma=0.9, num_epochs=15):
    """
    1. 提取 flow_state 中的各 Agent 狀態與動作
    2. 針對 session_id 群組計算 discounted returns (遞延回饋)
    3. 呼叫 LLM Judge 評分
    4. [NEW] 使用 Shuffled Buffer 進行多 Epoch 訓練，提高穩定性
    """
    # --- 離線訓練設定 ---
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 
    
    # 1. 初始化 3 個 Agent
    rerank_agent = RLAgentManager(model_path="rl_pipeline/agents/reranker/models/rerank_agent.pth")
    memory_agent = MemoryAgent(model_path="rl_pipeline/agents/memory/models/memory_agent.pth")
    planning_agent = PlanningAgent(model_path="rl_pipeline/agents/planner/models/planning_agent.pth")

    # 2. 初始化 LLM Judge
    judge = MultiAgentRewardJudge(api_key=OPENAI_API_KEY, model="gpt-5.4-mini")

    # 3. 讀取數據與分組
    with app.app_context():
        print("[Trainer] 從 ChatMessage 表讀取 bot 回覆紀錄...")
        # [Problem 8] 只取最近 TRAINING_WINDOW 筆，避免用全量歷史導致 off-policy 偏移
        bot_msgs = ChatMessage.query.filter(
            ChatMessage.is_user_message == False,
            ChatMessage.flow_state.isnot(None)
        ).order_by(ChatMessage.sent_at.desc()).limit(TRAINING_WINDOW).all()
        # Re-sort 以維持 session 內的時間順序（用於 discounted return 計算）
        bot_msgs.sort(key=lambda m: (m.session_id or "", m.sent_at))

        if not bot_msgs:
            print("[Trainer] 無可用訓練紀錄。")
            return

        all_user_msgs = ChatMessage.query.filter(ChatMessage.is_user_message == True).all()
        user_msg_by_child = {}
        for um in all_user_msgs:
            key = um.child_id or 0
            if key not in user_msg_by_child: user_msg_by_child[key] = []
            user_msg_by_child[key].append(um)

        sessions = {}
        for msg in bot_msgs:
            sid = msg.session_id or f"legacy_{msg.id}"
            if sid not in sessions: sessions[sid] = []
            sessions[sid].append(msg)

        print(f"[Trainer] 共 {len(bot_msgs)} 筆紀錄，正在進行 Cross-Agent LLM 評分與特徵提取...")

        # 4. 收集訓練資料 (Experience Buffers)
        planning_exp = []
        memory_exp = []
        rerank_exp = []  # SAR #4: Rerank Buffer
        rerank_count = 0
        skipped = 0

        for sid, msgs in tqdm(sessions.items(), desc="Sessions"):
            # 取出最後一個實質性回饋作為 Reward 基底
            final_reward = 0.0
            for msg in msgs:
                if msg.feedback_value: final_reward = float(msg.feedback_value)
            
            num_steps = len(msgs)
            for t, bot_msg in enumerate(msgs):
                # A. 遞延報酬計算 (Discounted Return)
                immediate_r = float(bot_msg.feedback_value or 0)
                discounted_r = immediate_r if immediate_r != 0 else (gamma ** (num_steps - 1 - t)) * final_reward

                # [Problem 9] actual_feedback: 只有使用者明確按讚/踩時才有值（非 discounted_r）
                actual_feedback = float(bot_msg.feedback_value) if bot_msg.feedback_value else None

                try:
                    flow_state = json.loads(bot_msg.flow_state)
                    retrieval_info = json.loads(bot_msg.retrieval_info or "[]")

                    # 匹配當前問題與上一輪問題
                    child_key = bot_msg.child_id or 0
                    child_um = user_msg_by_child.get(child_key, [])
                    u_q, p_q = "", ""
                    for i, um in enumerate(child_um):
                        if um.sent_at < bot_msg.sent_at:
                            u_q = um.message
                            if i > 0: p_q = child_um[i-1].message

                    if not u_q:
                        skipped += 1
                        continue

                    memory_action = flow_state.get("memory_action", "STAY")
                    planning_info = flow_state.get("planning_info", {})
                    semantic_scores = flow_state.get("semantic_section_scores", {})

                    # 代理獎勵（輕量，永遠計算，不呼叫 LLM）
                    proxy = compute_proxy_reward(flow_state, retrieval_info[:5])

                    if actual_feedback is not None:
                        # 路徑 A：有使用者明確回饋 → 直接用回饋，不呼叫 LLM Judge
                        rewards = {
                            "planning_reward": reward_from_feedback(actual_feedback, proxy["planning_proxy"]),
                            "rerank_reward":   reward_from_feedback(actual_feedback, proxy["rerank_proxy"]),
                            "memory_reward":   reward_from_feedback(actual_feedback, proxy["memory_proxy"]),
                        }
                    else:
                        # 路徑 B：無使用者回饋 → 呼叫 LLM Judge（3 個獨立評分）
                        p_judge = judge.judge_planning(u_q, retrieval_info[:5], planning_info)
                        r_judge = judge.judge_rerank(u_q, retrieval_info[:5])
                        m_judge = judge.judge_memory(p_q, u_q, memory_action)
                        rewards = {
                            "planning_reward": reward_from_llm(p_judge, proxy["planning_proxy"]),
                            "rerank_reward":   reward_from_llm(r_judge, proxy["rerank_proxy"]),
                            "memory_reward":   reward_from_llm(m_judge, proxy["memory_proxy"]),
                        }

                    # C. 提取特徵並存入 Buffer (等待批次訓練)
                    # 1. Planning Agent
                    if "planning_reward" in rewards and semantic_scores:
                        active = planning_info.get("active", [])
                        mask = [1.0 if s in active else 0.0 for s in planning_agent.section_labels]
                        # 獎勵已在 reward_judge 中完成 1~5 → 0~1 正規化，直接使用
                        p_reward = float(rewards["planning_reward"])
                        
                        state_p = {
                            "sem_assessment": semantic_scores.get("assessment", 0),
                            "sem_observation": semantic_scores.get("observation", 0),
                            "sem_training": semantic_scores.get("training", 0),
                            "sem_suggestion": semantic_scores.get("suggestion", 0),
                            "sem_community_resources": semantic_scores.get("community_resources", 0), # SAR #2
                            "sem_external_gpt": semantic_scores.get("external_gpt", 0),         # SAR #2
                            "domain_entropy": float(flow_state.get("normalized_entropy", 0.0)),   # SAR #2
                            "task_label": flow_state.get("task_pred", "A"),
                            "secondary_tasks": flow_state.get("secondary_tasks", []),  # 多任務：訓練與推理保持一致
                            "task_dist": flow_state.get("task_dist", {}),              # 多任務：confidence-weighted 特徵
                        }
                        planning_exp.append((state_p, mask, p_reward))

                    # 2. Memory Agent (對話狀態特徵)
                    if "memory_reward" in rewards:
                        act_idx = {"STAY":0, "REFRESH":1, "CLARIFY":2}.get(memory_action, 0)
                        # 獎勵已在 reward_judge 中完成 1~5 → 0~1 正規化，直接使用
                        m_reward = float(rewards["memory_reward"])
                        
                        state_m = {
                            "entropy": float(flow_state.get("normalized_entropy", 0.5)),
                            "tv_distance": float(flow_state.get("tv_distance", 0.5)),
                            "topic_overlap": float(flow_state.get("topic_overlap", 0.5)),
                            "context_sim": float(flow_state.get("context_sim", 0.5)),
                            "turn_index_norm": min(float(flow_state.get("turn_index", 0)) / 10.0, 1.0),
                            "query_len_norm": min(float(len(u_q)) / 50.0, 1.0),
                            "is_multi_domain": flow_state.get("is_multi_domain", False),
                        }
                        memory_exp.append((state_m, act_idx, m_reward))

                    if "rerank_reward" in rewards:
                        r_reward = float(rewards["rerank_reward"])
                        # 使用實際使用的權重供訓練
                        actual_weights = [
                            float(flow_state.get("rerank_w_semantic", 0.6)),
                            float(flow_state.get("rerank_w_structural", 0.2)),
                            float(flow_state.get("rerank_w_context", 0.2)),
                        ]
                        state_r_tuple = (
                            flow_state.get("task_pred", "A"), 
                            flow_state.get("top_domain", "整體概況"), 
                            flow_state.get("scope_pred", "S_domain")
                        )
                        cont_r = {
                            "entropy": float(flow_state.get("normalized_entropy", 0.5)),
                            "top_prob": float(flow_state.get("top_domain_prob", 0.3)),
                            "context_sim": float(flow_state.get("context_sim", 0.5)),
                            "topic_overlap": float(flow_state.get("topic_overlap", 0.5)),
                            "turn_index_norm": min(float(flow_state.get("turn_index", 0)) / 10.0, 1.0)
                        }
                        # RerankAgent.update 現在接受一個 buffer list
                        rerank_exp.append((state_r_tuple, actual_weights, r_reward, cont_r))
                        rerank_count += 1

                except Exception as e:
                    print(f"\n[Error] msg_id {bot_msg.id}: {e}")
                    skipped += 1

        # 5. [核心優化] 多次 Epoch 隨機批次訓練
        p_loss, m_loss = 0, 0
        if planning_exp:
            avg_p_reward = sum([x[2] for x in planning_exp]) / len(planning_exp)
            print(f"\n[Planning Agent Training] 樣本數: {len(planning_exp)} | 當前平均獎助: {avg_p_reward:.4f}")
            for epoch in range(num_epochs if num_epochs > 15 else 30): # 強制至少 30 epoch
                random.shuffle(planning_exp)
                p_loss = planning_agent.update(planning_exp)
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1} - Loss: {p_loss:.4f}")
            planning_agent.save()

        if memory_exp:
            # Reward Centering（基線減法）：移除平均值，減少 reward scale 偏差造成的梯度不穩定
            mean_m_reward = sum(e[2] for e in memory_exp) / len(memory_exp)
            memory_exp = [(s, a, r - mean_m_reward) for s, a, r in memory_exp]
            print(f"\n[Memory Agent Training] 樣本數: {len(memory_exp)} | 基線: {mean_m_reward:.4f}")
            for epoch in range(num_epochs):
                random.shuffle(memory_exp)
                m_loss = memory_agent.update(memory_exp)
                print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {m_loss:.4f}")
            memory_agent.save()

        if rerank_exp:
            print(f"\n[Rerank Agent Training] 樣本數: {len(rerank_exp)}")
            r_loss = rerank_agent.update(rerank_exp)
            print(f"  Dirichlet Policy Loss: {r_loss:.4f}")
            rerank_agent.save()

        # 6. [NEW] 儲存訓練日誌與指標
        history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../logs/training_history.json")
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        
        history = []
        if os.path.exists(history_path):
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except: history = []

        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        avg_p_reward = sum(e[2] for e in planning_exp) / len(planning_exp) if planning_exp else 0
        avg_m_reward = sum(e[2] for e in memory_exp) / len(memory_exp) if memory_exp else 0
        
        new_entry = {
            "timestamp": timestamp,
            "samples": {"planning": len(planning_exp), "memory": len(memory_exp), "rerank": rerank_count},
            "metrics": {
                "planning_avg_reward": round(float(avg_p_reward), 4),
                "memory_avg_reward": round(float(avg_m_reward), 4),
                "planning_loss": round(float(p_loss), 4),
                "memory_loss": round(float(m_loss), 4)
            }
        }
        
        history.append(new_entry)
        
        # 僅保留最近 50 次紀錄以免 JSON 過大
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history[-50:], f, indent=2, ensure_ascii=False)

        print(f"\n指標已更新至：{history_path}")
        
        # 自動生成儀表板
        try:
            from rl_pipeline.scripts.generate_dashboard import generate_html_dashboard
            dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../training_dashboard.html")
            generate_html_dashboard(history[-50:], dashboard_path)
        except Exception as de:
            print(f"自動生成儀表板失敗: {de}")

        print(f"{'=' * 60}")
        print(f" 訓練任務完成。 成功樣本：Planning={len(planning_exp)}, Memory={len(memory_exp)}, Rerank={rerank_count}")
        print(f"{'=' * 60}")

if __name__ == "__main__":
    run_unified_training()
