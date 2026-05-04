import sys
import os
import json
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from openai import OpenAI
from llm_generate_module.llm_generator import LLMGenerator, LLMConfig

class MultiAgentRewardJudge:
    """
    LLM-as-Judge: 為 3 個 RL Agent 分別獨立評分。
    解決 Credit Assignment Problem（功勞歸因問題）。
    """
    def __init__(self, api_key: str = None, model: str = "gpt-5.4-mini"):
        if api_key:
            # 強制指向 OpenAI，避免被本地 OPENAI_API_BASE 攔截
            self.client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
            self.model = model
            self.use_openai = True
            print(f"[MultiJudge] Using OpenAI API (Forced): {self.model}")
        else:
            config = LLMConfig()
            self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)
            self.model = config.model
            self.use_openai = False
            print(f"[MultiJudge] Using Local LLM: {self.model}")

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1.0,
                max_completion_tokens=500  # 支援 Gemma 3 / O1 等新模型要求
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[MultiJudge] LLM call failed: {e}")
            return ""

    def _parse_score(self, response_text: str) -> float:
        """從 LLM 回覆中提取分數 (1~5 → 正規化到 0~1)"""
        match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', response_text)
        if match:
            raw = float(match.group(1))
            if raw > 1.0:  # 1~5 scale → normalize to 0~1
                return min(max((raw - 1.0) / 4.0, 0.0), 1.0)
            return min(max(raw, 0.0), 1.0)
        # Fallback: try to find any standalone number
        match2 = re.search(r'\b([1-5])\b', response_text)
        if match2:
            return (float(match2.group(1)) - 1.0) / 4.0
        return 0.5  # neutral fallback

    # ========================================================================
    # Judge 1: Planning Agent (資料類型是否切題)
    # ========================================================================
    def judge_planning(self, query: str, retrieved_nodes: list, planning_info: dict) -> float:
        active_sections = planning_info.get("active", [])
        section_names = {
            "assessment": "評估分數",
            "observation": "觀察紀錄",
            "training": "訓練方向",
            "suggestion": "具體建議",
            "community_resources": "社區資源/機構補助",
            "external_gpt": "外部GPT通用衛教",
        }
        active_str = ", ".join([section_names.get(s, s) for s in active_sections])
        
        nodes_summary = ""
        for i, node in enumerate(retrieved_nodes[:5]):
            text = node.get("text", "")[:150]
            label = node.get("label", "Unknown")
            nodes_summary += f"  {i+1}. [{label}] {text}\n"

        system = "你是檢索規劃品質的評估專家。請用 JSON 格式回答，包含 score (1~5) 和 reason。"
        user = f"""使用者問題：「{query}」
系統決定撈取的資料類型：[{active_str}]
實際撈回的前 5 筆內容：
{nodes_summary}
請問：系統選擇的「資料類型」是否符合使用者的提問需求？
- 5 分 = 資料類型完全吻合
- 1 分 = 類型完全不相關
"""
        resp = self._call_llm(system, user)
        return self._parse_score(resp)

    # ========================================================================
    # Judge 2: Rerank Agent (排序是否合理)
    # ========================================================================
    def judge_rerank(self, query: str, retrieved_nodes: list) -> float:
        nodes_summary = ""
        for i, node in enumerate(retrieved_nodes[:5]):
            text = node.get("text", "")[:150]
            label = node.get("label", "Unknown")
            score = node.get("score", 0)
            nodes_summary += f"  {i+1}. (分數{score:.3f}) [{label}] {text}\n"

        system = "你是檢索排序品質的評估專家。請用 JSON 格式回答，包含 score (1~5) 和 reason。"
        user = f"""使用者問題：「{query}」
系統排序後的 Top-5 結果：
{nodes_summary}
請問：排序是否合理？最相關的內容是否排在最前面？
- 5 分 = 排序完美，最相關的內容在最前面
- 1 分 = 排序嚴重錯誤，不相關的排在前面
"""
        resp = self._call_llm(system, user)
        return self._parse_score(resp)

    # ========================================================================
    # Judge 3: Memory Agent (對話延續判定是否正確)
    # ========================================================================
    def judge_memory(self, prev_query: str, current_query: str, memory_action: str) -> float:
        action_names = {"STAY": "延續同一話題", "REFRESH": "重新搜索（切換話題）", "CLARIFY": "要求使用者澄清"}
        action_str = action_names.get(memory_action, memory_action)

        system = "你是對話狀態管理的評估專家。請用 JSON 格式回答，包含 score (1~5) 和 reason。"
        user = f"""使用者上一輪問：「{prev_query}」
使用者這一輪問：「{current_query}」
系統判斷：{action_str}

請問系統的判斷是否正確？
- 5 分 = 判斷完全正確
- 1 分 = 判斷完全錯誤（例如明明換話題了卻說延續）
"""
        resp = self._call_llm(system, user)
        return self._parse_score(resp)

    def judge_chat(self, query, response=None, retrieved_nodes=[], planning_info=None, prev_query=None, memory_action=None, user_feedback=0, raw_nodes=None, **kwargs):
        """
        統一評測核心：一通 API 同時獲取三個 Agent 的 Reward 子項與原因分析。
        支援直接傳入 response text 或透過 retrieved_nodes 評估。
        """
        active_str = ", ".join(planning_info.get("active", [])) if planning_info else "預設"
        
        # 整理最終節點摘要
        nodes_summary = ""
        for i, n in enumerate(retrieved_nodes[:3]):
            txt = n.get("text", "") if isinstance(n, dict) else n.text
            nodes_summary += f"- [{i+1}] {txt[:150]}...\n"

        # 整理原始節點摘要 (供對比用)
        raw_summary = "無"
        if raw_nodes:
            raw_summary = ""
            for i, n in enumerate(raw_nodes[:3]):
                txt = n.get("text", "") if isinstance(n, dict) else n.text
                raw_summary += f"- [Raw {i+1}] {txt[:100]}...\n"

        # 避免 LLM 產生「阿諛奉承 (Sycophancy)」幻覺，改用強制的布林值查核表
        active_sections = planning_info.get("active", []) if planning_info else []
        checklist = ""
        for sec in ["assessment", "observation", "training", "suggestion", "community_resources", "external_gpt"]:
            status = "【已開啟 ON】" if sec in active_sections else "【未開啟 OFF】"
            checklist += f"    - {sec}: {status}\n"

        # 針對 Memory Agent 加上強制的布林值查核表
        memory_checklist = ""
        for act in ["STAY", "REFRESH", "CLARIFY"]:
            status = "【已執行 ON】" if memory_action == act else "【未執行 OFF】"
            memory_checklist += f"    - {act}: {status}\n"

        raw_str = raw_summary if raw_summary.strip() else "(無原始抓取資料，代表 Reranker 無資料可排)"
        rerank_str = nodes_summary if nodes_summary.strip() else "(無最終抓取資料)"

        system = f"""你是一個 RAG (檢索增強生成) 系統的嚴格評審。
請針對以下對話上下文與 Agent 決策，分別為三個 Agent 的表現打分 (1~5分)。

【當前查詢 (Query)】: {query}
【系統回覆 (Response)】: {response if response else '無'}
【歷史上下文 (Prev Context)】: {prev_query if prev_query else '無'}
【使用者顯式回饋】: {user_feedback if user_feedback != 0 else '無'}

【檢索決策與結果 (嚴格事實)】
1. Planning Agent (檢索規劃員) 實際決策機率與開關狀態：
   - 預測機率 (Probs): {planning_info.get('probs', {})}
   - 最終啟動清單: {checklist}
   
   區塊定義：
   - assessment/observation/training/suggestion: 針對特定個案報告內容。
   - community_resources: 針對「外部機構、補助、社福資源」進行 MySQL 資料庫查詢。
   - external_gpt: 針對報告中不存在的「通用育兒建議/衛教」讓 GPT 直接回答。

2. Rerank Agent (排序優化員) 實際處理的資料如下 (嚴格對比下述資料，不要憑空想像)：
   [A. 原始抓取 (Raw)]: 
   {raw_str}
   
   [B. 最終排序 (Reranked)]: 
   {rerank_str}

3. Memory Agent (大腦) 實際執行的動作如下 (禁止幻想它做了別的動作)：
{memory_checklist}
   動作定義：
   - STAY: 延續上一輪話題與上下文，不重置檢索範圍。
   - REFRESH: 清除舊背景，視為全新話題開啟檢索。
   - CLARIFY: 對方語意不清或太簡短，主動發問引導。

【評核流程 - 嚴格事實對接】
1. 首先，請列出 Agent 「實際開啟」了哪些區塊 (即 active_str 裡標註的內容)。
2. 其次，根據對話與法律手冊，判斷「這題理論上需要」哪些區塊。
3. 差異比對：
   * 如果 Agent 實際開啟的內容與理論需求「完全不符」(例如問分數卻開了社區資源)，即便系統回答(Answer)再好，Planning 評分也必須判定為 1 分或 2 分！
   * 絕不能因為 Answer 做得好，就幻想 Planning 做了正確決策！

【評估準則 - 零容忍扣分制 (DEDUCTION-BASED EVALUATION)】
**警告：請絕對將 Planning (規劃) 與 Answer (回答) 分開評分！Planning 必須僅針對 `active_str` 所列內容進行事實審查。**

  * [基礎評分]: 所有 Agent 起評分均為 3 分。
  * 如果 Agent 選取了顯然與問題相關的內容，依據切題程度進行加分 (+1 ~ +2)。
  * 若漏選了「關鍵內容」(例如問分數卻沒選 assessment)，則進行減分 (-1 ~ -2)。
  * [暫定規則]：為了鼓勵初期探索，只要有部分切題，請給予及格分數 (3分以上)，不要因為开启過多或少量漏選而給予極低分。

- Reranking (1~5)：對比 [A] 與 [B]。
  * 如果 [A] 中有正確答案但 [B] 沒有，說明 Reranker 執行失誤 (低分)。
  * 如果 [B] 成功將相關內容頂到最前面，說明 Reranker 表現優異 (高分)。
- Memory (1~5)：決策是否合邏輯？
  * [關鍵決策：話題切換]: 
    - 如果 【當前查詢】 與 【歷史上下文】 屬於完全不同的臨床領域 (例如從 粗大動作 換到 口語、或者從 發展遲緩 換到 社區資源)，Memory Agent **必須選 REFRESH**。如果此時選了 STAY，視為「話題污染」，必須給予 1~2 分。
    - 如果對話是 針對同一主題的深入追問 (例如：問完走路穩嗎，接著問單腳站立)，**必須選 STAY**。
    - 如果語意模糊 (例如：那這個呢？) 且無法從前文推斷，應選 **CLARIFY**。
  * 評分時請參考下方 【Planning Probs】 分佈情形判斷領域 shift 是否發生。

- Answer (1~5)：最終回答是否專業、親切且具備臨床權威？

請直接回傳 JSON，格式如下:
{{{{
  "planning_reward": float (1~5),
  "rerank_reward": float (1~5),
  "answer_reward": float (1~5),
  "memory_reward": float (1~5),
  "analysis_log": "診斷原因"
}}}}
"""
        user = "請根據上述準則進行評估。"

        resp_text = self._call_llm(system, user)
        # 提取 JSON
        scores_raw = {}
        try:
            # 尋找 JSON 區塊
            json_match = re.search(r'\{.*\}', resp_text, re.DOTALL)
            if json_match:
                scores_raw = json.loads(json_match.group(0))
        except:
            pass

        # 轉換分數 (1~5 -> 0~1) 並結合 User Feedback
        feedback_weight = 0.4
        judge_weight = 0.6
        fb_norm = (user_feedback + 1.0) / 2.0 if user_feedback != 0 else None

        active_list = planning_info.get("active", []) if planning_info else []
        num_active = len(active_list)

        def get_final_reward(val_key, default_val=3.0):
            raw = scores_raw.get(val_key, default_val)
            try:
                val = float(raw)
            except:
                val = default_val
            
            # [移除所有硬編碼處罰與獎勵] 依照用戶要求，僅保留 LLM Judge 的原始評分。
            # 程式碼不再干預分數，鼓勵純粹由臨床邏輯引導的學習。
            if val_key == "planning_reward":
                pass 
            
            # 保存 un-normalized 到 results 外部 (不優雅但實用)
            if val_key == "planning_reward":
                self._last_p_val = val

            s = min(max((val - 1.0) / 4.0, 0.0), 1.0)
            if fb_norm is not None:
                return judge_weight * s + feedback_weight * fb_norm
            return s

        results = {
            "planning_reward": get_final_reward("planning_reward"),
            "rerank_reward": get_final_reward("rerank_reward"),
            "answer_reward": get_final_reward("answer_reward"),
            "memory_reward": get_final_reward("memory_reward"),
            "raw_scores": {
                "planning": float(scores_raw.get("planning_reward", 0)),
                "planning_final": float(self._last_p_val),
                "rerank": float(scores_raw.get("rerank_reward", 0)),
                "answer": float(scores_raw.get("answer_reward", 0)),
                "num_active": num_active
            },
            "analysis_log": scores_raw.get("analysis_log", "")
        }

        return results


if __name__ == "__main__":
    judge = MultiAgentRewardJudge()
    
    # Test
    query = "精細動作有什麼居家練習建議？"
    nodes = [
        {"label": "Recommendation", "text": "建議家長讓孩子多使用安全剪刀練習剪直線", "score": 0.85},
        {"label": "AssessmentScore", "text": "PDMS-2 精細動作原始分 83", "score": 0.42},
    ]
    planning_info = {"active": ["suggestion", "training"], "probs": {"assessment": 0.1, "suggestion": 0.8}}
    
    results = judge.judge_chat(
        query=query,
        retrieved_nodes=nodes,
        planning_info=planning_info,
        prev_query="他的評估結果怎樣？",
        memory_action="STAY",
        user_feedback=0.0
    )
    print(f"Results: {json.dumps(results, indent=2)}")
