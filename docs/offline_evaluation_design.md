# 離線評估資料集設計與評估指南 (Offline Evaluation Dataset Design)

本文件定義了多 Agent 對話系統（Memory Agent, Planning Agent, Retrieval Agent, Final Gen）的離線評估資料集 schema 與標註原則，並說明如何利用本資料集評估各模組的效能。

## 1. 資料集 Schema (JSONL 格式)

每筆測試資料將以 JSON 格式儲存，並包含以下關鍵評估欄位：

```json
{
  "sample_id": "eval_001",
  "conversation_context": [
    {"role": "user", "content": "我的小孩最近剛做完早療評估。"},
    {"role": "assistant", "content": "了解，請問您想了解評估報告中的哪個部分呢？例如認知能力、語言發展，或是整體的觀察結果？"}
  ],
  "user_query": "那他的語言發展分數正常嗎？",
  "gold_memory_action": "STAY",
  "gold_sections": ["assessment"],
  "gold_retrieval_targets": ["語言發展測驗分數 (Language Assessment Score)", "同年齡常模比較 (Norm Comparison)"],
  "gold_answer": "根據評估報告，孩子的語言發展分數落在同年齡層的正常範圍內（例如 PR 55），顯示語言理解與表達皆符合發展期待。",
  "answer_style_notes": "語氣應具有同理心，並且需明確引述資料中的數據（如 PR 值）。",
  "difficulty": "Easy",
  "notes_for_evaluation": "因為前文已經提到評估報告，此處使用代名詞「他」，Memory Agent 必須判定為 STAY。Planning 只要開 assessment 即可。"
}
```

## 2. 標註原則與定義

### A. Memory Action 的判定原則
根據 `MemoryAgent` 的 Action Space (`STAY`, `REFRESH`, `CLARIFY`)：
*   **STAY (延續話題)**
    *   **觸發條件**：User 的問題語意不完整，但可透過前幾輪的對話上下文完美補足。例如代名詞使用（如「那這項表現呢？」、「這個評估是幾月做的？」），或是針對剛剛 bot 的回答進行衍伸追問。
    *   **評估挑戰**：系統是否有正確保留 Memory Buffer 供後續檢索，不發生 Context Loss。
*   **REFRESH (轉換話題)**
    *   **觸發條件**：User 問了一個與前文完全無關、或屬於全新領域的獨立問題。例如前一秒還在問「認知分數」，下一秒問「請問要怎麼申請身心障礙手冊？」。
    *   **評估挑戰**：系統必須懂得清空或極大降低先前的 Context 權重，避免舊有關鍵字（如「認知」）去污染新問題的檢索結果，發生 Context Pollution。
*   **CLARIFY (語意澄清)**
    *   **觸發條件**：User 提問的資訊量過少，且前文也**無法**推斷其意圖（如單留一句：「怎麼辦？」、「那這個呢？」但前文有多個焦點），或者提出系統根本無法回答的矛盾問題。
    *   **評估挑戰**：系統必須拒絕瞎猜，選擇直接反問 User 以縮小檢索範圍。

### B. Planning Section 的標註原則
根據 `PlanningAgent` (6 維 Action Space)，針對該提取何種知識庫進行判斷：
*   **assessment (評估量表)**：當 Query 想要了解「測驗分數」、「常模比較」、「PR值」、「發展遲緩程度」時（例：魏氏智力測驗、VMI、嬰幼兒綜合發展測驗）。
*   **observation (臨床觀察)**：當 Query 詢問「上課專心度」、「異常行為」、「治療師怎麼看他」、「情緒表現」時。
*   **training (訓練計畫)**：當 Query 要求「治療所的課程規劃」、「治療目標」、「下週要在診所做什麼」時。
*   **suggestion (居家建議)**：當 Query 想要「家長回家可以怎麼教」、「日常生活怎麼引導」、「親子共讀建議」時。
*   **community_resources (社區資源)**：當 Query 涉及「政府補助」、「早療交通費」、「身心障礙手冊申請」、「轉介醫院名單」時。
*   **external_gpt (外部通用知識)**：當 Query 不是問個人報告，而是問客觀知識時，如「什麼是自閉症？」、「ADHD 會有什麼普遍症狀？」。

### C. 評估指標對應 (How to Use)
這個資料集可串聯整條 RL Pipeline 進行以下離線評估 (Offline Metrics)：
1.  **Memory Agent Accuracy**
    *   公式：`Count(pred_action == gold_memory_action) / Total Samples`
    *   分析：如果 STAY 準確率低，代表容易漏掉對話脈絡；如果 REFRESH 準確率低，代表系統容易被歷史干擾。
2.  **Planning Section F1-Score**
    *   公式：計算 `pred_sections` 與 `gold_sections` 的 Multi-label F1-Score。
    *   分析：檢查是否會出現過度檢索 (Recall 高但 Precision 低) 或檢索不足 (Precision 高但 Recall 低) 的問題。
3.  **Retrieval Hit@k / Recall@k**
    *   公式：比對 Retriever 最後生成的 Context 文本中，是否涵蓋 `gold_retrieval_targets` 列出的語義實體或節點概念。
    *   分析：這反映出 Rerank Agent 及 Vector DB 的檢索品質。
4.  **Final Answer Correctness / LLM-as-a-Judge**
    *   公式：將生成的 Answer、`gold_answer` 及 `notes_for_evaluation` 丟給高階模型（如 GPT-4）進行評分 (1-5分)。
    *   分析：檢驗「即時答案是否精準且不產生幻覺」，而且是否符合設定的語氣 (`answer_style_notes`)。
