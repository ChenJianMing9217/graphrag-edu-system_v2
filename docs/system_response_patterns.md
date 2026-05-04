# 系統應對方式總覽（System Response Patterns）

本文件說明系統在各種使用者輸入情境下的完整決策邏輯與應對方式。

---

## 一、總決策流程圖

```
使用者輸入
    │
    ▼
┌──────────────────────┐
│ B0. 閒聊快速攔截       │  字數 ≤ 10 且含「你好/謝謝/嗨」等關鍵字
│     → 直接回覆，跳過    │  不經過 DST / 檢索 / RL
│       所有下游模組       │  延遲：< 1 秒
└──────────┬───────────┘
           │ 非閒聊
           ▼
┌──────────────────────┐
│ B1. 文字編碼（一次性）  │  query_vector = encoder.encode(user_input)
│     全流程共用此向量     │  避免重複編碼（省 ~300ms）
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ B2. DST 分析           │
│  ├─ 領域分類            │  → top_domain + active_domains + entropy
│  ├─ 任務分類            │  → task_label (A~N) + 分布
│  ├─ 上下文相似度        │  → context_sim (C 值)
│  ├─ 主題延續度          │  → overlap_score (MT 值)
│  └─ 策略決策            │  → semantic_flow + memory_action + retrieval_action
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ C. 領域外/信心不足偵測  │  cosine < 0.30 或 top_prob < 閾值
│     → 引導式回覆        │  「請問您想了解報告的哪個部分？」
└──────────┬───────────┘
           │ 領域內
           ▼
┌──────────────────────┐
│ D. RL 權重 + 檢索      │  PlanningAgent → 開哪些 section
│                        │  RerankerAgent → 三維排序權重
│                        │  MemoryAgent → STAY/REFRESH/CLARIFY
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ E. LLM 生成            │  依任務、語意流、是否模糊
│                        │  選擇對應 prompt 模板與生成參數
└──────────┘
```

---

## 二、依「語意流向」分類的應對方式

語意流向由 C 值（上下文相似度）和 MT 值（主題延續度）共同決定。

### 2.1 continue（延續同話題）

| 條件 | C ≥ 0.55 且 MT ≥ 0.50 |
|------|------------------------|
| **memory_action** | STAY — 保留上一輪的領域上下文 |
| **retrieval_action** | NARROW_GRAPH — 限縮到 top 1 領域 |
| **prev_context 攜帶** | 攜帶上一輪 top-3 檢索結果作為輔助參考 |
| **LLM 行為** | 可使用代名詞解析、延續前文語境 |

**典型情境：**
```
用戶：他的粗大動作分數是多少？
系統：（回覆粗大動作分數）
用戶：那跟同年齡比是落後嗎？        ← continue
系統：（延續粗大動作語境回答）
```

**QueryRewriter 行為：**
- 啟動 Query Condensation：將「那跟同年齡比是落後嗎？」+ 歷史 → 「粗大動作分數與同齡常模比較是否落後」

---

### 2.2 shift_soft（軟切換）

| 條件 | C 或 MT 其中一個高，另一個低；或雙方都在中間地帶 |
|------|--------------------------------------------------|
| **memory_action** | REFRESH — 更新為當前偵測的領域 |
| **retrieval_action** | WIDE_IN_DOMAIN — 在 active_domains 範圍內廣泛搜尋 |
| **prev_context 攜帶** | 若 context_sim ≥ 0.45 則攜帶（去重），否則不帶 |
| **LLM 行為** | 同時考慮新舊領域，回覆中可做自然過渡 |

**典型情境：**
```
用戶：精細動作表現如何？
系統：（回覆精細動作觀察）
用戶：那這跟粗大動作有關嗎？        ← shift_soft（從精細→粗大，但有關聯）
系統：（帶入精細動作背景，說明與粗大動作的關係）
```

---

### 2.3 shift_hard（硬切換）

| 條件 | C < 0.45 且 MT < 0.30 |
|------|------------------------|
| **memory_action** | REFRESH — 完全重置領域上下文 |
| **retrieval_action** | DUAL_OR_CLARIFY 或 WIDE_IN_DOMAIN |
| **prev_context 攜帶** | 不攜帶 |
| **LLM 行為** | 獨立回答，不引用前文 |

**典型情境：**
```
用戶：粗大動作的百分等級是多少？
系統：（回覆 PR 值）
用戶：對了，政府有什麼補助可以申請？  ← shift_hard（從分數→補助，完全不同）
系統：（獨立回答補助相關問題）
```

---

## 三、依「任務類型 (A~N)」分類的應對方式

### A — 報告總覽與閱讀順序

| 項目 | 設定 |
|------|------|
| **檢索重點** | Summary + Meta 節點 |
| **Planning sections** | assessment (偏高) |
| **LLM 風格** | 結構化摘要、條列式、先給結論再展開 |
| **Scope 通常為** | S_overview |
| **Reranker boost** | Summary, Meta |

**典型問法：** 「這份報告主要在講什麼？」「幫我用三句話抓重點」

---

### B — 分數/量表/百分位解讀

| 項目 | 設定 |
|------|------|
| **檢索重點** | Score + AssessmentTool 節點 |
| **Planning sections** | assessment (高) |
| **LLM 風格** | 精確引用數字、解釋常模意義 |
| **特殊行為** | 引用 PR 值、標準分、信心區間 |
| **Reranker boost** | Assessment, Score |

**典型問法：** 「標準分 20 是什麼意思？」「百分等級 8 代表落後嗎？」

---

### C — 臨床觀察行為描述

| 項目 | 設定 |
|------|------|
| **檢索重點** | Observation 節點 |
| **Planning sections** | observation (高) |
| **LLM 風格** | 描述性語言、連結日常情境 |
| **臨床增強** | 對接 ClinicalNorm（觀察指標→對應能力） |
| **Reranker boost** | Observation, Assessment |

**典型問法：** 「報告裡觀察到他有什麼具體表現？」「核心肌力不足會怎樣？」

---

### D — 能力剖面（強弱項分析）

| 項目 | 設定 |
|------|------|
| **檢索重點** | Assessment + Observation（多面向） |
| **Planning sections** | assessment + observation |
| **LLM 風格** | 比較式分析、明確指出強項與弱項 |
| **Reranker boost** | Assessment, Observation |

**典型問法：** 「他最弱的是哪一項？」「強項在哪裡？」

---

### E — 在家 PT 訓練建議

| 項目 | 設定 |
|------|------|
| **檢索重點** | TrainingDirection + Recommendation |
| **Planning sections** | training + suggestion (高) |
| **LLM 風格** | 行動導向、具體步驟、居家可執行 |
| **臨床增強** | 訓練策略 + 訓練活動（里程碑對接） |
| **Reranker boost** | TrainingDirection, Recommendation |

**典型問法：** 「在家可以怎麼幫他練習？」「有推薦的居家訓練嗎？」

---

### F — 融入日常粗大動作練習

| 項目 | 設定 |
|------|------|
| **檢索重點** | TrainingDirection + Observation |
| **Planning sections** | training + observation |
| **LLM 風格** | 生活化、融入作息、不像「上課」 |
| **Reranker boost** | TrainingDirection, Observation |

**典型問法：** 「上下樓梯時可以順便練什麼？」「去公園玩什麼設施有幫助？」

---

### G — 是否需要 PT / 早療評估

| 項目 | 設定 |
|------|------|
| **檢索重點** | Assessment + Recommendation |
| **Planning sections** | assessment + suggestion |
| **LLM 風格** | 謹慎判斷、引用數據佐證、不替代醫療決策 |
| **Reranker boost** | Assessment, Recommendation |

**典型問法：** 「他需要去做物理治療嗎？」「這樣算嚴重嗎？」

---

### H — 轉介在地資源 / 機構查詢

| 項目 | 設定 |
|------|------|
| **檢索重點** | MySQL 社區資源 + SubsidyProgram |
| **特殊行為** | 偵測地區關鍵字（台北市、新北市...） |
| **retrieval_action** | LOCAL_RESOURCE_SEARCH（有地區） / LOCAL_RESOURCE_CLARIFY（無地區） |
| **LLM 風格** | 具體機構名稱、地址、電話 |

**無地區時的應對：**
```
用戶：有推薦的早療機構嗎？
系統：請問您目前住在哪個縣市呢？這樣我可以幫您查詢最近的資源。  ← LOCAL_RESOURCE_CLARIFY
```

**有地區時的應對：**
```
用戶：台北市有推薦的復健診所嗎？
系統：（查詢 MySQL，回傳台北市的機構清單）  ← LOCAL_RESOURCE_SEARCH
```

---

### I — 報告分享 / 隱私與安全

| 項目 | 設定 |
|------|------|
| **檢索重點** | 外部 GPT 知識 + 報告 Meta |
| **Planning sections** | external_gpt (高) |
| **LLM 風格** | 隱私保護導向、最小必要分享原則 |

**典型問法：** 「我可以把報告給老師看嗎？」「保險要用報告，給哪幾頁安全？」

---

### J — 與學校合作

| 項目 | 設定 |
|------|------|
| **檢索重點** | Observation + TrainingDirection + 外部知識 |
| **Planning sections** | observation + training + external_gpt |
| **LLM 風格** | 協作建議、如何與老師溝通 |

**典型問法：** 「怎麼跟老師說明這份報告？」「體育課可以怎麼調整？」

---

### K — 補助 / 福利申請

| 項目 | 設定 |
|------|------|
| **檢索重點** | MySQL SubsidyProgram + 外部知識 |
| **特殊行為** | 同 H，偵測地區關鍵字 |
| **LLM 風格** | 具體申請流程、所需文件、金額 |

**典型問法：** 「早療補助要怎麼申請？」「桃園有交通費補助嗎？」

---

### L — 追蹤再評估

| 項目 | 設定 |
|------|------|
| **檢索重點** | Assessment + TrainingDirection |
| **Planning sections** | assessment + training |
| **LLM 風格** | 時間建議、追蹤指標、目標設定 |
| **Reranker boost** | Assessment, TrainingDirection |

**典型問法：** 「什麼時候該再評估一次？」「追蹤重點放哪？」

---

### M — 家長情緒支持與壓力調適

| 項目 | 設定 |
|------|------|
| **檢索重點** | 外部 GPT 知識為主 |
| **Planning sections** | external_gpt (高) + suggestion |
| **LLM 風格** | 同理心優先、溫暖語氣、肯定家長努力 |
| **特殊行為** | temperature 較高 (更自然)、不強調數據 |

**典型問法：** 「我最近教到好累不知道怎麼辦」「我覺得壓力好大」

---

### N — 進步查詢（跨報告比較）

| 項目 | 設定 |
|------|------|
| **檢索重點** | 最近 2 份報告的 Assessment + Score |
| **特殊行為** | 自動抓取最近 2 份 report，同時檢索兩份 |
| **LLM 風格** | 對比式分析、明確說明進步/持平/退步 |
| **Reranker boost** | Assessment, Score |

**典型問法：** 「跟上次比有進步嗎？」「哪個領域進步最多？」

---

## 四、依「特殊狀態」分類的應對方式

### 4.1 模糊查詢（is_ambiguous = True）

**觸發條件：** entropy ≥ 0.7（領域分布接近均勻）

| 應對 | 說明 |
|------|------|
| **memory_action** | CLARIFY |
| **retrieval_action** | DUAL_OR_CLARIFY |
| **LLM 行為** | 反問使用者以縮小範圍 |
| **domain 處理** | 融合上一輪 + 當前輪的領域分布 |

```
用戶：那怎麼辦？
系統：您是想了解關於「粗大動作」的訓練建議，還是想知道其他方面的狀況呢？
```

---

### 4.2 多領域查詢（is_multi_domain = True）

**觸發條件：** active_domains 有 2 個以上領域

| 應對 | 說明 |
|------|------|
| **scope** | S_multi_domain |
| **LLM 行為** | 分段回答各領域、交叉比較 |
| **檢索** | 同時檢索多個 subdomain |

```
用戶：粗大動作跟精細動作哪個比較好？
系統：（分別說明兩個領域的表現，再做比較）
```

---

### 4.3 首輪對話（turn_index = 0）

| 應對 | 說明 |
|------|------|
| **context_sim** | 強制為 0（無前文） |
| **memory_action** | REFRESH（無歷史可延續） |
| **prev_context** | 無 |
| **LLM 行為** | 完整回答，不假設前文語境 |

---

### 4.4 無報告狀態

| 應對 | 說明 |
|------|------|
| **檢索** | 跳過 Neo4j 個案檢索 |
| **system_prompt** | 切換為「尚無報告」模板 |
| **LLM 行為** | 引導上傳報告，提供一般性衛教知識 |

```
系統：目前尚未上傳評估報告，建議您先上傳報告以獲得個人化的分析。
      如果您有一般性問題，我也可以提供一些參考資訊。
```

---

### 4.5 閒聊 / 感謝 / 打招呼

| 應對 | 說明 |
|------|------|
| **判斷** | 字數 ≤ 10 + 含閒聊關鍵字 |
| **路徑** | 跳過 DST + 檢索 + RL |
| **回覆** | `generate_chitchat()` 快速回覆 |
| **延遲** | < 1 秒 |

```
用戶：謝謝你的回答
系統：不客氣！有任何問題隨時可以問我。
```

---

### 4.6 領域外查詢（Out-of-Domain）

**觸發條件：** 所有領域的 cosine similarity < 0.30

| 應對 | 說明 |
|------|------|
| **LLM 行為** | 禮貌拒絕 + 引導回早療主題 |
| **不檢索** | 不進入 RAG 流程 |

```
用戶：今天天氣好嗎？
系統：我是早療評估報告的諮詢助手，主要可以幫您解讀報告內容、
      提供訓練建議和查詢在地資源。請問有什麼我可以幫您的嗎？
```

---

## 五、prev_context 攜帶規則

上一輪的檢索結果（top-3）在特定條件下會作為「前次對話參考」傳入 LLM。

| semantic_flow | context_sim | 是否攜帶 | 說明 |
|---------------|-------------|----------|------|
| continue | 任意 | 攜帶 | 延續話題，前文一定相關 |
| shift_soft | ≥ 0.45 | 攜帶 | 話題有關聯，前文可輔助 |
| shift_soft | < 0.45 | 不帶 | 關聯太弱，避免干擾 |
| shift_hard | 任意 | 不帶 | 完全不同話題 |

**攜帶時的處理：**
- 去重：排除已在當前輪檢索結果中出現的節點
- 標記：在 prompt 中以「前次對話參考」標籤明確區分，提示 LLM 以本次結果為主

---

## 六、RL Agent 決策對照表

### MemoryAgent（記憶策略）

| 決策 | 效果 | 典型場景 |
|------|------|---------|
| **STAY** | 沿用上一輪的 top_domain + active_domains | 同主題追問 |
| **REFRESH** | 更新為當前偵測的領域 | 換話題 |
| **CLARIFY** | 保留舊領域，標記模糊 | 指代不清 |

### PlanningAgent（區塊選擇）

| Section | 包含內容 | 高機率觸發的任務 |
|---------|---------|-----------------|
| assessment | 評估工具、分數、正式評量 | A, B, D, G, L, N |
| observation | 臨床觀察紀錄 | C, D, F |
| training | 訓練方向、治療計畫 | E, F, L |
| suggestion | 居家建議、推薦事項 | E, F, G |
| community_resources | MySQL 社區資源 | H, K |
| external_gpt | 外部通用知識 | I, J, M |

### RerankerAgent（排序權重）

| 權重 | 高值時的效果 | 適用場景 |
|------|-------------|---------|
| w_semantic | 語義相似度主導排序 | 精確問題（問特定分數） |
| w_structural | 結構標籤加成主導 | 類別導向（問所有觀察） |
| w_context | 上下文路徑加成主導 | 延續對話（追問同領域） |

---

## 七、Retrieval Action 決策矩陣

| C 值 \ MT 值 | MT 高 (≥0.50) | MT 中 (0.30~0.50) | MT 低 (<0.30) |
|--------------|---------------|-------------------|---------------|
| **C 高 (≥0.55)** | NARROW_GRAPH | CONTEXT_FIRST | CONTEXT_FIRST |
| **C 中 (0.45~0.55)** | WIDE_IN_DOMAIN | WIDE_IN_DOMAIN | DUAL_OR_CLARIFY |
| **C 低 (<0.45)** | WIDE_IN_DOMAIN | DUAL_OR_CLARIFY | DUAL_OR_CLARIFY |

**特殊覆蓋：**
- Task H/K + 有地區 → LOCAL_RESOURCE_SEARCH（優先於上表）
- Task H/K + 無地區 → LOCAL_RESOURCE_CLARIFY（優先於上表）

---

## 八、LLM 生成參數依情境調整

| 情境 | temperature | max_tokens | context_format | response_style |
|------|-------------|------------|----------------|----------------|
| continue + 非模糊 | 0.7 | 1500 | concise | 延續式 |
| shift_soft | 0.8 | 1800 | detailed | 過渡式 |
| shift_hard | 1.0 | 2000 | detailed | 獨立式 |
| 模糊查詢 | 0.6 | 1000 | concise | 釐清式 |
| Task M (情緒支持) | 1.0 | 2000 | concise | 同理心式 |
| Task B (分數解讀) | 0.5 | 1500 | structured | 精確式 |
| Task H/K (資源查詢) | 0.7 | 2000 | structured | 資訊式 |

---

## 九、完整情境範例

### 範例 1：同主題深入追問

```
[Turn 1] 用戶：粗大動作的標準分數是多少？
         DST:  task=B, domain=粗大動作, flow=shift_hard (首輪)
         Action: WIDE_IN_DOMAIN → 檢索粗大動作 assessment
         系統：根據報告，孩子的粗大動作標準分數為 42 分...

[Turn 2] 用戶：那百分等級呢？
         DST:  task=B, domain=粗大動作, C=0.82, MT=0.91 → flow=continue
         Memory: STAY
         Action: NARROW_GRAPH → 限縮粗大動作 assessment
         prev_context: 攜帶 Turn 1 的 top-3
         系統：粗大動作的百分等級(PR)為 21...

[Turn 3] 用戶：這樣算落後嗎？
         DST:  task=G, domain=粗大動作, C=0.76, MT=0.88 → flow=continue
         Memory: STAY
         QueryRewriter: 「粗大動作PR 21是否落後同齡」
         系統：PR 21 代表在同年齡中排在約第 21 百分位...
```

### 範例 2：話題自然轉換

```
[Turn 1] 用戶：精細動作表現怎麼樣？
         DST:  task=C, domain=精細動作, flow=shift_hard (首輪)
         系統：根據觀察，孩子在精細動作方面...

[Turn 2] 用戶：那這跟他粗大動作有關係嗎？
         DST:  task=D, domain=[精細動作, 粗大動作], C=0.51, MT=0.42 → flow=shift_soft
         Memory: REFRESH
         prev_context: 攜帶（context_sim=0.51 ≥ 0.45）
         系統：精細動作和粗大動作確實有關聯...（帶入上輪精細的背景）

[Turn 3] 用戶：那在家可以怎麼練？
         DST:  task=E, domain=粗大動作, C=0.62, MT=0.65 → flow=continue
         Memory: STAY
         系統：在家可以透過以下方式練習粗大動作...
```

### 範例 3：硬切換到資源查詢

```
[Turn 1] 用戶：他的認知分數正常嗎？
         DST:  task=B, domain=認知功能
         系統：認知功能的標準分數為...

[Turn 2] 用戶：台北市有推薦的早療機構嗎？
         DST:  task=H, region=台北市, C=0.12, MT=0.08 → flow=shift_hard
         Memory: REFRESH
         Action: LOCAL_RESOURCE_SEARCH → 查詢 MySQL
         prev_context: 不帶
         系統：台北市的早療相關機構有：1. XX復健診所...
```

### 範例 4：模糊查詢處理

```
[Turn 1] 用戶：他的粗大動作分數...
         系統：（回覆粗大動作分數）

[Turn 2] 用戶：嗯那怎麼辦？
         DST:  entropy=0.78 (高), is_ambiguous=True
         Memory: CLARIFY
         Action: DUAL_OR_CLARIFY
         系統：您是想了解如何在家練習粗大動作，還是想知道是否需要安排物理治療呢？
```
