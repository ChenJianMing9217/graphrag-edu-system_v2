# 系統架構文件

## 1. 整體系統架構

```
┌─────────────────────────────────────────────────────────────────────┐
│                         使用者端 (Browser)                           │
│                     templates/index.html                             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ HTTP (REST API)
┌──────────────────────────▼──────────────────────────────────────────┐
│                    Flask Web App (app.py, port 5001)                 │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │
│  │ /api/upload  │  │  /api/chat   │  │ /api/login_with_code etc. │  │
│  └──────┬──────┘  └──────┬───────┘  └───────────────────────────┘  │
│         │                │                                           │
│  ┌──────▼──────┐  ┌──────▼──────────────────────────────────────┐  │
│  │ IEPPipeline │  │           DialogueManager                    │  │
│  │  (pdf_parser)│  │  (dialogue_manager.py)                      │  │
│  └──────┬──────┘  └──────┬───────────────────────────────────────┘  │
└─────────┼────────────────┼────────────────────────────────────────┘
          │                │
          ▼                ▼
  [PDF 解析管線]    [對話回覆管線]（詳見第2節）
```

---

## 2. 對話回覆管線（核心資料流）

使用者每次傳入問題，`DialogueManager.get_response()` 依序執行：

```
使用者輸入 (user_input)
    │
    ▼
【B0. 閒聊快速攔截】
  字數 ≤ 10 且含「你好/謝謝/嗨」等關鍵字
  └── 直接回覆，跳過所有下游模組（< 1 秒）
    │
    ▼ （非閒聊才繼續）
【B1. 文字編碼（一次性）】
  query_vector = encoder.encode(user_input)
  └── 此向量全流程共用，傳入 DST / Retrieval / Reranker，不重複編碼
    │
    ▼
【B2. 載入狀態】
  flow_classifier.load_state(u_id, c_id)
  └── 從 dialogue_states/ 讀入前幾輪 DST 狀態
  載入歷史紀錄（含 messages + last_retrieved_context）
    │
    ▼
【B3. DST 分析】SemanticFlowClassifier.predict(user_input, query_vec=query_vector)
  ├── TaskScopeClassifier       → task_label (A~N) + 分布
  ├── DomainRouter              → top_domain, active_domains, entropy
  ├── ContextSimilarity         → 與前輪語意相似度 (C 值)
  ├── MultiTopicTracker         → 主題延續度 (MT 值)
  └── DSTPolicyDecision         → retrieval_action, semantic_flow, memory_action
    │
    ▼
【C. RL 加權預測】RLAgentManager.predict_weights(...)
  ├── 輸入：33 維特徵（任務 one-hot + 領域 one-hot + 連續特徵）
  └── 輸出：w_semantic, w_structural, w_context（三維 Dirichlet 分布）
    │
    ▼
【D. RAG 檢索】RetrievalModuleV2.retrieve(...)
  │
  ├── [查詢改寫] QueryRewriter.rewrite()
  │     ├── Query Condensation（continue 時，將代名詞解析為獨立問句）
  │     └── 生成最多 2 個改寫版本，後用 RRF 融合
  │
  ├── [第一階段：個案資料檢索]
  │     StrategyMapper.map_dst_to_strategy()（複用 query_vector）
  │     └── PlanningAgent (RL) 選擇要抓哪些 section
  │     ExecutionEngine.execute_initial()
  │     └── Neo4j Cypher + MySQL 社區資源
  │
  ├── [中間重排] Reranker.rerank()（複用 query_vector + batch encode）
  │     ├── 初步篩選
  │     └── 分類萃取：Observation / TrainingDirection / Recommendation 各取 top-4
  │
  ├── [第二階段：臨床知識增強]（複用第一階段的 primary_strategy）
  │     ExecutionEngine.execute_enrichment()
  │     └── ClinicalBridgeService → 臨床常模 JSON
  │
  └── [最終整合] ExternalGPT 置頂 → 臨床增強 → 排序後個案資料
    │
    ▼
【E. prev_context 條件攜帶】
  ├── continue → 攜帶上一輪 top-3
  ├── shift_soft + context_sim ≥ 0.45 → 攜帶（去重）
  └── 其他 → 不帶
    │
    ▼
【F. LLM 生成】LLMGenerator.generate_response(...)
  ├── PromptManager.build_user_prompt()（含 prev_context）
  ├── 依任務/流向選擇 system_prompt 模板與生成參數
  └── vLLM API call → 回傳 response
    │
    ▼
【G. 更新狀態 & 儲存】
  ├── ContextSimilarity.update()
  ├── 寫入 dialogue_states/ JSON（DST 狀態 + 對話歷史 + last_retrieved_context）
  └── 寫入 MySQL ChatMessage（含 flow_state, retrieval_info JSON）
```

---

## 3. PDF 解析管線

```
上傳 PDF → process_report_task(report_id, file_path, child_id)
    │
    ├── Step 1: pdf_parser.process_iep_pdf()
    │     └── pdfplumber/pdfminer 解析 → 結構化 dict
    │
    ├── Step 2: 封存 JSON → uploads/json_archives/{child_id}_{timestamp}.json
    │
    └── Step 3: Neo4jImporter.import_iep(data, report_id)
          └── Report → Domain → Subdomain → CategoryHub → Item
```

---

## 4. Neo4j 圖譜結構

### 個案報告圖

```
(:Report {id: "v7_report_{report.id}_{child_id}"})
  ├──[:HAS_DOMAIN]──> (:Domain)
  │     └──[:HAS_SUBDOMAIN]──> (:Subdomain)
  │           └──[:HAS_ASSESSMENT_TOOLS|HAS_OBSERVATIONS|HAS_TRAINING_PLAN|HAS_RECOMMENDATIONS|HAS_SCORES|HAS_FORMAL_ASSESSMENTS]──> (:CategoryHub)
  │                 └──[:USED_TOOL|OBSERVED|RECOMMENDED|TRAINED_BY|HAS_VALUE]──> (:AssessmentTool|Observation|Recommendation|TrainingDirection|Score)
  │                       └──[:HAS_SUB_ITEM]──> (:SubItem)
  ├──[:HAS_SUMMARY]──> (:Summary) → (:CategoryHub) → (:Item)
  └──[:HAS_META]──> (:Meta {patient_name, gender, age, ...})
```

### 臨床知識圖譜（knowledge_graph_extra/json/）

由靜態 JSON 定義，ClinicalBridgeAnalyzer 以向量比對方式使用：
- `abilities_v3.json` — 發展能力清單
- `milestones_v3.json` — 月齡發展里程碑
- `observation_indicators_v3.json` — 臨床觀察指標
- `training_strategies.json` / `training_activities.json` — 訓練策略與活動

---

## 5. 資料庫模型（MySQL）

| 資料表 | 主要欄位 | 說明 |
|--------|---------|------|
| `user` | id, username, email, role | 使用者 |
| `child` | id, name, access_code, creator_id | 兒童個案 |
| `report` | id, filename, file_path, child_id, assessment_date | IEP 報告 |
| `chat_message` | id, session_id, msg_uuid, user_id, child_id, message, flow_state, retrieval_info, feedback_value | 對話紀錄 |
| `subsidy_program` | id, county, program_name, amount, ... | 早療補助資訊 |

---

## 6. RL 管線架構

三個獨立 RL Agent，各自學習不同決策：

| Agent | 輸入 | 輸出 | 訓練目標 |
|-------|------|------|---------|
| **PlanningAgent** | 語義分 + 任務 one-hot + entropy | 6 維 sigmoid（section 開關） | 選擇最相關的報告區塊 |
| **RerankAgent** | 任務/領域/範圍 one-hot + 連續特徵 (33維) | 3 維 Dirichlet（語義/結構/上下文權重） | 最佳化重排序 |
| **MemoryAgent** | 對話狀態特徵 | STAY / REFRESH / CLARIFY | 控制對話記憶更新 |

獎勵來源：`MultiAgentRewardJudge`（LLM-as-Judge 獨立評分 1~5）+ 使用者回饋 (+1/-1)。

---

## 7. 模組依賴關係

```
app.py
  ├── dialogue_manager.py
  │     ├── dialogue_state_module/semantic_flow_module_v2.py
  │     │     ├── domain_router.py ──> embedding.py
  │     │     ├── context_similarity.py
  │     │     ├── multi_topic_tracker.py
  │     │     ├── task_scope_classifier.py
  │     │     └── dst_policy.py
  │     ├── retrieval_module_v2/
  │     │     ├── strategy_mapper.py ──> topic_ontology.py
  │     │     ├── execution_engine.py ──> knowledge_graph_extra/clinical_api.py
  │     │     ├── reranker.py（batch encode_many）
  │     │     ├── query_rewriter.py ──> llm_generate_module/
  │     │     └── graph_client.py / mysql_client.py
  │     ├── llm_generate_module/
  │     │     ├── llm_generator.py（支援 prev_context）
  │     │     └── prompt_manager.py（動態生成參數）
  │     └── rl_pipeline/agents/
  │           ├── reranker/rerank_agent.py
  │           ├── planner/planning_agent.py
  │           └── memory/memory_agent.py
  └── pdf_parser/
        ├── pdf_processor_main.py
        ├── pdf_parser.py
        └── neo4j_importer.py
```

---

## 8. 效能最佳化設計

| 最佳化項目 | 說明 |
|-----------|------|
| 閒聊快速路徑 | 字數 ≤ 10 + 關鍵字匹配 → 跳過所有下游模組 |
| query_vector 共用 | 入口編碼一次，傳入 DST → DomainRouter → TaskClassifier → StrategyMapper → Reranker |
| Batch encode | Reranker 使用 `encode_many()` 一次編碼所有候選節點 |
| Strategy 複用 | 第一次查詢的 primary_strategy 直接供 enrichment 階段複用 |
| 查詢改寫限制 | 最多 2 個改寫版本，減少 LLM 呼叫 |
