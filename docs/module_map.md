# 模組地圖（Module Map）

## API 路由

| 路由 | 方法 | 說明 |
|------|------|------|
| `/` | GET | 主頁 (前端 SPA) |
| `/api/upload` | POST | 上傳 IEP PDF，觸發解析與圖譜建置 |
| `/api/chat` | POST | 主對話介面，回傳 LLM 生成回覆 |
| `/api/new_chat` | POST | 重置對話狀態 |
| `/api/login_with_code` | POST | 以 access_code 登入兒童個案 |
| `/api/logout` | POST | 登出 |
| `/api/session_status` | GET | 查詢登入狀態 |
| `/api/chat_history` | GET | 取得對話歷史 |
| `/api/feedback` | POST | 提交回饋（+1/-1） |

---

## DST 模組（`dialogue_state_module/`）

| 檔案 | 類別 | 功能 |
|------|------|------|
| `semantic_flow_module_v2.py` | `SemanticFlowClassifier` | DST 主協調器；`predict(user_input, query_vec)` 輸出 `FlowResult` |
| `domain_router.py` | `DomainRouter` | 11 領域的 cosine + softmax 分類；`predict(text, query_vec)` |
| `dst_policy.py` | `decide_policy()` | C x MT 四象限策略 → `PolicyDecision` |
| `task_scope_classifier.py` | `TaskScopeClassifier` | 任務 A~N 分類 + 範圍分類；`predict_task(text, query_vec)` |
| `context_similarity.py` | `ContextSimilarity` | 滑動視窗語意相似度 |
| `multi_topic_tracker.py` | `MultiTopicTracker` | 跨輪次主題延續追蹤 |
| `embedding.py` | `TextEncoder` | 遠端 Embedding API + `encode_many()` 批次編碼 |
| `domain_anchors.py` | — | 領域錨點定義（從 `config/domain_anchors.json` 載入） |
| `state_persistence.py` | — | 狀態檔案讀寫 |
| `utils/region_extractor.py` | `extract_region()` | 台灣行政區偵測 |

### DST 輸出（FlowResult）

```python
FlowResult
├── turn_index: int
├── task_label: str              # A~N
├── task_dist: Dict[str, float]
├── scope_label: str             # S_overview | S_domain | S_multi_domain
├── detected_region: str
├── domain_analysis
│   ├── top_domain, top_prob, entropy
│   ├── distribution: Dict[str, float]  # 11 維
│   └── active_domains: List[str]
├── context_analysis
│   ├── similarity_score: float         # C 值
│   └── source: str
├── topic_analysis
│   ├── overlap_score: float            # MT 值
│   └── tv_distance: float
└── policy_decision
    ├── retrieval_action: str    # NARROW_GRAPH | CONTEXT_FIRST | WIDE_IN_DOMAIN | DUAL_OR_CLARIFY | LOCAL_RESOURCE_*
    ├── semantic_flow: str       # continue | shift_soft | shift_hard
    ├── memory_action: str       # STAY | REFRESH | CLARIFY
    └── is_ambiguous: bool
```

---

## 檢索模組（`retrieval_module_v2/`）

| 檔案 | 類別 | 功能 |
|------|------|------|
| `__init__.py` | `RetrievalModuleV2` | 三階段 RAG 入口；RRF 融合；primary_strategy 複用 |
| `strategy_mapper.py` | `StrategyMapper` | DST → `SearchStrategy`（含 PlanningAgent 決策）|
| `execution_engine.py` | `ExecutionEngine` | Neo4j Cypher / MySQL / 臨床常模執行 |
| `reranker.py` | `Reranker` | 多訊號重排；接受 `query_vec` + `encode_many()` 批次編碼 |
| `query_rewriter.py` | `QueryRewriter` | Query Condensation + 改寫（最多 2 版本）|
| `pcst_solver.py` | `PCSTSolver` | 子圖發現（目前 `enable_pcst=False`）|
| `graph_client.py` | `GraphClient` | Neo4j Driver 封裝 |
| `mysql_client.py` | `MySQLResourceClient` | MySQL 社區資源查詢 |
| `topic_ontology.py` | `TopicOntology` | 報告區塊語意本體 |
| `types.py` | — | `CandidateNode`, `SearchStrategy`, `SearchOperation`, `SearchOperationType` |

### SearchOperationType

| 類型 | 對應動作 |
|------|---------|
| `SUBDOMAIN_FETCH` | 依子領域從 Neo4j 抓取 |
| `SUMMARY_FETCH` | 抓取報告摘要 |
| `META_FETCH` | 抓取個案基本資料 |
| `MYSQL_RESOURCE_FETCH` | MySQL 社區資源 |
| `CLINICAL_FETCH` | 臨床常模查詢 |
| `GPT_FETCH` | 外部 GPT 知識 |

---

## LLM 生成模組（`llm_generate_module/`）

| 檔案 | 類別 | 功能 |
|------|------|------|
| `llm_generator.py` | `LLMGenerator` | `generate_response()` 支援 `prev_context` 參數 |
| `prompt_manager.py` | `LLMPromptManager` | 動態生成參數（依 semantic_flow / task / scope）+ `build_user_prompt(prev_context)` |
| `prompt_config.json` | — | 各維度的 prompt 模板與參數覆蓋設定 |

### 生成參數決定優先順序（高 → 低）

1. 任務特定覆蓋（task_config: A~N）
2. 特殊情境覆蓋（多領域、模糊）
3. Retrieval action 覆蓋
4. Semantic flow 基礎設定
5. 預設值

---

## PDF 解析模組（`pdf_parser/`）

| 檔案 | 功能 |
|------|------|
| `pdf_processor_main.py` | `IEPPipeline`：解析 → 封存 → 匯入 |
| `pdf_parser.py` | PDF 文字提取與結構化 |
| `neo4j_importer.py` | Cypher MERGE 匯入 Neo4j |

---

## 臨床知識圖譜（`knowledge_graph_extra/`）

| 路徑 | 說明 |
|------|------|
| `clinical_api.py` | `ClinicalBridgeService`：對外 API |
| `src/clinical_bridge_analyzer.py` | 核心分析器（向量比對能力/里程碑）|
| `json/*.json` | 靜態知識庫（能力、里程碑、觀察指標、訓練策略）|
| `rebuild_all_v4.py` | 一鍵重建臨床知識圖譜 |

---

## RL 管線（`rl_pipeline/`）

| 路徑 | 說明 |
|------|------|
| `agents/reranker/rerank_agent.py` | `RLAgentManager`：33 維輸入 → 3 維 Dirichlet 權重 |
| `agents/planner/planning_agent.py` | `PlanningAgent`：語義分 + 任務 → 6 section 開關 |
| `agents/memory/memory_agent.py` | `MemoryAgent`：STAY / REFRESH / CLARIFY |
| `shared/reward_judge.py` | `MultiAgentRewardJudge`：LLM-as-Judge 評分 |
| `scripts/unified_train_db.py` | 從 MySQL 讀取對話 → 統一離線訓練 |
| `scripts/pretrain_agents.py` | 三個 Agent 預訓練 |
| `scripts/auto_query_bot.py` | 自動測試 Bot（模擬對話收集訓練資料）|
| `scripts/generate_varied_questions.py` | 從種子題庫生成多樣化多輪測試對話 |
| `agents/*/models/*.pth` | 模型權重（重啟 Flask 後生效）|

---

## 持久化資料

| 資料 | 位置 | 格式 |
|------|------|------|
| DST 狀態 | `dialogue_states/user_{u}_child_{c}_state.json` | JSON |
| 對話歷史 | `dialogue_states/user_{u}_child_{c}_history.json` | `{"messages": [...], "last_retrieved_context": [...]}` |
| 上傳 PDF | `uploads/{timestamp}_{filename}` | PDF |
| 解析後 JSON | `uploads/json_archives/{child_id}_{timestamp}.json` | JSON |
| RL 訓練歷史 | `rl_pipeline/logs/training_history.json` | JSON |
| RL 模型 | `rl_pipeline/agents/*/models/*.pth` | PyTorch |
