# 開發注意事項（Dev Notes）

## 1. 基礎設施

所有服務指向 `192.168.150.136`：

| 服務 | Port | 說明 |
|------|------|------|
| MySQL | 3306 | `early_intervention_db_v2` |
| Neo4j | 7687 (Bolt) | `neo4j` / `password` |
| vLLM | 8000 | `Qwen/Qwen3-4B-Instruct-2507` |
| Embedding | 8080 | `/embed` 端點 |

設定集中在 `config.py`，可透過 `MYSQL_HOST` / `MYSQL_DATABASE` 環境變數覆蓋。Neo4j、LLM、Embedding 目前無環境變數覆蓋，搬遷需改 `config.py`。

---

## 2. 兒童身份識別

- 以 `access_code`（8 位英數碼）識別兒童個案
- 未登入使用者上傳 PDF 時自動建立新兒童
- `session['active_child_id']` 是所有對話的核心身份依據
- **doc_id 格式**：`v7_report_{report.id}_{child_id}`，三處使用需保持一致：
  1. `app.py:process_report_task()` — 建立
  2. `app.py:chat()` — 查詢
  3. `dialogue_manager.py:get_response()` — 傳入 RAG

---

## 3. 對話歷史格式

歷史檔案已更新為 dict 格式，需向後相容：

```json
{
  "messages": [
    {"role": "user", "content": "...", "id": "uuid"},
    {"role": "assistant", "content": "...", "id": "uuid"}
  ],
  "last_retrieved_context": [
    {"id": "node_id", "label": "Observation", "text": "...", "score": 0.85}
  ]
}
```

**注意**：舊格式是純 list。所有讀取歷史的地方都已加入相容處理：
- `dialogue_manager.py` — 載入歷史
- `app.py:get_chat_history()` — API 回傳
- `app.py:submit_feedback()` — 回饋寫入

`last_retrieved_context` 儲存每輪的 top-3 檢索結果，供下一輪 prev_context 攜帶使用。

---

## 4. query_vector 共用機制

`dialogue_manager.py` 入口處單次編碼 `query_vector`，透過參數傳遞給所有下游：

```
encoder.encode(user_input) → query_vector
    ├── SemanticFlowClassifier.predict(query_vec=)
    │     ├── DomainRouter.predict(query_vec=)
    │     └── TaskScopeClassifier.predict_task(query_vec=)
    ├── StrategyMapper.map_dst_to_strategy(query_vector=)
    └── Reranker.rerank(query_vec=)
```

**修改任何模組的 encode 邏輯時**，確認是否已接受 `query_vec` 參數。所有模組都有 fallback：`if query_vec is None: query_vec = self.encoder.encode(text)`。

---

## 5. RL Agent 特徵維度

修改輸入/輸出維度時需同步更新，否則 `torch.mm()` 崩潰：

### RerankAgent
- 輸入 33 維 = 14（任務 A~N one-hot）+ 11（領域 one-hot）+ 3（範圍 one-hot）+ 5（連續特徵）
- `RLAgentManager.tasks` / `.domains` / `.scopes` 清單

### PlanningAgent
- 輸入 20 維 = 6（語義分）+ 13（任務 one-hot，不含 N）+ 1（entropy）
- `PlanningAgent.section_labels` / `.task_list` 清單
- `TASK_SECTION_MAP` 定義各任務的 section 預設權重

### MemoryAgent
- 輸入：對話狀態特徵
- 輸出：STAY (0) / REFRESH (1) / CLARIFY (2)

---

## 6. 臨床知識圖譜的兩種角色

`knowledge_graph_extra/` 有兩個用途：

1. **靜態 JSON 知識庫**（`json/*.json`）：`ClinicalBridgeAnalyzer` 直接讀取，向量比對，**不需要 Neo4j**
2. **匯入 Neo4j**（`src/import_*.py`）：選用功能，支援圖遍歷

目前實作以**本地 JSON + 向量計算**為主。

---

## 7. LLM 訊息格式

vLLM 要求 user/assistant 嚴格交替。`llm_generator.py:_normalize_messages()` 負責自動修正。

---

## 8. 已知問題

| 問題 | 位置 | 嚴重程度 |
|------|------|---------|
| PDF 解析為同步阻塞 | `app.py:upload_file()` | 中 |
| 未登入使用者 user_id fallback 為 1 | `app.py:chat()` | 低 |
| `pcst_solver` 預設 `enable_pcst=False` | `retrieval_module_v2/__init__.py` | 低 |
| `_fetch_gpt_knowledge()` 為 Placeholder | `execution_engine.py` | 低 |
| `dialogue_states/` JSON 為本機路徑 | 多實例部署時衝突 | 中（若需水平擴展）|

---

## 9. 訓練腳本

```bash
# 從 MySQL 讀取對話 → 統一離線訓練
python rl_pipeline/scripts/unified_train_db.py

# 三個 Agent 預訓練（含 TASK_SECTION_MAP）
python rl_pipeline/scripts/pretrain_agents.py

# 生成多樣化測試對話
python rl_pipeline/scripts/generate_varied_questions.py

# 自動測試 Bot（需先啟動 Flask）
python rl_pipeline/scripts/auto_query_bot.py
```

模型覆寫至 `rl_pipeline/agents/*/models/*.pth`，**重啟 Flask 後**新權重生效。

---

## 10. 前端

- 單頁應用（SPA），`templates/index.html` + `static/css/style.css`
- 原生 JS + Fetch API，無框架
- PDF 上傳用 `FormData`，對話用 JSON body
- `message_id` 對應 `msg_uuid`（UUID v4），用於回饋
