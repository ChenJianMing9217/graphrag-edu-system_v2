# 早療 AI 諮詢系統 (app_v7)

## 專案目的

以早期療育為場景的 AI 對話諮詢平台。使用者為**家長/照顧者**與**治療師**。

核心流程：
1. 上傳兒童 IEP 評估報告 PDF → 自動解析並存入 Neo4j 知識圖譜
2. 透過對話提問 → DST 意圖理解 + RAG 檢索 + 臨床常模對接 + LLM 生成回覆
3. 三個 RL Agent 動態最佳化：記憶策略、檢索區塊選擇、排序權重

---

## 環境需求

- Python 3.11+
- MySQL（`early_intervention_db_v2`）
- Neo4j（Bolt port 7687）
- vLLM（OpenAI 相容 API，Qwen3-4B-Instruct，port 8000）
- Embedding 服務（HTTP API，port 8080）

## 安裝與啟動

```bash
pip install flask flask-sqlalchemy flask-login werkzeug openai pymysql neo4j torch numpy
python init_db.py          # 初始化 MySQL
python app.py              # 啟動 http://localhost:5001
```

設定在 `config.py`，環境變數：`MYSQL_HOST`、`MYSQL_DATABASE`。

---

## 目錄結構

```
app_v7/
├── app.py                        # Flask 入口、路由、DB Model
├── config.py                     # 集中設定
├── dialogue_manager.py           # 對話核心：DST → RL → RAG → LLM
│
├── dialogue_state_module/        # DST：領域/任務/流向分析
│   ├── semantic_flow_module_v2.py   # 主協調器
│   ├── domain_router.py             # 11 領域分類
│   ├── task_scope_classifier.py     # 任務 A~N + 範圍分類
│   ├── dst_policy.py                # C x MT 四象限策略
│   ├── context_similarity.py        # 上下文相似度
│   ├── multi_topic_tracker.py       # 主題延續追蹤
│   └── embedding.py                 # TextEncoder（支援 encode_many）
│
├── retrieval_module_v2/          # 三階段 RAG
│   ├── __init__.py               # 入口：改寫 → 初步檢索 → 臨床增強 → 整合
│   ├── strategy_mapper.py        # DST → SearchStrategy
│   ├── execution_engine.py       # Neo4j / MySQL / 臨床常模
│   ├── reranker.py               # 多訊號重排（batch encode）
│   ├── query_rewriter.py         # 改寫 + RRF 融合
│   └── types.py                  # CandidateNode 等
│
├── llm_generate_module/          # LLM 回覆生成
│   ├── llm_generator.py          # vLLM 呼叫（支援 prev_context）
│   └── prompt_manager.py         # 動態 prompt + 生成參數
│
├── pdf_parser/                   # PDF 解析 → Neo4j
├── knowledge_graph_extra/        # 臨床知識圖譜（JSON + 向量比對）
│
├── rl_pipeline/                  # 強化學習
│   ├── agents/                   # PlanningAgent / RerankAgent / MemoryAgent
│   ├── scripts/                  # 訓練、測試、資料生成腳本
│   └── shared/                   # 獎勵評估
│
└── docs/                         # 文件
    ├── README.md                 # 本文件
    ├── architecture.md           # 完整架構與資料流
    ├── module_map.md             # 模組清單與資料結構
    ├── dev_notes.md              # 開發注意事項
    ├── system_response_patterns.md  # 各情境的系統應對方式
    └── offline_evaluation_design.md # 離線評估資料集設計
```

---

## 主要模組

| 模組 | 責任 |
|------|------|
| `dialogue_manager.py` | 核心協調：閒聊攔截 → 編碼 → DST → RL → RAG → LLM |
| `dialogue_state_module/` | 理解意圖 (A~N)、領域 (11種)、流向 (continue/shift/hard) |
| `retrieval_module_v2/` | 三階段 RAG + RRF + batch rerank + 臨床增強 |
| `llm_generate_module/` | 依情境動態調整 prompt 與生成參數 |
| `rl_pipeline/` | 三個 RL Agent 的定義、訓練、獎勵評估 |

---

## 文件導覽

| 文件 | 適合誰看 | 內容 |
|------|---------|------|
| [architecture.md](architecture.md) | 全體開發者 | 完整資料流、Neo4j 圖譜結構、RL 架構 |
| [module_map.md](module_map.md) | 接手維護者 | 每個檔案的類別/方法清單、資料結構定義 |
| [dev_notes.md](dev_notes.md) | 日常開發 | 設定、踩坑、已知問題、訓練指令 |
| [system_response_patterns.md](system_response_patterns.md) | 產品/測試 | 各種使用者情境的系統決策邏輯與回覆方式 |
| [offline_evaluation_design.md](offline_evaluation_design.md) | RL/ML 工程師 | 離線評估資料集 schema 與標註原則 |
