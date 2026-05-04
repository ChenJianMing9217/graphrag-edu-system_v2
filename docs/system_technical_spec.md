# 早療 AI 諮詢系統 (app_v7) — 完整技術規格書

> 版本：v7.2 | 更新日期：2026-04-20

---

## 目錄

1. [系統總覽與架構](#1-系統總覽與架構)
2. [基礎設施與模型](#2-基礎設施與模型)
3. [單輪對話完整流程 (A → G)](#3-單輪對話完整流程-a--g)
4. [DST 對話狀態追蹤 — 數學細節](#4-dst-對話狀態追蹤--數學細節)
5. [RL Agent 群 — 網路結構與訓練](#5-rl-agent-群--網路結構與訓練)
6. [RAG 檢索管線 (Retrieval Pipeline)](#6-rag-檢索管線-retrieval-pipeline)
7. [Slot Filling 槽位機制](#7-slot-filling-槽位機制)
8. [LLM 生成模組](#8-llm-生成模組)
9. [狀態持久化與跨輪記憶](#9-狀態持久化與跨輪記憶)
10. [Embedding 快取機制](#10-embedding-快取機制)
11. [模組總表與檔案索引](#11-模組總表與檔案索引)

---

## 1. 系統總覽與架構

```
使用者輸入
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  DialogueManager (dialogue_manager.py) — 主協調器     │
│                                                       │
│  A. 載入狀態  ──────────────────────────────────┐     │
│  B0. 閒聊快篩                                     │     │
│  B1. Embedding 編碼 (query_vector)               │     │
│  B2. DST 分析 (SemanticFlowClassifier)           │     │
│  B2.5 Slot 回填偵測                               │     │
│  B3. Out-of-Domain 偵測                           │     │
│  B5. Slot 檢查                                    │     │
│  C. 報告 ID 確定                                  │     │
│  D. RAG 檢索 (RetrievalModuleV2)                 │     │
│  E. Context 組裝                                  │     │
│  F. LLM 生成回覆                                  │     │
│  G. 狀態保存                                      │     │
└─────────────────────────────────────────────────────┘
    │
    ▼
系統回覆
```

### 核心子系統

| 子系統 | 職責 | 主要檔案 |
|--------|------|----------|
| **DST (Dialogue State Tracking)** | 領域路由、上下文追蹤、主題延續、策略決策 | `dialogue_state_module/semantic_flow_module_v2.py` |
| **Task Classifier** | 14 類任務意圖分類 (A-N) | `dialogue_state_module/task_scope_classifier.py` |
| **Slot Tracker** | 槽位追蹤與回填偵測 | `dialogue_state_module/slot_tracker.py` |
| **Memory Agent (RL)** | 記憶策略決策 (STAY/REFRESH/CLARIFY) | `rl_pipeline/agents/memory/memory_agent.py` |
| **Rerank Agent (RL)** | 檢索結果重排權重 | `rl_pipeline/agents/reranker/rerank_agent.py` |
| **Planning Agent (RL)** | 知識區塊選擇決策 | `rl_pipeline/agents/planner/planning_agent.py` |
| **RAG Retrieval** | 多策略檢索與融合 | `retrieval_module_v2/` |
| **LLM Generator** | 基於 context 的回覆生成 | `llm_generate_module/llm_generator.py` |

---

## 2. 基礎設施與模型

### 2.1 服務端點

| 服務 | 地址 | 用途 |
|------|------|------|
| Embedding Server | `http://192.168.150.136:8080/embed` | 文本向量化 (1024 維) |
| LLM (vLLM) | `http://192.168.150.136:8000/v1` | 回覆生成 |
| Neo4j | `bolt://192.168.150.136:7687` | 知識圖譜 (報告結構化資料) |
| MySQL | `192.168.150.136:3306` | 關聯式資料庫 (個案/社區資源/補助) |

### 2.2 使用的模型

| 模型 | 類型 | 說明 |
|------|------|------|
| **Qwen3-4B-Instruct-2507** | LLM | 用於回覆生成、Query Rewriting |
| **Embedding Model** (TEI Server) | Embedding | 1024 維文本向量，用於語義比對 |
| **DialoguePolicyNet** | RL Policy | 9→32→32→3 全連接網路 (Memory Agent) |
| **RerankAgent** | RL Dirichlet Policy | 33→64→32→3 全連接網路 (Rerank Agent) |
| **PlanningPolicyNet** | RL Multi-label Policy | 21→64→64→6 全連接網路 (Planning Agent) |
| **PrototypeClassifier** | 零次分類 | 基於原型句向量的 cosine 分類器 (14 任務) |

---

## 3. 單輪對話完整流程 (A → G)

### Step A — 載入狀態

```python
flow_classifier.load_state(user_id, child_id)   # DST 狀態 (turn_index, prev_scope, prev_dist...)
slot_tracker.load_state(slot_state_raw)           # Slot 狀態 (pending_slot, filled_slots)
chat_history = load_history_json()                # 對話歷史 (最近 5 輪)
```

### Step B0 — 閒聊快篩

```
條件：len(input) ≤ 10  且  input 包含 ["你好", "謝謝", "再見", ...] 中任一關鍵字
動作：直接呼叫 LLM generate_chitchat()，跳過所有後續步驟
```

### Step B1 — Embedding 編碼

```python
query_vector = encoder.encode(user_input)   # → np.ndarray shape=(1024,)
```

全程共用此向量，避免重複呼叫 Embedding API。

### Step B2 — DST 分析 (SemanticFlowClassifier.predict)

DST 內部依序執行五層分析（見 [第 4 節](#4-dst-對話狀態追蹤--數學細節) 詳細數學）：

```
1. _analyze_domain()     → DomainAnalysis  (領域路由)
2. _analyze_context()    → ContextAnalysis (上下文相似度 C)
3. _analyze_topic()      → TopicAnalysis   (主題延續 MT, TV 距離)
4. _classify_task_only() → task_label, task_dist, secondary_tasks, entropy
5. _decide_policy()      → PolicyDecision  (semantic_flow, retrieval_action, memory_action)
6. _classify_scope()     → scope_label     (S_overview / S_domain / S_multi_domain)
```

輸出：`FlowResult` 包含所有分析結果。

### Step B2.5 — Slot 回填偵測

```
條件 A：len(input) ≤ 8  且  不含 "？" 或 "?"
條件 B：task_top_prob < 0.5  或  task_entropy > 0.5

若 pending_slot 存在  且  (A ∧ B)  →  繼承上輪任務 (intent = inherited_task)
若 pending_slot 存在  且  長句或含問號  →  清除 pending，視為新問題
```

### Step B3 — Out-of-Domain 偵測

```
_is_out_of_domain = (
    task_top_score < 0.30                                    # 條件 1：跟所有任務都遠
    OR (task_top_score < 0.40 AND task_top_prob < 0.25)     # 條件 2：稍遠且不確定
)
```

觸發時跳過 RAG，生成引導回覆。

### Step B5 — Slot 檢查

從各來源收集可用槽位值：

| 槽位 | 來源 |
|------|------|
| `region` | `extract_region()` 正則比對 (22 縣市) |
| `domain_focus` | DomainRouter top_domain (排除「整體概況」) |
| `ability_focus` | `extract_ability_focus()` 關鍵字比對 (11 能力類別) |
| `child_age` | DB Child.birth_date → age_months |
| `school_type` | `extract_school_type()` 關鍵字比對 (9 學校類型) |
| `time_range` | `extract_time_range()` 正則比對 (時間表達式) |
| `report_range` | 預設 "latest_2" |

檢查任務所需槽位 → 產生 `SlotCheckResult`（all_filled / has_missing / no_slots）。

### Step C — 報告 ID 確定

```
普通任務：doc_id = f"v7_report_{report_id}_{child_id}"
進步查詢 (N)：doc_id = [最近兩份報告 ID 的列表]
```

### Step D — RAG 檢索

1. **RL Rerank 權重預測**：`RLAgentManager.predict_weights(task, domain, scope, continuous)`
2. **檢索**：`RetrievalModuleV2.retrieve(query, turn_state, doc_id, rerank_config)`

（見 [第 6 節](#6-rag-檢索管線-retrieval-pipeline) 完整流程）

### Step E — Context 組裝

```python
retrieved_context = top-20 候選節點 → [{text, score, label, id, path, properties}, ...]

# 上輪 Context 帶入判斷
if semantic_flow == "continue":
    carry_over = True                                    # 主題延續 → 帶入上輪 top-3
elif semantic_flow in ("shift_soft", "shift_hard") and context_sim >= 0.45:
    carry_over = True                                    # 跨主題但語義相關 → 帶入
else:
    carry_over = False                                   # 乾淨切換 → 不帶
```

### Step F — LLM 生成回覆

```python
# 系統提示詞根據情境動態選擇：
# - 無報告 → 一般性知識回答 + 提示上傳
# - LOCAL_RESOURCE_CLARIFY → 詢問縣市
# - is_ambiguous → 引導釐清領域
# - 無檢索結果 → 醫學知識 + 引導提供生活細節
# - has_missing slots → 在結尾追加 Slot 追問

response = generator.generate_response(
    user_query, retrieved_context, chat_history,
    system_prompt, generation_config, prev_context
)
```

### Step G — 狀態保存

```python
context_similarity.update(user_input, response)     # 更新上下文向量
flow_classifier.save_state(user_id, child_id)       # DST 狀態持久化
slot_tracker.update_pending(intent, slot_result)     # Slot pending 更新

# 保存對話歷史 JSON：
{
    "messages": chat_history[-10:],                  # 最近 5 輪
    "last_retrieved_context": top-5 context 快照,    # 供下輪 carry-over
    "slot_state": slot_tracker.save_state()          # Slot 狀態
}
```

---

## 4. DST 對話狀態追蹤 — 數學細節

### 4.1 領域路由 (DomainRouter)

**領域列表** (10 個臨床領域 + 1 整體)：

```
整體概況, 粗大動作, 精細動作, 感覺統合, 口腔動作,
情緒行為與社會適應功能, 吞嚥功能, 口語理解, 口語表達, 說話, 認知功能
```

**Step 1 — 計算原始相似度分數**

對每個領域 d，使用 Max Pooling 取得最高相似度：

```
raw_score(d) = max { cosine_sim(query_vec, anchor_vec) | anchor_vec ∈ anchors(d) }
```

其中：
```
cosine_sim(a, b) = (a · b) / (‖a‖ × ‖b‖)
```

**Step 1.5 — 關鍵字 Boosting**

若 query 包含「整體概況」的高意圖關鍵字（整體、總覽、全部、總結...），則：
```
raw_score("整體概況") += 0.05
```

**Step 2 — Softmax 轉換為機率分布**

```
P(d) = softmax(raw_scores, temperature=0.04)

softmax(x_i, τ) = exp(x_i / τ) / Σ_j exp(x_j / τ)
```

> temperature = 0.04 極低，使分布非常尖銳（接近 argmax 但仍保有梯度資訊）。

**Step 3 — 計算歸一化熵**

```
H_norm = H(P) / H_max

H(P) = -Σ_d P(d) × ln P(d)
H_max = ln(|domains|)

H_norm ∈ [0, 1]
  0 → 非常集中（確定某個領域）
  1 → 非常均勻（完全不確定）
```

**Step 4 — 選擇 Active Domains**

```
active(d) = True  if  P(d) ≥ 0.2  OR  P(d) ≥ P(top1) × 0.60

min_active_domains = 1
max_active_domains = 4
```

### 4.2 上下文相似度 (Context Similarity — C)

**計算方式**：

```
C_user = cosine_sim(cur_user_vec, prev_user_vec)
C_bot  = cosine_sim(cur_user_vec, prev_bot_vec)

C = max(C_user, C_bot)    # 預設取最大值
```

**特殊處理**：
- 第一輪：`C = 0.5`（中性值）
- Bot 回覆截斷：保留前 800 字 + 尾 400 字（避免長回覆稀釋語義）

### 4.3 主題延續追蹤 (MultiTopicTracker — MT)

**核心指標：Total Variation Distance (TV)**

```
TV(P, Q) = 0.5 × Σ_d |P(d) - Q(d)|

TV ∈ [0, 1]
  0 → 分布完全相同
  1 → 分布完全不同
```

**MT 計算**（基於 TV 距離轉相似度）：

```
topic_overlap = 1.0 - TV(memory_dist, cur_dist)
```

其中 `memory_dist` 是歷史累積的 EMA 記憶分布（見下方）。

**延續判斷（非模糊情境，TV 主導）**：

| TV 範圍 | continuation_mode | 效果 |
|---------|-------------------|------|
| TV ≤ 0.2 | **strong** (強延續) | `topic_overlap = max(overlap, 0.7)`，沿用上輪 active_domains 和分布 |
| 0.2 < TV < 0.5 | **soft** (軟延續) | 取 top_domain match / overlap / cosine 判斷，合併新舊 active_domains，分布混合 α=0.5 |
| TV ≥ 0.5 | **shift** (切換) | `topic_overlap = min(overlap, 0.2)`，切換到本輪分布 |
| TV ≥ 0.6 | **hard shift** | 重置記憶 `memory_dist = cur_dist`（不做 EMA） |

**模糊情境**（`is_ambiguous=True`）：

TV 僅觀察，不主導切換判斷。預設視為延續，由上層 DST 模糊延續規則決定。

**記憶分布 EMA 更新**：

```
memory_dist(d) = α × memory_dist_old(d) + (1-α) × cur_dist(d)

α = decay_factor = 0.7（保留 70% 歷史，吸收 30% 本輪）

更新後 L1 歸一化：memory_dist = L1_normalize(memory_dist)
```

### 4.4 任務分類 (PrototypeClassifier)

**原型向量建構**（啟動時一次性完成）：

```
對每個任務 t（A-N, 共 14 類）：
  1. 編碼該任務的所有原型句子: embs = encode(sentences_t)     # shape: [n_t, 1024]
  2. L2 歸一化: embs = L2_normalize(embs)
  3. 取平均: proto_t = mean(embs, axis=0)                     # shape: [1024]
  4. 再次 L2 歸一化: proto_t = L2_normalize(proto_t)

proto_mat = stack([proto_A, proto_B, ..., proto_N])           # shape: [14, 1024]
```

**預測**：

```
q = L2_normalize(query_vec)
sims = proto_mat @ q                   # cosine similarities，shape: [14]

task_label = argmax(sims)
task_top_score = max(sims)             # 原始 cosine（用於 OOD 偵測）
```

**Softmax 分布**（用於 Policy）：

```
logits = sims × temperature            # temperature = 12.0（放大差異）
probs = softmax(logits)
task_dist = {label: prob for each task}
```

**歸一化熵**（用於 Slot 回填偵測）：

```
task_entropy = H_norm(probs) = -Σ p_i × ln(p_i) / ln(14)

task_entropy ∈ [0, 1]
  低 → 分類器很確定
  高 → 分類器模糊
```

**多任務偵測**（cosine 差值法）：

```
sorted_sims = sort(raw_sims, descending)
top1_sim = sorted_sims[0]
secondary_tasks = [label for (label, sim) in sorted_sims[1:2]
                   if top1_sim - sim ≤ max_sim_gap]

max_sim_gap = 0.05   # 兩任務 cosine 差距 ≤ 0.05 視為多任務
```

### 4.5 策略決策 (Policy Decision)

#### 4.5.1 模糊判定 (is_ambiguous)

```
is_ambiguous = (
    H_norm ≥ 0.7                                          # 領域熵高
    OR (TV > 0.8 AND q_len_norm < 0.35 AND top_domain ∉ input)  # 極端跳轉
)

其中 q_len_norm = min(len(query) / 50, 1.0)
```

**額外模糊保險**（Fallback 路徑）：

```
if topic_overlap < 0.2 and q_len_norm < 0.35:  ambig = True   # 極端跳轉 + 短句
if topic_overlap < 0.1 and C ≥ 0.55:           ambig = True   # 文字相似度幻覺
```

#### 4.5.2 Memory Agent 決策（優先路徑）

**輸入特徵向量（9 維）**：

| # | 特徵 | 計算方式 | 範圍 |
|---|------|----------|------|
| 0 | entropy | 領域分布歸一化熵 | [0, 1] |
| 1 | tv_distance | 當前分布與記憶分布的 TV 距離 | [0, 1] |
| 2 | topic_overlap | 原始（未經規則調整的）MT 分數 | [0, 1] |
| 3 | context_sim | 上下文相似度 C | [0, 1] |
| 4 | turn_index_norm | min(turn_index / 10, 1.0) | [0, 1] |
| 5 | query_len_norm | min(len(query) / 50, 1.0) | [0, 1] |
| 6 | is_multi_domain | 1.0 if multi-domain else 0.0 | {0, 1} |
| 7 | prev_action_norm | 上輪動作：STAY=0.0, REFRESH=0.5, CLARIFY=1.0 | {0, 0.5, 1} |
| 8 | consecutive_stay_count | min(連續 STAY 次數 / 5, 1.0) | [0, 1] |

**輸出**：3 類動作 `[STAY, REFRESH, CLARIFY]` 的 softmax 機率。

**信心門檻**：`max(probs) ≥ 0.5` → 使用 Agent 決策；否則 fallback 到規則。

**動作映射**：

| Agent 決策 | memory_action | semantic_flow | retrieval_action | 域選擇效果 |
|-----------|---------------|---------------|------------------|-----------|
| STAY | STAY | continue | NARROW_GRAPH / CONTEXT_FIRST | 沿用上輪分布、active_domains |
| REFRESH | REFRESH | shift_hard | WIDE_IN_DOMAIN | 使用本輪分布 |
| CLARIFY | CLARIFY | shift_soft | DUAL_OR_CLARIFY | 保留上輪分布，等待釐清 |

#### 4.5.3 Threshold 規則 Fallback（Agent 未啟用時）

**MT 聚合**：

```
MT = clamp01(topic_overlap + bonus)
bonus = 0.2 if topic_continue else 0.0
```

**四象限決策 (C × MT)**：

| C | MT | can_narrow | policy_case | retrieval_action |
|---|----|-----------:|-------------|-----------------|
| high (≥0.55) | high (≥0.50) | Yes | CH_MTH_NARROW | NARROW_GRAPH |
| high | high | No | CH_MTH_CTX | CONTEXT_FIRST |
| high | low | — | CH_MTL_CTX | CONTEXT_FIRST |
| low | high | Yes | CL_MTH_WIDE | WIDE_IN_DOMAIN |
| low | high | No | CL_MTH_CTX | CONTEXT_FIRST |
| low | low | Yes | CL_MTL_WIDE | WIDE_IN_DOMAIN |
| low | low | No | CL_MTL_DUAL | DUAL_OR_CLARIFY |

**Semantic Flow 計算**（Fallback 路徑）：

```python
C_hi = C ≥ 0.55
MT_hi = MT ≥ 0.50
C_soft = C ≥ 0.45
MT_soft = MT ≥ 0.30

if C_hi and MT_hi:         return "continue"
if C_hi or MT_hi:          return "shift_soft"
if C_soft and MT_soft:     return "shift_soft"
else:                      return "shift_hard"
```

#### 4.5.4 整體概況特殊規則（最高優先，覆蓋 Agent）

| 規則 | 條件 | 動作 |
|------|------|------|
| A | is_ambiguous + top_domain=整體概況 + 上輪非整體 | 重置記憶，啟用新對話 |
| B | is_ambiguous + 上輪為整體 | 沿用整體意圖 |

#### 4.5.5 Hard TV Override（覆寫）

```
if is_ambiguous and q_len_norm < 0.35:
    action = "DUAL_OR_CLARIFY"
    memory_action = "CLARIFY"
    semantic_flow = "shift_soft"
```

#### 4.5.6 Task H/K 特殊覆寫

```
if task ∈ {H, K}:
    if no detected_region:
        action = "LOCAL_RESOURCE_CLARIFY"  → 追問縣市
        semantic_flow = "shift_soft"
    else:
        action = "LOCAL_RESOURCE_SEARCH"   → 直接查詢
        semantic_flow = "shift_hard"
```

#### 4.5.7 域選擇融合

**Agent STAY**（has_history=True）：

```python
fused_distribution = prev_dist                   # 直接使用上輪分布
active_domains = prev_active_domains             # 沿用上輪
topic_tracker.memory_dist = prev_dist            # 回退記憶基準
topic_overlap = max(original, 0.5)               # 保底 overlap
```

**非模糊 Strong 延續**（TV ≤ 0.2）：

```python
fused_distribution = prev_dist
active_domains = prev_active_domains
```

**非模糊 Soft 延續**（0.2 < TV < 0.5）：

```python
active_domains = union(cur_active, prev_active)
fused_distribution = 0.5 × prev_dist + 0.5 × cur_dist   # 混合分布
fused = L1_normalize(fused)
```

### 4.6 Scope 分類（規則式）

| 條件 | Scope | 說明 |
|------|-------|------|
| top_domain = 整體概況 | S_overview | 查整張圖 |
| 模糊 + 有 fused_distribution + entropy ≥ 0.7 | 沿用上輪 scope | 模糊延續 |
| turn=0 + entropy ≥ 0.7 | S_overview | 首輪模糊預設整體 |
| len(active_domains) ≥ 2 | S_multi_domain | 多領域 |
| else | S_domain | 單領域 |

---

## 5. RL Agent 群 — 網路結構與訓練

### 5.1 Memory Agent (DialoguePolicyNet)

**網路結構**：

```
Input(9) → Linear(9, 32) → ReLU → Dropout(0.2)
         → Linear(32, 32) → ReLU → Dropout(0.2)
         → Linear(32, 3)  → Softmax → [STAY, REFRESH, CLARIFY]
```

**訓練算法**：REINFORCE (Policy Gradient)

```
Loss = -Σ log π(a_t | s_t) × R_t / N

梯度裁剪：max_norm = 1.0
學習率：lr = 0.001
探索率：ε = 0.1（ε-greedy）
```

**獎勵來源**：使用者回饋（滿意/不滿意/重問 → 正/負 reward）。

### 5.2 Rerank Agent (RerankAgent — Dirichlet Policy)

**網路結構**：

```
Input(33) → Linear(33, 64) → ReLU
          → Linear(64, 32)  → ReLU
          → Linear(32, 3)   → Softplus + 1.0 → α = [α_sem, α_str, α_ctx]
```

**輸入編碼（33 維）**：

```
task_onehot (15) + domain_onehot (11) + scope_onehot (3) + continuous (5)

continuous = [entropy, top_prob, context_sim, topic_overlap, turn_index_norm]
```

**輸出**：Dirichlet 分布的 α 參數，採樣得到 3 個權重。

```
weights = Dirichlet(α).sample()   # → [w_semantic, w_structural, w_context]

# 確定模式：weights = α / Σα（分布均值）
```

**權重 Clamping**（防止坍縮）：

```
semantic_weight   = max(w_sem, 0.25)
structural_weight = min(w_str, 0.50)
context_weight    = max(w_ctx, 0.10)
# L1 歸一化
```

**訓練算法**：REINFORCE on Dirichlet

```
Loss = -log_prob(Dirichlet(α), sampled_w) × (reward - 0.5)

baseline = 0.5（固定）
學習率：lr = 0.001
```

### 5.3 Planning Agent (PlanningPolicyNet)

**網路結構**：

```
Input(21) → Linear(21, 64) → ReLU → Dropout(0.2)
          → Linear(64, 64)  → ReLU → Dropout(0.2)
          → Linear(64, 6)   → Sigmoid → [p_assess, p_obs, p_train, p_suggest, p_community, p_gpt]
```

**輸入編碼（21 維）**：

```
semantic_section_scores (6) + task_onehot (14, 可加權) + domain_entropy (1)

semantic_section_scores = cosine_sim(query_vec, section_definition_vec) for each section
task_onehot: 若多任務，以 task_dist 機率加權（non-binary）
```

**6 個知識區塊**：

| 區塊 | 說明 | 資料來源 |
|------|------|----------|
| assessment | 評估結果/量表/分數 | Neo4j |
| observation | 臨床觀察紀錄 | Neo4j |
| training | 訓練方向/目標 | Neo4j |
| suggestion | 建議/推薦 | Neo4j |
| community_resources | 社區早療資源 | MySQL |
| external_gpt | 外部 GPT 知識補充 | LLM |

**決策方式**（線上推論 deterministic=True）：

```
active_sections = [section_i for i where p_i > 0.5]

保底：若無 section 被選中，取 argmax(probs)
```

**訓練算法**：Multi-label BCE + Entropy Regularization

```
Loss_policy = Σ BCE(output, target) × reward         # 正 reward
Loss_policy = Σ BCE(output, 0) × |reward| × target   # 負 reward（懲罰被選中的）

Loss_entropy = -β × mean(-p log p - (1-p) log(1-p))  # β = 0.02，防止策略坍縮

Total Loss = (Loss_policy + Loss_entropy) / N
```

---

## 6. RAG 檢索管線 (Retrieval Pipeline)

### 6.1 總流程

```
                    user_query
                        │
          ┌─────────────┼─────────────┐
          │        QueryRewriter       │
          │  (LLM 改寫 + 歷史壓縮)     │
          └─────────────┼─────────────┘
                        │
            queries = [original, rewritten, ...]
                        │
          ┌─────────────┼─────────────┐
          │       StrategyMapper        │
          │  (Planning Agent + Ontology)│
          └─────────────┼─────────────┘
                        │
          ┌─────────────┼─────────────┐
          │  Phase 1: Initial Retrieval │  ← Neo4j 圖譜 / MySQL / GPT
          │  Phase 1.5: Intermediate    │  ← 初步 Rerank + 多維度分類萃取
          │  Phase 2: Enrichment        │  ← ClinicalNorm 臨床對接
          │  Phase 3: Final Integration │  ← ExternalGPT 置頂 + 合併
          └─────────────┼─────────────┘
                        │
                  final_results
```

### 6.2 Query Rewriting

**Step 1 — Query Condensation**（僅 `semantic_flow="continue"` 時）：

```
利用 LLM 將多輪對話壓縮為獨立查詢：
  輸入：chat_history[-5:] + current_query
  輸出：standalone_query（不依賴上下文即可理解）
```

**Step 2 — Query Rewriting**：

```
LLM 改寫產生最多 2 個變體查詢，提升召回率
```

### 6.3 Reciprocal Rank Fusion (RRF)

多個查詢的結果合併：

```
RRF_score(node) = Σ_list 1 / (k + rank_in_list)

k = 60（平滑常數）
```

### 6.4 StrategyMapper — 檢索域決定

根據 `retrieval_action` 映射查詢範圍：

| retrieval_action | 查詢域 |
|-----------------|--------|
| NARROW_GRAPH | top_domain 的 1 個領域 |
| CONTEXT_FIRST | 機率最高的 2 個領域 |
| WIDE_IN_DOMAIN | 全部 active_domains |
| DUAL_OR_CLARIFY | 全部 active_domains |
| LOCAL_RESOURCE_SEARCH | 跳過 SUBDOMAIN_FETCH，只查 MySQL |

### 6.5 Reranker 評分公式

```
score(node) = cosine_sim(query_vec, node_vec) × w_semantic
            + task_boost(node.label, task)     × w_structural × 0.1
            + domain_prob(node.subdomain)      × w_context × 0.2

# 特殊調整：
if "(未選用)" in text:  score × 0.01
if "(非重點)" in text:  score × 0.30
if label == "ClinicalNorm":  score += 0.4
```

**Task-Label Boost 映射**：

| Task | 優先 Label |
|------|-----------|
| A | Summary, Meta |
| B, N | Assessment, Score |
| C, D | Observation, Assessment |
| E, F | TrainingDirection, Recommendation |
| G | Assessment, Recommendation |
| L | Assessment, TrainingDirection |

---

## 7. Slot Filling 槽位機制

### 7.1 任務槽位定義

| 任務 | 所需槽位 | 說明 |
|------|----------|------|
| B (分數解讀) | domain_focus | 哪個領域的分數 |
| C (觀察解讀) | domain_focus | 哪個領域的觀察 |
| E (在家訓練) | ability_focus | 哪個能力 |
| F (融入作息) | child_age | 小朋友年齡 |
| H (轉介資源) | region | 縣市 |
| J (學校合作) | school_type | 學校類型 |
| K (補助福利) | region | 縣市 |
| L (後續追蹤) | time_range | 上次評估時間 |
| N (進步查詢) | report_range | 比較哪幾份 |

> 任務 A, D, G, I, M 無需額外槽位。

### 7.2 非阻塞設計

```
缺槽 → 寬範圍檢索 + 回答 + 追問提示
槽位填滿 → 精確檢索
```

Slot 是 refinement，不是 blocking gate。

### 7.3 回填偵測流程

```
上輪有追問 (pending_slot 存在)?
  ├── No → 正常分類
  └── Yes → 判斷回填 vs 跳題
              ├── 長句或含問號 → 跳題 (clear pending)
              └── 短句 (≤8 字) 且無問號
                   ├── 分類信心低 → 回填 (繼承上輪任務)
                   └── 分類信心高
                        ├── 同上輪任務 → 回填
                        └── 不同任務 → 跳題 (clear pending)
```

---

## 8. LLM 生成模組

### 8.1 模型配置

```
Model: Qwen/Qwen3-4B-Instruct-2507
Server: vLLM (OpenAI 相容 API)
Temperature: 0.7 (回覆) / 0.3 (改寫) / 0.2 (壓縮)
```

### 8.2 系統提示詞動態選擇

| 情境 | 系統提示詞策略 |
|------|---------------|
| 無報告 ID | 一般醫學知識 + 提示上傳報告 |
| LOCAL_RESOURCE_CLARIFY (H/K 無縣市) | 溫和詢問所在縣市 |
| is_ambiguous | 引導釐清領域 + 初步回答 |
| 無檢索結果 | 醫學知識 + 引導提供生活細節 |
| Slot has_missing | 在結尾追加追問 (不覆蓋主提示) |
| 正常 | 專業早療助手基礎提示 |

### 8.3 輸入結構

```python
messages = [
    {"role": "system", "content": system_prompt},         # 動態選擇
    *conversation_history[-10:],                           # 最近 5 輪
    {"role": "user",   "content": f"【參考資料】\n{context}\n\n用戶問題：{query}"}
]
```

---

## 9. 狀態持久化與跨輪記憶

### 9.1 持久化檔案

| 檔案 | 路徑 | 內容 |
|------|------|------|
| DST 狀態 | `dialogue_states/user_{uid}_child_{cid}_state.json` | turn_index, prev_scope, prev_was_overview, context_sim 狀態, topic_tracker 狀態 |
| 對話歷史 | `dialogue_states/user_{uid}_child_{cid}_history.json` | messages, last_retrieved_context, slot_state |

### 9.2 跨輪傳遞的關鍵狀態

| 狀態 | 載體 | 用途 |
|------|------|------|
| turn_index | DST state | 控制首輪特殊邏輯 |
| prev_user_vec / prev_bot_vec | ContextSimilarity | 計算 C |
| memory_dist / prev_dist | MultiTopicTracker | 計算 TV, MT |
| prev_active_domains | MultiTopicTracker | 域融合判斷 |
| prev_scope | SemanticFlowClassifier | 模糊時 Scope 沿用 |
| prev_was_overview | SemanticFlowClassifier | 規則 B 判斷 |
| slot_state | SlotTracker | 回填偵測 |
| last_retrieved_context | history JSON | 上輪 context carry-over |

---

## 10. Embedding 快取機制

### 10.1 記憶體快取 (TextEncoder)

```python
self._cache = {}   # key: text → value: np.ndarray
# 上限 2000 筆，超過不再新增（LRU 策略未實作，但可避免記憶體耗盡）
```

### 10.2 磁碟快取 (_encode_with_cache)

```
位置：dialogue_state_module/cache/
命名：{cache_name}_{sha256_hash[:16]}.npy

流程：
1. 計算文字列表的 SHA-256 hash
2. 若 cache/{name}_{hash}.npy 存在且 shape[0] == len(texts) → np.load() 直接返回
3. 否則呼叫 encoder.encode_many() → np.save() 存檔

失效條件：文字內容或順序改變 → hash 不同 → 自動重新編碼
```

**已快取的資料集**：

| 快取名稱 | 句子數 | 說明 |
|----------|--------|------|
| domain_anchors | ~數十句 | 領域錨點句子 |
| overview_anchors | ~數句 | 整體概況錨點 |
| task_prototypes | ~374 句 | 任務原型句子 (A-N) |

---

## 11. 模組總表與檔案索引

### 核心模組

| 檔案 | 模組 | 說明 |
|------|------|------|
| `dialogue_manager.py` | DialogueManager | 主協調器，串接所有子系統 |
| `config.py` | — | 全域配置（DB/LLM/Embed 端點） |

### DST 子系統 (`dialogue_state_module/`)

| 檔案 | 類別 | 說明 |
|------|------|------|
| `semantic_flow_module_v2.py` | SemanticFlowClassifier | DST 主分類器（整合所有子模組） |
| `domain_router.py` | DomainRouter | 領域路由（Max Pooling + Softmax） |
| `context_similarity.py` | ContextSimilarity | 上下文相似度 C |
| `multi_topic_tracker.py` | MultiTopicTracker | 主題延續追蹤（TV + EMA） |
| `dst_policy.py` | DSTPolicyConfig, decide_policy | 四象限策略決策 |
| `task_scope_classifier.py` | PrototypeClassifier, TaskScopeClassifier | 任務分類（原型向量） |
| `slot_tracker.py` | SlotTracker | 槽位追蹤與回填偵測 |
| `slot_extractors.py` | extract_ability/school/time | 槽位值提取器 |
| `embedding.py` | TextEncoder, cosine_sim | 文本編碼與向量計算 |
| `domain_anchors.py` | DOMAINS, DOMAIN_ANCHORS | 領域錨點定義與載入 |
| `state_persistence.py` | save/load_dialogue_state | DST 狀態持久化 |

### RL Agent (`rl_pipeline/agents/`)

| 檔案 | 類別 | 輸入維度 → 輸出 | 訓練算法 |
|------|------|----------------|----------|
| `memory/memory_agent.py` | MemoryAgent | 9 → 3 (STAY/REFRESH/CLARIFY) | REINFORCE |
| `reranker/rerank_agent.py` | RLAgentManager | 33 → 3 (Dirichlet α) | REINFORCE on Dirichlet |
| `planner/planning_agent.py` | PlanningAgent | 21 → 6 (Sigmoid) | BCE + Entropy Reg. |

### 檢索子系統 (`retrieval_module_v2/`)

| 檔案 | 類別 | 說明 |
|------|------|------|
| `__init__.py` | RetrievalModuleV2 | 檢索主管線（RRF + 3 階段） |
| `strategy_mapper.py` | StrategyMapper | DST → 檢索策略映射 |
| `execution_engine.py` | ExecutionEngine | 執行 Neo4j/MySQL/GPT 查詢 |
| `reranker.py` | Reranker | 語義 + 結構 + 上下文重排 |
| `query_rewriter.py` | QueryRewriter | LLM 查詢改寫 |
| `pcst_solver.py` | PCSTSolver | 子圖發現 (PCST 演算法) |
| `topic_ontology.py` | TopicOntology | 知識區塊定義與 section 權重 |

### 生成子系統 (`llm_generate_module/`)

| 檔案 | 類別 | 說明 |
|------|------|------|
| `llm_generator.py` | LLMGenerator | OpenAI 相容 API 呼叫 |
| `prompt_manager.py` | LLMPromptManager | 提示詞模板管理 |

---

## 附錄：決策優先級總表

```
┌────────────────────────────────────────────────────┐
│ 優先級 1：閒聊快篩 (B0)                             │
│   → 關鍵字命中 + 短句 → 直接生成閒聊回覆              │
├────────────────────────────────────────────────────┤
│ 優先級 2：整體概況規則 A/B                           │
│   → 最高優先，不被 Agent 覆蓋                        │
├────────────────────────────────────────────────────┤
│ 優先級 3：Memory Agent 決策                         │
│   → max(probs) ≥ 0.5 時生效                        │
├────────────────────────────────────────────────────┤
│ 優先級 4：Hard TV Override                          │
│   → is_ambiguous + 短句 → 強制 CLARIFY              │
├────────────────────────────────────────────────────┤
│ 優先級 5：Threshold 規則 Fallback                    │
│   → C × MT 四象限決策                               │
├────────────────────────────────────────────────────┤
│ 優先級 6：Task H/K 特殊覆寫                          │
│   → 覆蓋 action + semantic_flow                     │
├────────────────────────────────────────────────────┤
│ 優先級 7：OOD 偵測 (B3)                             │
│   → task_top_score 過低 → 跳過 RAG                  │
├────────────────────────────────────────────────────┤
│ 優先級 8：Slot 追問 (F)                              │
│   → 缺槽 → 回覆結尾附加追問                          │
└────────────────────────────────────────────────────┘
```
