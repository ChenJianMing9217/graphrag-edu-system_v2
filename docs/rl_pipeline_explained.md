# EduBot v7 強化學習管線技術說明文件

> **文件版本**：v1.0 | **最後更新**：2026-04-21  
> **適用範圍**：`rl_pipeline/` 目錄下所有模組  
> **閱讀對象**：開發者、研究人員、論文審查委員

---

## 目錄

1. [系統總覽](#1-系統總覽)
2. [三大 RL Agent 架構概覽](#2-三大-rl-agent-架構概覽)
3. [Memory Agent（對話記憶決策）](#3-memory-agent對話記憶決策)
4. [Planning Agent（檢索規劃決策）](#4-planning-agent檢索規劃決策)
5. [Rerank Agent（重排序優化）](#5-rerank-agent重排序優化)
6. [獎勵機制（Reward System）](#6-獎勵機制reward-system)
7. [訓練流程](#7-訓練流程)
8. [預訓練與冷啟動](#8-預訓練與冷啟動)
9. [線上推論整合（DialogueManager）](#9-線上推論整合dialoguemanager)
10. [設計決策摘要](#10-設計決策摘要)

---

## 1. 系統總覽

EduBot v7 的 RL 管線採用**多代理人強化學習（Multi-Agent Reinforcement Learning）**架構，三個彼此獨立的 Agent 協作完成每一輪對話的檢索與回覆生成：

```
使用者輸入
    │
    ▼
┌─────────────────────────────────────────┐
│      對話狀態追蹤 (DST / SemanticFlow)    │
│  - 意圖分類 (Task A-N)                   │
│  - 領域路由 (Domain Router)              │
│  - 語義流程 (continue / switch / new)   │
└─────────────────────────────────────────┘
    │
    ├─► [Memory Agent]  → STAY / REFRESH / CLARIFY
    │
    ├─► [Planning Agent] → 決定撈哪幾類知識區塊
    │
    └─► [Rerank Agent]  → 預測語義/結構/上下文 3 維權重
              │
              ▼
         RAG 檢索 + LLM 生成回覆
              │
              ▼
    LLM-as-Judge 評分（離線）
              │
              ▼
    REINFORCE 更新各 Agent 神經網路
```

**核心設計哲學**：
- **Credit Assignment 分離**：三個 Agent 各自獲得獨立評分，避免功勞混淆
- **LLM-as-Judge**：用另一個 LLM 作為評審，繞過人工標注瓶頸
- **代理獎勵 + Judge 混合**：LLM 評分（70%）+ 輕量代理獎勵（30%）的混合架構，確保穩定性
- **行為克隆冷啟動**：先 SFT 預訓練至合理初始策略，再進行 RL 微調

---

## 2. 三大 RL Agent 架構概覽

| 屬性 | Memory Agent | Planning Agent | Rerank Agent |
|------|-------------|----------------|--------------|
| **決策問題** | 這輪要延續/重置/澄清？ | 要撈哪幾類知識？ | 三種相似度各給多少權重？ |
| **動作空間** | 離散 3 類（STAY/REFRESH/CLARIFY） | 離散多標籤（6 個 on/off 開關） | 連續 3 維（Dirichlet 分布採樣） |
| **神經網路** | 3 層 MLP（9→32→32→3） | 3 層 MLP（21→64→64→6） | 3 層 MLP（33→64→32→3）|
| **輸出層** | Softmax（最終用 argmax） | Sigmoid（多標籤獨立） | Softplus + 1.0（Dirichlet α）|
| **訓練演算法** | REINFORCE (Policy Gradient) | 加權 BCE + 熵正則化 | REINFORCE on Dirichlet |
| **源碼位置** | `agents/memory/memory_agent.py` | `agents/planner/planning_agent.py` | `agents/reranker/rerank_agent.py` |

---

## 3. Memory Agent（對話記憶決策）

### 3.1 任務定義

Memory Agent 決定每一輪對話應採取的**脈絡管理策略**：

| 動作 | 索引 | 意義 |
|------|------|------|
| `STAY` | 0 | 延續上一輪話題，保持相同的檢索上下文 |
| `REFRESH` | 1 | 清除舊背景，視為全新話題開啟檢索 |
| `CLARIFY` | 2 | 使用者語意不清，主動發問引導澄清 |

### 3.2 State（狀態向量）— 9 維

```python
features = [
    entropy,                # 領域分布熵值（0~1，高→多領域模糊）
    tv_distance,            # TV 距離（Total Variation，>=0.6 代表強切換）
    topic_overlap,          # 與上輪的主題重疊率（0~1）
    context_sim,            # 與上輪對話的語義相似度（cos sim）
    turn_index_norm,        # 對話輪次（turn / 10，clamped to 1.0）
    query_len_norm,         # 使用者問句長度（len / 50，clamped to 1.0）
    is_multi_domain,        # 是否跨多領域（0 或 1）
    prev_action_norm,       # 上一輪執行的動作（0=STAY, 0.5=REFRESH, 1.0=CLARIFY）
    consecutive_stay_count  # 連續 STAY 次數（n/5，clamped to 1.0）
]
```

**設計說明**：
- `entropy`（熵值）是 CLARIFY vs REFRESH 的關鍵分界線：高熵（>0.65）→ 意圖模糊 → CLARIFY；低熵 + TV 距離大 → 明確切換 → REFRESH
- `prev_action_norm` 與 `consecutive_stay_count` 是後期加入的時序特徵（SAR #3），解決「系統卡在 STAY 狀態不知切換」的問題

### 3.3 神經網路架構

```
輸入層 (9 維)
    │  fc1: Linear(9 → 32) + ReLU + Dropout(0.2)
    │  fc2: Linear(32 → 32) + ReLU + Dropout(0.2)
    │  fc3: Linear(32 → 3)
    ▼
輸出 logits (3 維) → Softmax → 機率分布
```

- **三層全連接 MLP**（`DialoguePolicyNet`）
- Dropout (p=0.2) 防止過擬合
- 訓練時輸出 logits，推論時取 argmax（或 ε-greedy 探索）

### 3.4 Action（動作選取）

```python
def select_action(state_dict, deterministic=False):
    logits = policy_net(state_tensor)
    probs = F.softmax(logits, dim=1)
    
    if not deterministic and np.random.rand() < epsilon:
        # 探索：隨機選一個動作 (ε=0.1)
        action_idx = np.random.choice(3)
    else:
        # 開發：選機率最高的動作
        action_idx = torch.argmax(probs).item()
```

### 3.5 訓練演算法：REINFORCE

```python
# Policy Gradient Loss
loss += -log_prob(action) * reward

# 梯度裁剪（防止極端獎勵導致梯度爆炸）
torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
```

訓練時會先對獎勵進行**Reward Centering（基線減法）**：
```python
mean_r = mean(all_rewards)
adjusted_rewards = [r - mean_r for r in rewards]
```
這步去除平均值偏移，大幅降低梯度方差，讓訓練更穩定。

---

## 4. Planning Agent（檢索規劃決策）

### 4.1 任務定義

Planning Agent 決定從知識庫撈取哪幾類內容，對應六個知識區塊：

| 區塊 | 說明 |
|------|------|
| `assessment` | 評估分數（PDMS-2、PEP-3 等標準化測量結果） |
| `observation` | 臨床觀察記錄（治療師的現場觀察描述） |
| `training` | 訓練方向（建議訓練策略） |
| `suggestion` | 具體建議（給家長的居家活動） |
| `community_resources` | 社區資源（機構、補助、社福資源 MySQL 查詢） |
| `external_gpt` | 外部通用衛教（不在報告中的一般知識，GPT 直答） |

### 4.2 State（狀態向量）— 21 維

```
狀態向量 = [語義分數 6 維] + [Task One-hot 14 維] + [領域熵 1 維]
```

**第 1-6 維：語義區塊相似度分數**
```python
sem_assessment         # 問題向量與「評估」知識的 cosine similarity
sem_observation        # 問題向量與「觀察」知識的 cosine similarity
sem_training           # 問題向量與「訓練」知識的 cosine similarity
sem_suggestion         # 問題向量與「建議」知識的 cosine similarity
sem_community_resources  # 問題向量與「社區資源」知識的 cosine similarity
sem_external_gpt         # 問題向量與「外部知識」的 cosine similarity
```

**第 7-20 維：Task One-hot（軟權重）**  
系統支援 14 種任務類型（A~N），使用**Confidence-Weighted Soft One-hot**：
```python
# 支援多任務時按各任務信心分數歸一化分配
task_dist = {"B": 0.7, "D": 0.3}
# 對應向量位置：task_onehot[1]=0.7, task_onehot[3]=0.3
```

**第 21 維：領域分布熵值**
```python
domain_entropy  # 高熵代表使用者問題橫跨多個領域（需要更廣泛的知識撈取）
```

### 4.3 任務-區塊先驗映射（用於預訓練）

| 任務 | 啟用區塊 | 說明 |
|------|---------|------|
| A（報告總覽） | assessment, observation | 看評估+觀察 |
| B（分數解讀） | assessment, observation | 數據+臨床 |
| E（在家訓練） | training, suggestion | 練習方式+建議 |
| H（轉介資源） | community_resources, external_gpt | 不需個案報告 |
| K（補助福利） | community_resources, external_gpt | 外部資源 |
| N（進步查詢） | assessment, observation | 雙報告比對 |

### 4.4 神經網路架構

```
輸入層 (21 維)
    │  fc1: Linear(21 → 64) + ReLU + Dropout(0.2)
    │  fc2: Linear(64 → 64) + ReLU + Dropout(0.2)
    │  fc3: Linear(64 → 6)
    ▼
輸出 (6 維) → Sigmoid（每個區塊獨立的 0~1 機率）
```

- **Sigmoid 輸出**（非 Softmax）：每個區塊是**獨立的二元決策**（可以同時開啟多個）
- 這是典型的**多標籤分類（Multi-label Classification）**問題

### 4.5 Action（動作選取）

```python
def select_sections(state_dict, threshold=0.5, deterministic=False):
    probs = policy_net(state_tensor)  # 6 維 sigmoid 輸出
    
    if deterministic:
        # 上線模式：硬門檻 0.5（穩定決策）
        active = [section for i, section in enumerate(labels) if probs[i] > 0.5]
    else:
        # 訓練模式：Bernoulli 採樣（提高探索多樣性）
        active = [section for i, section in enumerate(labels) if random.random() < probs[i]]
    
    # 保底機制：確保至少選一個
    if not active:
        active = [labels[argmax(probs)]]
    
    # 額外 15% 機率強制隨機擾動（冷啟動保險）
    if not deterministic and random.random() < 0.15:
        # 隨機插入一個额外區塊
```

### 4.6 訓練演算法：加權 BCE + 熵正則化

```python
# 正獎勵：最小化實際決策與目標選區的 BCE 損失
if reward >= 0:
    bce_loss = F.binary_cross_entropy(output_probs, target_mask)
    policy_loss = bce_loss.mean() * reward

# 負獎勵：只懲罰「選了卻拿負分」的動作
else:
    bce_loss = F.binary_cross_entropy(output_probs, zeros)
    policy_loss = (bce_loss * target * abs(reward)).mean()

# 熵獎勵（Entropy Bonus）：防止策略坍縮至全 0
# H(p) = -p·log(p) - (1-p)·log(1-p)
entropy_loss = -entropy_beta * entropy.mean()  # entropy_beta=0.02

total_loss = policy_loss + entropy_loss
```

**熵正則化的意義**：Sigmoid 輸出若全部趨近 0，策略會退化為「什麼都不撈」。加入熵項讓模型維持足夠的探索不確定性。

---

## 5. Rerank Agent（重排序優化）

### 5.1 任務定義

Rerank Agent 為三種重排序維度預測最佳**混合權重**：

| 維度 | 代號 | 說明 |
|------|------|------|
| 語義相似度 | `w_semantic` | 向量空間最近鄰排序（BM25/cosine） |
| 結構重要性 | `w_structural` | 節點在知識圖譜中的層級位置分數 |
| 上下文相關性 | `w_context` | 與對話歷史的關聯程度 |

最終分數：`score = w_s × semantic + w_t × structural + w_c × context`

### 5.2 State（狀態向量）— 33 維

```
狀態向量 = [Task one-hot 15 維] + [Domain one-hot 11 維] + [Scope one-hot 3 維] + [連續特徵 5 維]
          = 15 + 11 + 3 + 5 = 34 維（實際 input_dim 動態計算）
```

**One-hot 編碼部分**：
```python
tasks   = ["A", "B", ..., "N", "T_meta_query"]  # 15 種任務
domains = ["整體概況", "粗大動作", "精細動作", "感覺統合", 
           "口腔動作", "情緒行為與社會適應功能", "吞嚥功能",
           "口語理解", "口語表達", "說話", "認知功能"]  # 11 個領域
scopes  = ["S_overview", "S_domain", "S_multi_domain"]  # 3 種範圍
```

**連續特徵部分（5 維）**：
```python
entropy         # 領域分布熵值
top_prob        # 最高機率領域的信心度
context_sim     # 與對話歷史的語義相似度
topic_overlap   # 主題重疊率
turn_index_norm # 正規化輪次
```

### 5.3 神經網路架構：Dirichlet 策略網路

```
輸入層 (33 維)
    │  fc1: Linear(33 → 64) + ReLU
    │  fc2: Linear(64 → 32) + ReLU
    │  fc3: Linear(32 → 3)
    │  Softplus(x) + 1.0  → 確保輸出 > 1.0（Dirichlet alpha 參數）
    ▼
alpha 向量 = [α_semantic, α_structural, α_context]  (3 維，值域 > 1)
```

**為什麼用 Dirichlet 分布？**

Dirichlet 分布天然輸出「總和為 1 的比例向量」，完全符合三維混合權重的需求。相比直接輸出 softmax，Dirichlet 採樣：
- 允許更大的探索多樣性（高 α → 集中；低 α → 均勻分散）
- 支援「不確定性感知」決策——在上下文不明確時自動趨向均勻分配
- log_prob 可直接用於 REINFORCE 梯度計算

### 5.4 Action（權重預測）

```python
def select_weights(task, domain, scope, continuous, deterministic=False):
    alphas = model(state_tensor)  # Dirichlet α 參數
    
    if deterministic:
        # 上線模式：使用分布均值（α_i / Σα）
        weights = alphas / alphas.sum()
    else:
        # 訓練模式：從 Dirichlet 分布採樣
        dist = Dirichlet(alphas)
        weights = dist.sample()  # 隨機探索
    
    return [w_semantic, w_structural, w_context]
```

### 5.5 訓練演算法：Dirichlet-REINFORCE

```python
def update(memory_buffer):
    for (task, domain, scope), sampled_w, reward, cont in memory_buffer:
        alphas = model(state_tensor)
        dist = Dirichlet(alphas)
        
        # REINFORCE loss（帶基線 0.5）
        log_prob = dist.log_prob(target_weights)
        loss = -log_prob * (reward - 0.5)  # 0.5 作為基線，讓正負獎勵對稱
```

- **基線（Baseline）= 0.5**：相當於期望正規化獎勵在 [0,1] 空間的中位數，讓 reward>0.5 的樣本正向強化，reward<0.5 的樣本負向懲罰
- `log_prob` 計算的是：在目前的 Dirichlet 分布下，採樣到「當初那組權重」的對數機率

---

## 6. 獎勵機制（Reward System）

### 6.1 LLM-as-Judge 架構

```
┌──────────────────────────────────────────────────────────────┐
│                MultiAgentRewardJudge                          │
│                                                              │
│  輸入：{ query, response, retrieved_nodes, planning_info,    │
│          prev_query, memory_action, user_feedback }          │
│                                                              │
│  一次 LLM 呼叫同時輸出：                                      │
│    planning_reward  (1~5)                                    │
│    rerank_reward    (1~5)                                    │
│    memory_reward    (1~5)                                    │
│    answer_reward    (1~5)                                    │
│    analysis_log     (診斷原因)                                │
└──────────────────────────────────────────────────────────────┘
```

**評分維度分工**：

| 獎勵 | 評估重點 |
|------|---------|
| `planning_reward` | 撈取的知識類型是否契合問題？（嚴格比對 active_sections 清單） |
| `rerank_reward` | 排序後的 Top-5 是否比原始候選更相關？（對比 Raw vs Reranked） |
| `memory_reward` | STAY/REFRESH/CLARIFY 的判斷是否符合對話邏輯？ |
| `answer_reward` | 最終回覆是否專業、親切且具臨床權威性？（不用於 RL 訓練） |

### 6.2 正規化公式

```python
# 1~5 分正規化到 0~1
normalized = (raw_score - 1.0) / 4.0

# 有使用者回饋時（按讚=+1，踩=-1）
reward = 0.7 × normalized_judge_score + 0.3 × proxy_score

# 無使用者回饋時
reward = 0.7 × llm_judge_score + 0.3 × proxy_score
```

### 6.3 代理獎勵（Proxy Reward）— 輕量補充

無需 LLM，每輪都會計算的低延遲獎勵訊號：

```python
def compute_proxy_reward(flow_state, retrieved_nodes):
    # 取前 3 筆節點的向量分數平均（衡量檢索品質下限）
    retrieval_proxy = mean([node.score for node in nodes[:3]])
    
    # Memory 代理：context_sim 與 topic_overlap 的平均
    memory_proxy = (context_sim + topic_overlap) / 2.0
    
    return {
        "planning_proxy": retrieval_proxy,
        "rerank_proxy":   retrieval_proxy,
        "memory_proxy":   memory_proxy,
    }
```

### 6.4 遞延獎勵（Discounted Return）

同一 Session 內，每個時間步的獎勵按 γ=0.9 進行折扣：

```python
gamma = 0.9

# 若當前步有明確回饋（按讚/踩），直接使用
immediate_r = feedback_value or 0

# 若無，使用 session 末端回饋的折扣回傳
discounted_r = (gamma ** (T - 1 - t)) * final_reward
```

---

## 7. 訓練流程

### 7.1 離線訓練腳本：`unified_train_db.py`

```
流程圖：
資料庫讀取 (ChatMessage)
    │
    ▼
Session 分組 + 時序排序
    │
    ▼
for 每個 bot 訊息:
    1. 計算遞延獎勵
    2. 代理獎勵（不呼叫 LLM）
    3. 有使用者明確回饋？
       ├─ 是 → 用回饋計算 reward，跳過 LLM 呼叫
       └─ 否 → 呼叫 LLM Judge（3 個獨立評分 API）
    4. 將 (state, action, reward) 加入各 Agent 的 Buffer
    │
    ▼
多 Epoch 訓練（Planning: ≥30 epochs, Memory: 15 epochs）
    │
    ▼
儲存模型權重 (.pth)
    │
    ▼
更新訓練日誌 (training_history.json)
    │
    ▼
自動生成訓練儀表板 (training_dashboard.html)
```

### 7.2 訓練超參數

| 參數 | Memory Agent | Planning Agent | Rerank Agent |
|------|-------------|----------------|--------------|
| 學習率 | 0.001 | 0.005 | 0.001 |
| Gamma | 0.95 | N/A | N/A |
| Epsilon (探索率) | 0.1 | 0.1 | N/A |
| Entropy Beta | N/A | 0.02 | N/A |
| Epochs per run | 15 | ≥30 | 1 |
| 梯度裁剪 | max_norm=1.0 | 無 | 無 |
| Optimizer | Adam | Adam | Adam |

### 7.3 多輪 Epoch 訓練策略

```python
# Planning Agent：至少 30 輪，每輪隨機 Shuffle Buffer
for epoch in range(max(num_epochs, 30)):
    random.shuffle(planning_exp)
    loss = planning_agent.update(planning_exp)
```

Shuffle 的意義：防止 mini-batch 相關性導致梯度偏向，模擬 i.i.d. 假設。

---

## 8. 預訓練與冷啟動

### 8.1 行為克隆（Behavioral Cloning）

在 RL 訓練正式開始前，先用**監督學習**給 Agent 一個合理的初始策略，避免冷啟動期間的隨機決策傷害使用者體驗。

```
資料來源（優先順序）：
  1. sft_dataset_v4_final.jsonl（真實臨床對話，人工標注）
  2. 合成資料（Task→Section 規則映射，自動生成）
```

### 8.2 Memory Agent 預訓練

- **損失函數**：CrossEntropyLoss（帶類別權重 + 標籤平滑 0.1）
- **類別不平衡處理**：
  - CLARIFY 樣本 **4 倍 oversampling**（真實資料中 CLARIFY 比例僅 ~6%）
  - 額外生成 **邊界增強樣本**（REFRESH vs CLARIFY 的對比對）：
    - REFRESH 樣本：低熵（entropy=0.08~0.35）+ 長句子
    - CLARIFY 樣本：高熵（entropy=0.80~0.99）+ 短句子
- **訓練設定**：80 epochs, lr=0.005, weight_decay=1e-4

### 8.3 Planning Agent 預訓練

- **損失函數**：Binary Cross Entropy（多標籤）
- **Label Smoothing**：target 值 1→0.88, 0→0.06（防止 sigmoid 飽和）
- **資料混合策略**：
  - 真實 SFT 資料（主要）
  - 合成 Task-Prior 資料（補充硬性先驗，避免 SFT 資料中的標注雜訊污染）
- **訓練設定**：≥60 epochs, lr=0.005, weight_decay=1e-4

---

## 9. 線上推論整合（DialogueManager）

### 9.1 推論呼叫鏈

```python
# 1. DST 分析
flow_result = flow_classifier.predict(user_input)  # 意圖 + 領域 + 語義流程

# 2. Memory Agent 推論（透過 SemanticFlowClassifier 內部呼叫）
memory_action = flow_result.policy_decision.memory_action  # STAY/REFRESH/CLARIFY

# 3. 構建 Rerank Agent 狀態
rl_continuous = {entropy, top_prob, context_sim, topic_overlap, turn_index_norm}
w_semantic, w_structural, w_context = rl_agent.predict_weights(task, domain, scope, rl_continuous)

# 4. Planning Agent 推論（由 RetrievalModuleV2 內部呼叫）
planning_info = retriever.retrieve(user_query, turn_state, doc_id, rerank_config)
```

### 9.2 Flow State 記錄（供離線訓練使用）

每一輪 bot 回覆都會把**完整的決策上下文**記入 `ChatMessage.flow_state` 欄位：

```json
{
  "memory_action": "STAY",
  "planning_info": {
    "active": ["assessment", "observation"],
    "probs": {"assessment": 0.82, "observation": 0.71, ...}
  },
  "semantic_section_scores": {
    "assessment": 0.78,
    "observation": 0.65,
    ...
    "domain_entropy": 0.32,
    "task_label": "B"
  },
  "rerank_w_semantic": 0.65,
  "rerank_w_structural": 0.22,
  "rerank_w_context": 0.13,
  "normalized_entropy": 0.32,
  "tv_distance": 0.15,
  "context_sim": 0.84,
  "topic_overlap": 0.71,
  "turn_index": 3,
  "task_pred": "B",
  "top_domain": "精細動作",
  "prev_memory_action_norm": 0.0,
  "consecutive_stay_count": 0.2
}
```

這個 JSON 是離線訓練的黃金資料源，讓訓練時能夠精確重建各 Agent 在推論時所面對的狀態。

---

## 10. 設計決策摘要

### 10.1 為何選用 REINFORCE 而非 DQN/PPO？

| 考量 | 說明 |
|------|------|
| **動作空間多樣** | Memory 是離散 3 類，Planning 是多標籤，Rerank 是連續 Dirichlet。REINFORCE 可統一處理所有類型 |
| **資料稀疏** | 每次對話才產生一筆訓練資料，無法維持 DQN 所需的 Experience Replay 規模 |
| **延遲獎勵** | 每輪獎勵由 LLM Judge 評估，不需要環境即時回饋的 Q-value 估計 |
| **實作簡潔** | REINFORCE 只需要 log_prob × reward，無需存儲 Q-table 或 Critic 網路 |

### 10.2 關鍵穩定化措施

| 問題 | 解法 |
|------|------|
| 梯度爆炸 | 梯度裁剪（max_norm=1.0）|
| 策略坍縮（全 0） | 熵正則化（entropy_beta=0.02）|
| 獎勵尺度偏移 | Reward Centering（減去 batch 平均值）|
| 冷啟動隨機決策 | 行為克隆預訓練（SFT → RL）|
| Off-policy 偏移 | 只取最近 200 筆訓練（TRAINING_WINDOW=200）|
| 類別不平衡（CLARIFY 稀少） | 4 倍 oversampling + 邊界增強樣本 |
| 維度不符（模型升級後） | 載入時檢查 fc1.weight 維度，不一致時重置 |

### 10.3 Credit Assignment 設計

三個 Agent 有不同的「責任範圍」：

```
使用者的一句話
    │
    ├── Memory Agent 負責：「我是否正確判斷了話題脈絡？」
    │     評估依據：prev_query vs current_query 的語意轉換
    │
    ├── Planning Agent 負責：「我選的知識類型對嗎？」
    │     評估依據：active_sections vs 問題的實際需求
    │
    └── Rerank Agent 負責：「我的排序讓最相關的在最前面嗎？」
          評估依據：Raw Top-5 vs Reranked Top-5 的品質對比
```

若三個 Agent 共用同一個獎勵（如最終回覆品質），會產生嚴重的 **Credit Assignment Problem**（無法判斷是誰做對/做錯）。獨立評分機制是解決這個問題的核心設計。

---

*文件由 Antigravity 根據 `rl_pipeline/` 原始碼自動分析生成*  
*如有問題請參考各 Agent 的源碼檔案或聯繫開發者*
