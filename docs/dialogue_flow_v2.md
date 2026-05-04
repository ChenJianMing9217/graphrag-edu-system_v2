# 對話流程設計 v2（含 Slot Filling 與 Agent 優先機制）

> **狀態**：設計稿，待確認後實作
> **日期**：2026-04-20

---

## 0. 設計原則

1. **Agent 優先、規則兜底**：MemoryAgent / PlanningAgent / RerankAgent 在信心足夠時主導決策，信心不足回退到閾值規則
2. **Slot 是 refinement，不是阻塞**：槽位缺失時仍執行寬範圍檢索並回答，追問附帶在回覆中
3. **任務每輪重判，不做 EMA 追蹤**：Task 是意圖，本質上應該每輪獨立辨識（與領域不同）
4. **追問繼承靠 Slot Tracker**：系統追問後使用者的短回答，走 slot 回填路徑而非重新分類

---

## 1. 完整對話管線（單輪）

```
使用者輸入 (user_input)
    │
    ▼
┌─────────────────────────────────────────────────┐
│ A. 載入狀態                                       │
│   ├── DST 狀態 (dialogue_states/..._state.json)   │
│   ├── 對話歷史 + prev_retrieved_context            │
│   └── 【新增】Slot 狀態 (pending_slot)             │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│ B0. 閒聊快篩                                      │
│   字數 ≤ 10 + 關鍵字（你好/謝謝/嗨…）              │
│   └── 命中 → 直接回覆，跳過所有下游（< 1 秒）       │
└─────────────────────┬───────────────────────────┘
                      │ （未命中）
                      ▼
┌─────────────────────────────────────────────────┐
│ B1. 文字編碼（一次性）                              │
│   query_vector = encoder.encode(user_input)       │
│   └── 全流程共用，不重複編碼                        │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ B2.【新增】Slot 回填偵測                                      │
│                                                               │
│   if pending_slot 存在:                                       │
│     ├── 跳題偵測：                                             │
│     │   ├── 輸入短（≤ 8 字）+ 無問號 + task 分類信心低（< 0.5）│
│     │   │   → 判定為 slot 回填                                 │
│     │   │   → 繼承上輪 task_label                              │
│     │   │   → 提取 slot 值（region / domain_focus / …）        │
│     │   │   → 清除 pending_slot                                │
│     │   │                                                      │
│     │   └── 輸入長 or 含問號 or task 信心高且不同任務           │
│     │       → 判定為跳題                                       │
│     │       → 清除 pending_slot                                │
│     │       → 走正常分類（B3）                                  │
│     │                                                          │
│   else:                                                        │
│     └── 無 pending_slot → 走正常分類（B3）                      │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────┐
│ B3. DST 分析                                      │
│   SemanticFlowClassifier.predict()                │
│                                                    │
│   ├── DomainRouter                                │
│   │   → top_domain, active_domains, entropy       │
│   │                                                │
│   ├── TaskScopeClassifier                         │
│   │   → task_label (A~N), task_dist               │
│   │   → secondary_tasks (top-2)                   │
│   │   →【新增】task_entropy                        │
│   │                                                │
│   ├── ContextSimilarity                           │
│   │   → similarity_score (C 值)                    │
│   │                                                │
│   ├── MultiTopicTracker（領域追蹤）                 │
│   │   → overlap_score (MT 值), tv_distance         │
│   │                                                │
│   ├── RegionExtractor                             │
│   │   → detected_region                            │
│   │                                                │
│   └── PolicyDecision（決策層，見第 2 節）            │
│       → retrieval_action, semantic_flow,           │
│         memory_action, is_ambiguous                │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│ B4. 早期攔截                                      │
│                                                    │
│   ├── 閒聊後備偵測                                │
│   │   task == CHITCHAT → 生成閒聊回覆，結束        │
│   │                                                │
│   └── Out-of-domain 偵測                          │
│       task_top_score < 0.30                        │
│       or (< 0.40 且 top_prob < 0.25)              │
│       → 生成引導回覆，結束                         │
└─────────────────────┬───────────────────────────┘
                      │ （通過攔截）
                      ▼
┌───────────────────────────────────────────────────────────┐
│ B5.【新增】Slot 檢查 & Pending 設定                         │
│                                                             │
│   根據 TASK_SLOTS[task_label] 檢查必要槽位：                 │
│                                                             │
│   TASK_SLOTS = {                                            │
│       "B": ["domain_focus"],                                │
│       "C": ["domain_focus"],                                │
│       "E": ["ability_focus"],                               │
│       "F": ["child_age"],       ← 可從 DB 自動填充          │
│       "H": ["region"],                                      │
│       "K": ["region"],                                      │
│       "J": ["school_type"],                                 │
│       "L": ["time_range"],                                  │
│       "N": ["report_range"],                                │
│   }                                                         │
│                                                             │
│   槽位值來源（優先順序）：                                    │
│     1. Slot 回填值（B2 步驟提取的）                           │
│     2. 使用者輸入中直接提取                                   │
│     │   region       → RegionExtractor（已有）                │
│     │   domain_focus → DomainRouter top_domain                │
│     │   child_age    → DB Child.birth_date（已有）            │
│     │   report_range → 預設「最近兩份」                        │
│     3. DB / Session 自動補充                                  │
│     4. 無法取得 → slot 為空                                   │
│                                                              │
│   slot_status:                                               │
│     ├── all_filled   → 精確檢索                               │
│     └── has_missing  → 寬範圍檢索 + 標記 pending_slot         │
│                        （不阻塞，仍然執行檢索）                │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────┐
│ C. RL 加權預測                                    │
│   RerankAgent.predict_weights(                    │
│     task, domain, scope, continuous_features      │
│   )                                                │
│   → w_semantic, w_structural, w_context           │
│     （三維 Dirichlet 分布）                        │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ D. RAG 檢索                                                  │
│                                                               │
│   ┌── [查詢改寫] QueryRewriter.rewrite()                     │
│   │     ├── continue 時：代名詞解析                           │
│   │     └── 最多 2 個改寫版本 → RRF 融合                     │
│   │                                                           │
│   ├── [第一階段：個案資料]                                     │
│   │     StrategyMapper.map_dst_to_strategy()                  │
│   │     ├── PlanningAgent（優先）→ 選擇 section               │
│   │     └── 靜態 ontology（回退）                              │
│   │     ExecutionEngine.execute_initial()                     │
│   │     └── Neo4j + MySQL                                     │
│   │                                                           │
│   │  【變更】slot_status 影響檢索寬度：                        │
│   │     all_filled  → 用 slot 值精確限縮                      │
│   │     has_missing → 用 task 預設寬範圍                      │
│   │     例：Task K + region=台北 → 只查台北補助               │
│   │         Task K + region=null → 查全國通用補助說明          │
│   │                                                           │
│   ├── [中間重排] Reranker.rerank()                            │
│   │     ├── 複用 query_vector + batch encode                  │
│   │     └── RL 權重 (w_semantic, w_structural, w_context)     │
│   │                                                           │
│   ├── [第二階段：臨床知識增強]                                 │
│   │     ExecutionEngine.execute_enrichment()                  │
│   │     └── ClinicalBridgeService → 臨床常模                  │
│   │                                                           │
│   └── [最終整合] 臨床增強 → 排序後個案資料                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│ E. prev_context 條件攜帶                          │
│                                                    │
│   semantic_flow 的來源（已修正）：                  │
│     Agent 路徑 → 從 memory_action 映射             │
│       STAY    → continue                           │
│       REFRESH → shift_hard                         │
│       CLARIFY → shift_soft                         │
│     Fallback  → C+MT 公式計算                      │
│                                                    │
│   攜帶規則：                                       │
│     continue                    → 帶入上輪 top-3   │
│     shift_soft + ctx_sim ≥ 0.45 → 帶入（去重）     │
│     shift_hard / 其他           → 不帶              │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────────────────┐
│ F. LLM 生成                                                │
│                                                             │
│   PromptManager.build_user_prompt()                         │
│   ├── 依任務/流向選擇 system_prompt 模板                     │
│   ├── 依 generation_config 調整生成參數                      │
│   └── prev_context 注入                                     │
│                                                             │
│  【變更】追問策略：                                          │
│     has_missing slot → prompt 附加追問指令                   │
│     ├── 回答仍基於檢索結果（寬範圍）                         │
│     └── 結尾自然追問缺失的資訊                               │
│     例：Task K + region=null                                │
│         → 先回答「早療補助通常包含交通費、訓練費…」           │
│         → 追問「請問您在哪個縣市？我可以查更精確的方案」       │
│                                                             │
│   all_filled slot → 正常回答，不追問                         │
└─────────────────────┬─────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│ G. 更新狀態 & 儲存                                │
│                                                    │
│   ├── ContextSimilarity.update()                  │
│   ├── DST 狀態 → dialogue_states/ JSON            │
│   ├── 對話歷史 + last_retrieved_context            │
│   ├── MySQL ChatMessage (含 turn_state)            │
│   └──【新增】Slot 狀態                             │
│       ├── has_missing → 儲存 pending_slot          │
│       │   { "active_task": "K",                    │
│       │     "pending_slot": "region",              │
│       │     "filled_slots": {} }                   │
│       └── all_filled → 清除 pending_slot           │
└─────────────────────────────────────────────────┘
```

---

## 2. 決策層 (_decide_policy) 細節

決策層決定三個核心變數：`retrieval_action`、`memory_action`、`semantic_flow`

### 2.1 優先順序（高 → 低）

```
① Task H/K 硬覆寫（最高優先）
   └── 有 region → LOCAL_RESOURCE_SEARCH + REFRESH + shift_hard
       無 region → LOCAL_RESOURCE_CLARIFY + CLARIFY + shift_soft
       （不管 Agent 說什麼都會蓋掉）

② Hard TV Override
   └── is_ambiguous + 短查詢 (q_len < 0.35)
       → action = DUAL_OR_CLARIFY
       → memory_action = CLARIFY
       → semantic_flow = shift_soft
       （蓋掉 Agent 的 retrieval_action 和 memory_action）

③ MemoryAgent（信心 ≥ 0.5 時生效）
   └── STAY    → NARROW_GRAPH    + continue
       REFRESH → WIDE_IN_DOMAIN  + shift_hard
       CLARIFY → DUAL_OR_CLARIFY + shift_soft

④ 閾值規則（Agent 信心 < 0.5 時的 Fallback）
   └── C+MT 公式 → retrieval_action
       retrieval_action → 推導 memory_action
       C+MT 公式 → semantic_flow
```

### 2.2 三個變數的一致性保證

| 路徑 | retrieval_action | memory_action | semantic_flow |
|------|-----------------|---------------|---------------|
| Agent STAY | NARROW_GRAPH / CONTEXT_FIRST | STAY | continue |
| Agent REFRESH | WIDE_IN_DOMAIN | REFRESH | shift_hard |
| Agent CLARIFY | DUAL_OR_CLARIFY | CLARIFY | shift_soft |
| Hard TV Override | DUAL_OR_CLARIFY | CLARIFY | shift_soft |
| Task H/K + region | LOCAL_RESOURCE_SEARCH | REFRESH | shift_hard |
| Task H/K - region | LOCAL_RESOURCE_CLARIFY | CLARIFY | shift_soft |
| Fallback | 由 C+MT 決定 | 由 action 推導 | 由 C+MT 決定 |

三個變數在任何路徑下都保持語義一致，不會出現矛盾。

---

## 3. Slot Filling 機制

### 3.1 設計定位

- Slot **不是阻塞門檻**，是 refinement 機制
- 缺槽位時：寬範圍檢索 + 回答 + 自然追問
- 有槽位時：精確檢索 + 直接回答

### 3.2 任務槽位定義

| Task | 名稱 | 必要槽位 | 值來源 | 缺槽時的寬範圍策略 |
|------|------|---------|--------|-----------------|
| **B** | 分數解讀 | `domain_focus` | DomainRouter top_domain | 給整份報告分數總覽 |
| **C** | 觀察解讀 | `domain_focus` | DomainRouter top_domain | 給所有觀察摘要 |
| **E** | 在家訓練 | `ability_focus` | 使用者輸入 / 報告建議 | 給所有訓練方向 |
| **F** | 融入作息 | `child_age` | DB Child.birth_date | 給通用建議 |
| **H** | 轉介資源 | `region` | RegionExtractor | 給一般性轉介建議 |
| **J** | 學校合作 | `school_type` | 使用者輸入 | 給通用合作建議 |
| **K** | 補助福利 | `region` | RegionExtractor | 給全國通用補助說明 |
| **L** | 後續追蹤 | `time_range` | 使用者輸入 | 給一般性追蹤建議 |
| **N** | 進步查詢 | `report_range` | 預設最近兩份 | 自動用最近兩份 |
| A, D, G, I, M | — | **無** | — | 直接回答 |

### 3.3 Slot 回填流程

```
前提：上輪留有 pending_slot

使用者輸入進來
    │
    ▼
跳題偵測（三個條件綜合判斷）：
    │
    │  條件 A：輸入長度 ≤ 8 字 且 不含問號
    │  條件 B：task 分類信心 < 0.5（分類器對短回答沒把握）
    │  條件 C：領域 TV 距離 < 0.5（話題沒有大幅切換）
    │
    ├── A+B 成立（或 A+C 成立）
    │   → 判定為 slot 回填
    │   → 繼承上輪 active_task
    │   → 從輸入中提取對應 slot 值
    │   → 用填完的 slot 走精確檢索
    │
    └── 不成立（輸入長 / 有問號 / task 信心高且不同任務）
        → 判定為跳題
        → 清除 pending_slot
        → 走正常 B3 分類
```

### 3.4 Slot 值提取方式

| 槽位 | 提取方法 | 已有元件？ |
|------|---------|-----------|
| `region` | `extract_region(user_input)` | ✅ 已有 |
| `domain_focus` | DomainRouter 本輪 top_domain | ✅ 已有 |
| `ability_focus` | 從輸入比對臨床能力清單 | ❌ 需新增 |
| `child_age` | DB Child.birth_date → age_months | ✅ 已有 |
| `school_type` | 關鍵字比對（公幼/私幼/國小…） | ❌ 需新增 |
| `time_range` | 時間表達式提取（去年/三個月前…） | ❌ 需新增 |
| `report_range` | 預設最近兩份 / 使用者指定 | ✅ 已有邏輯 |

---

## 4. Task Entropy（新增）

### 4.1 計算方式

```python
# 複用現有的 normalized_entropy() 函數
task_entropy = normalized_entropy(task_dist_array)
# 0 = 非常確定某個任務
# 1 = 完全不確定
```

### 4.2 用途

| task_entropy | 判定 | 影響 |
|-------------|------|------|
| < 0.3 | 任務明確 | 正常流程 |
| 0.3 ~ 0.6 | 略有模糊 | 可作為 Agent 輸入特徵 |
| ≥ 0.6 | 任務高度模糊 | 標記 task_ambiguous，檢索策略偏保守 |

### 4.3 與 Slot 回填的關係

- Slot 回填偵測時，task_entropy 高（分類器對短回答沒信心）是判定「回填而非跳題」的信號之一
- 正常分類時，task_entropy 高可觸發更寬的檢索範圍

### 4.4 未來 Agent 擴展

當 task_entropy 信號穩定後，可加入 MemoryAgent 的輸入特徵：

```
目前 9 維 → 擴展為 11 維
  + task_entropy         （任務亂度，0~1）
  + has_pending_slot     （是否有未填槽位，0/1）
```

此步驟需重新訓練模型，在信號驗證穩定後再執行。

#### 權重遷移方案（過去訓練不會白費）

MemoryAgent 模型結構（`DialoguePolicyNet`）：

```
fc1: Linear(9, 32)  → 擴展後 Linear(11, 32)   ← 只有這層 shape 改變
fc2: Linear(32, 32) → 不變
fc3: Linear(32, 3)  → 不變
```

直接 `load_state_dict()` 會因 shape mismatch 失敗。需要權重移植腳本：

```python
old_state = torch.load("memory_agent.pth")
new_net = DialoguePolicyNet(input_dim=11)

# fc1: 舊 [32,9] → 新 [32,11]，前 9 列複製，新 2 列初始為 0
new_net.fc1.weight.data[:, :9] = old_state["fc1.weight"]
new_net.fc1.weight.data[:, 9:] = 0.0
new_net.fc1.bias.data = old_state["fc1.bias"]

# fc2, fc3: shape 不變，直接複製
new_net.fc2.weight.data = old_state["fc2.weight"]
new_net.fc2.bias.data = old_state["fc2.bias"]
new_net.fc3.weight.data = old_state["fc3.weight"]
new_net.fc3.bias.data = old_state["fc3.bias"]
```

遷移後的效果：

| 部分 | 狀態 |
|------|------|
| 舊 9 維特徵的權重 | 完整保留，行為與擴展前一致 |
| fc2, fc3 | 完整保留 |
| 新 2 維���徵（task_entropy, has_pending_slot） | 權重 = 0，初始時不影響輸出 |

**擴展後、訓練前，模型行為跟舊版完全相同。**

後續用較低 learning rate（如 0.0003）做 fine-tune，讓模型逐漸學會利用新特徵即可。
PlanningAgent / RerankAgent 如需擴展，同理處理。

### 4.5 多任務偵測（secondary_tasks）

#### 現狀

`predict_task_topk(k=2)` 在**單輪內**偵測是否同時符合兩個任務。

觸發條件（目前）：
```
top2_prob >= 0.12  且  top2_prob / top1_prob >= 0.55
```

**問題**：分類器使用 `temperature = 12.0` 的 softmax，分布極度尖銳。
cosine similarity 只要差 0.05 以上，第二名就幾乎無法通過 0.55 的比值門檻。
實際上只有 H/K 這對語義最接近的任務偶爾能互觸，其他組合幾乎不會出現。

#### 改進方案

改用**原始 cosine similarity 差值**判斷，不受 softmax 溫度影響：

```python
# 現行（softmax 後比值，受溫度影響極大）
if top2_prob / top1_prob >= 0.55:  # ← 幾乎不會過

# 改為（原始 cosine 差值，更穩定）
if top1_sim - top2_sim <= 0.05:    # ← 語義真的很接近才觸發
```

#### 下游用途

| 位置 | 怎麼用 |
|------|--------|
| `_decide_policy()` | 檢查 secondary_tasks 含 H/K → 觸發資源覆寫 |
| `strategy_mapper.py` | 傳給 PlanningAgent，合併多任務的 section |
| `planning_agent.py` | 把 secondary_tasks 的 section 也開啟 |
| Slot Tracker（新增） | 同時檢查主任務和副任務的槽位 |

#### 定位

- **單輪多意圖偵測**，不是跨輪追蹤
- 跨輪的任務延續由 Slot Tracker 的 `pending_slot` 機制處理，不需要任務 EMA

---

## 5. 多輪對話範例

### 範例 1：補助查詢（Slot 回填）

```
Turn 1:
  User: 「我想了解早療補助」
  ├── Task: K（補助福利）
  ├── Slot 檢查: region = null → has_missing
  ├── 檢索: 寬範圍（全國通用補助說明）
  ├── 回覆: 早療補助一般包含交通費、訓練費…請問您在哪個縣市？
  └── 儲存: pending_slot = { active_task: "K", slot: "region" }

Turn 2:
  User: 「台北」
  ├── 偵測 pending_slot 存在
  ├── 跳題偵測: 2 字、無問號、task 分類信心低 → slot 回填
  ├── 繼承 Task K，提取 region = 台北
  ├── Slot 檢查: region = 台北 → all_filled
  ├── 檢索: 精確查詢台北 subsidy_program
  ├── 回覆: 台北市早療補助方案：交通費每次 200 元…
  └── 清除 pending_slot
```

### 範例 2：補助查詢後跳題

```
Turn 1:
  User: 「台北有什麼補助」
  ├── Task: K, region = 台北 → all_filled
  ├── 檢索: 精確查詢台北
  ├── 回覆: 台北市早療補助…
  └── 無 pending_slot

Turn 2:
  User: 「那他的語言能力怎麼樣？」
  ├── 無 pending_slot → 正常分類
  ├── Task: C（觀察解讀）
  ├── 正常 DST + RAG 流程
  └── 回覆: 根據報告，孩子的語言理解…
```

### 範例 3：追問中跳題

```
Turn 1:
  User: 「可以申請什麼補助？」
  ├── Task: K, region = null → has_missing
  ├── 回覆: 各縣市補助方案不同…請問您在哪個縣市？
  └── 儲存: pending_slot = { active_task: "K", slot: "region" }

Turn 2:
  User: 「他的語言發展正常嗎？」
  ├── 偵測 pending_slot 存在
  ├── 跳題偵測: 10 字、含問號、task 信心高判 C → 跳題
  ├── 清除 pending_slot
  ├── Task: C, 正常流程
  └── 回覆: 根據報告…
```

### 範例 4：同領域換任務（無 Slot）

```
Turn 1:
  User: 「他的語言評估分數怎麼看？」
  ├── Task: B（分數解讀）, Domain: 語言
  └── 回覆: 語言領域的標準化分數為…

Turn 2:
  User: 「那語言要怎麼在家練？」
  ├── 無 pending_slot → 正常分類
  ├── Task: E（在家訓練）, Domain: 語言
  ├── 領域 TV 距離低 → 但 Task 從 B 切到 E
  ├── MemoryAgent 只看到領域穩定 → 傾向 STAY
  │   （目前限制：Agent 看不到任務切換）
  │   （未來：加入 task_entropy 等特徵後改善）
  └── 回覆: 語言訓練建議…
```

### 範例 5：分數解讀追問領域（Slot 回填）

```
Turn 1:
  User: 「報告的分數代表什麼意思？」
  ├── Task: B（分數解讀）
  ├── Slot 檢查: domain_focus = 整體概況（DomainRouter 判定）
  │   → 若 domain entropy 高（不確定哪個領域）→ has_missing
  ├── 檢索: 寬範圍（全部分數總覽）
  ├── 回覆: 報告中各領域分數代表…想進一步了解哪個領域的分數？
  └── 儲存: pending_slot = { active_task: "B", slot: "domain_focus" }

Turn 2:
  User: 「語言」
  ├── 偵測 pending_slot 存在
  ├── 跳題偵測: 2 字、無問號 → slot 回填
  ├── 繼承 Task B，domain_focus = 語言
  ├── 檢索: 精確查詢語言領域分數
  └── 回覆: 語言領域的標準化分數為 78 分，百分等級 PR 12…
```

---

## 6. 各模組的角色分工（修正後）

| 模組 | 負責什麼 | 不負責什麼 |
|------|---------|-----------|
| **DomainRouter** | 領域分類 + entropy | 跨輪追蹤（交給 MultiTopicTracker） |
| **TaskScopeClassifier** | 任務分類 + task_dist + task_entropy | 跨輪追蹤（不做 EMA） |
| **MultiTopicTracker** | 領域分布的跨輪追蹤 (EMA + TV) | 任務追蹤 |
| **Slot Tracker（新增）** | 槽位狀態追蹤 + 回填偵測 | 意圖分類（交給 TaskScopeClassifier） |
| **ContextSimilarity** | 語句級語意相似度 (C 值) | 領域/任務級判斷 |
| **MemoryAgent** | 記憶策略決策 (STAY/REFRESH/CLARIFY) | 領域/任務分類 |
| **PlanningAgent** | Section 選擇（抓哪些報告區塊） | 記憶/排序 |
| **RerankAgent** | 排序權重 (semantic/structural/context) | 記憶/規劃 |

---

## 7. 新增元件清單

| 元件 | 位置 | 說明 | 優先級 |
|------|------|------|--------|
| Slot Tracker | `dialogue_state_module/slot_tracker.py` | 槽位定義、狀態管理、回填偵測、跳題判斷 | 高 |
| Task Entropy | `task_scope_classifier.py` 內 | 在 `predict_task()` 中計算 `normalized_entropy(task_dist)` | 高 |
| 多任務偵測改進 | `task_scope_classifier.py` 內 | `predict_task_topk` 改用 cosine 差值判斷，取代 softmax 比值 | 高 |
| Slot 值提取器 | `dialogue_state_module/slot_extractors.py` | ability_focus / school_type / time_range 提取 | 中 |
| Prompt 追問模板 | `llm_generate_module/prompt_config.json` | 各 task 缺槽時的追問模板 | 中 |
| MemoryAgent 特徵擴展 | `semantic_flow_module_v2.py` + `memory_agent.py` | 加入 task_entropy 等（需重新訓練） | 低（等信號穩定後） |

---

## 8. 不變的部分

以下機制維持現狀，不做修改：

- 領域分類 + 領域跨輪追蹤（MultiTopicTracker EMA）
- 閒聊快篩 + Out-of-domain 偵測
- query_vector 共用機制
- prev_context 攜帶邏輯
- 三個 RL Agent 的模型結構（短期內）
- PDF 解析 → Neo4j 圖譜建置
- 對話歷史格式（dict: messages + last_retrieved_context）
