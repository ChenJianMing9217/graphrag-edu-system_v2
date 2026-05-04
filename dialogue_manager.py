from retrieval_module_v2.graph_client import GraphClient
from retrieval_module_v2 import RetrievalModuleV2
from dialogue_state_module.embedding import TextEncoder
from dialogue_state_module.semantic_flow_module_v2 import SemanticFlowClassifier
from dialogue_state_module.domain_router import DomainRouter
from llm_generate_module.llm_generator import LLMGenerator
from config import get_neo4j_uri, get_neo4j_auth, LLM_CONFIG
from rl_pipeline.agents.reranker.rerank_agent import RLAgentManager
import os
import uuid
import json
try:
    from rl_pipeline.agents.memory.memory_agent import MemoryAgent
except ImportError:
    MemoryAgent = None

try:
    from rl_pipeline.agents.planner.planning_agent import PlanningAgent
except ImportError:
    PlanningAgent = None

class DialogueManager:
    def __init__(self):
        # 0. 基本配置
        neo4j_uri = get_neo4j_uri()
        user, password = get_neo4j_auth()
        
        # 1. 初始化資料庫與基礎向量組件
        self.graph_client = GraphClient(uri=neo4j_uri, user=user, password=password)
        self.encoder = TextEncoder()
        self.generator = LLMGenerator()
        
        # 2. 載入並預先編碼領域錨點 (用於 DomainRouter)
        from dialogue_state_module.domain_anchors import DOMAINS, DOMAIN_ANCHORS, OVERVIEW_ANCHORS, load_domain_anchors
        from dialogue_state_module.embedding import encode_anchors, encode_overview_anchors
        from dialogue_state_module.domain_router import DomainRouterConfig
        
        encoded_domain_anchors = encode_anchors(self.encoder, DOMAIN_ANCHORS, DOMAINS)
        overview_vecs = encode_overview_anchors(self.encoder, OVERVIEW_ANCHORS)
        
        # 3. 初始化領域路由與分類器 (DST)
        self.domain_router = DomainRouter(
            encoder=self.encoder,
            domains=DOMAINS,
            anchor_vecs=encoded_domain_anchors,
            cfg=DomainRouterConfig()
        )

        # [FIX] 初始化任務分類器（TaskScopeClassifier），啟用 A-N 任務偵測與多任務支援
        from dialogue_state_module.task_scope_classifier import TaskScopeClassifier, load_prototypes_from_jsonl
        _task_protos, _ = load_prototypes_from_jsonl()
        self.task_scope_clf = TaskScopeClassifier(
            embedder=self.encoder,
            task_prototypes=_task_protos,
        )
        print("[INIT] TaskScopeClassifier 初始化完成（任務 A-N 分類已啟用）")
        
        # [NEW] Initialize Memory Agent (即便權重文件不存在也初始化，以便從頭訓練)
        try:
            mem_path = os.path.join(os.path.dirname(__file__), "rl_pipeline/agents/memory/models/memory_agent.pth")
            if MemoryAgent:
                self.memory_agent = MemoryAgent(model_path=mem_path)
            else:
                self.memory_agent = None
        except Exception as e:
            print(f"[DialogueManager] Memory Agent 初始化失敗: {e}")
            self.memory_agent = None

        # [NEW] Initialize Planning Agent
        try:
            plan_path = os.path.join(os.path.dirname(__file__), "rl_pipeline/agents/planner/models/planning_agent.pth")
            if os.path.exists(plan_path) and PlanningAgent:
                self.planning_agent = PlanningAgent(model_path=plan_path)
            else:
                # 即使沒模型檔案也初始化一個空的，以啟用「語義權重」邏輯
                self.planning_agent = PlanningAgent(model_path=plan_path) if PlanningAgent else None
        except Exception as e:
            print(f"[DialogueManager] Planning Agent 初始化失敗: {e}")
            self.planning_agent = None

        self.flow_classifier = SemanticFlowClassifier(
            text_encoder=self.encoder,
            domain_router=self.domain_router,
            overview_anchor_vecs=overview_vecs,
            memory_agent=self.memory_agent,
            enable_task_scope=True,          # [FIX] 啟用任務分類 A-N
            task_scope_clf=self.task_scope_clf,  # [FIX] 傳入分類器實例
        )
        
        # 4. 初始化核心 RAG 檢索器與 RL Agent
        self.retriever = RetrievalModuleV2(
            graph_client=self.graph_client,
            text_encoder=self.encoder,
            llm_generator=self.generator,
            planning_agent=self.planning_agent
        )
        
        model_path = os.path.join(os.path.dirname(__file__), "rl_pipeline/agents/reranker/models/rerank_agent.pth")
        self.rl_agent = RLAgentManager(model_path=model_path)

        # 5. 初始化 Slot Tracker
        from dialogue_state_module.slot_tracker import SlotTracker
        self.slot_tracker = SlotTracker()

        print("[INIT] DialogueManager RAG 與 DST 組件初始化完成")

    def get_response(self, user_input, child_id, session_data, all_reports=None, age_months=None, on_delta=None):
        """
        處理使用者輸入並生成回覆核心邏輯
        """
        # 0. 確定持久化身分 (用於狀態存取)
        u_id = session_data.get('user_id', 0)
        c_id = child_id if child_id else 0
        
        # A. 載入對話狀態
        self.flow_classifier.load_state(u_id, c_id)

        # 載入對話歷史 (供 QueryRewriter 改寫使用)
        history_file = os.path.join("dialogue_states", f"user_{u_id}_child_{c_id}_history.json")
        chat_history = []
        prev_retrieved_context = []
        slot_state_raw = None
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                # 向後相容：舊格式為 list，新格式為 dict
                if isinstance(history_data, list):
                    chat_history = history_data
                elif isinstance(history_data, dict):
                    chat_history = history_data.get("messages", [])
                    prev_retrieved_context = history_data.get("last_retrieved_context", [])
                    slot_state_raw = history_data.get("slot_state")
            except Exception as e:
                print(f"[DialogueManager] 讀取對話歷史失敗: {e}")

        # 載入 Slot 狀態
        self.slot_tracker.load_state(slot_state_raw)
                
        # B0. 閒聊快篩（純關鍵字，在 encode / DST 之前攔截，省掉所有後續計算）
        _CHITCHAT_KEYWORDS = ["你好", "哈囉", "嗨", "謝謝", "感謝", "再見", "掰掰", "早安", "晚安",
                               "辛苦了", "辛苦", "你好嗎", "妳好嗎", "hello", "hi", "thanks", "bye"]
        if len(user_input.strip()) <= 10 and any(kw in user_input for kw in _CHITCHAT_KEYWORDS):
            print(f"[DialogueManager] 閒聊快篩命中，跳過 DST + RAG")
            chitchat_resp = self.generator.generate_chitchat(user_input)
            return chitchat_resp, None, {}, []

        # B1. 預算 query_vector（全程共用，避免重複 encode）
        query_vector = self.encoder.encode(user_input)

        # B2. DST 分析（意圖分類與領域識別）
        flow_result = self.flow_classifier.predict(user_input, query_vec=query_vector)

        # 提取意圖與領域
        intent = flow_result.task_label or "A"
        domain = flow_result.domain_analysis.top_domain
        active_domains = flow_result.domain_analysis.active_domains

        # B2.5 Slot 回填偵測
        _task_top_prob = max(flow_result.task_dist.values()) if flow_result.task_dist else 1.0
        _task_entropy = flow_result.task_entropy if flow_result.task_entropy is not None else 0.0
        _is_slot_refill = False

        inherited_task = self.slot_tracker.detect_refill(
            user_input=user_input,
            task_label=intent,
            task_entropy=_task_entropy,
            task_top_prob=_task_top_prob,
        )
        if inherited_task is not None:
            print(f"[SlotTracker] Slot 回填偵測：繼承上輪任務 {inherited_task}（原判定 {intent}）")
            intent = inherited_task
            _is_slot_refill = True
        
        # [NEW] 印出 DST 詳細資訊
        print(f"\n{'='*80}")
        print(f"【對話狀態追蹤 (DST) 分析結果】 Turn {flow_result.turn_index}")
        print(f" 用戶輸入: '{user_input}'")
        
        # 1. 意圖分類
        print("\n【1. 任務與範圍分類】")
        task_str = {k: round(v,3) for k, v in sorted(flow_result.task_dist.items(), key=lambda x: x[1], reverse=True)[:3]} if flow_result.task_dist else {}
        scope_str = {k: round(v,3) for k, v in sorted(flow_result.scope_dist.items(), key=lambda x: x[1], reverse=True)[:3]} if flow_result.scope_dist else {}
        print(f"  推論任務: {flow_result.task_label} (分布: {task_str})")
        print(f"  推論範圍: {flow_result.scope_label} (分布: {scope_str})")
        
        # 2. 領域分析
        print("\n【2. 領域判定 (Domain Analysis)】")
        print(f"  Top Domain: {flow_result.domain_analysis.top_domain} (Prob: {flow_result.domain_analysis.top_prob:.4f})")
        print(f"  活躍領域 (Active): {flow_result.domain_analysis.active_domains}")
        print(f"  領域分布熵值 (Entropy): {flow_result.domain_analysis.entropy:.4f}")
        
        # 3. 上下文分析
        print("\n【3. 上下文與主題分析】")
        print(f"  語義相似度來源: {flow_result.context_analysis.source} (相似度: {flow_result.context_analysis.similarity_score:.4f})")
        print(f"  是否主題延續: {flow_result.topic_analysis.is_continuing} (與上輪領域重疊率: {flow_result.topic_analysis.overlap_score:.4f})")
        if flow_result.topic_analysis.tv_distance is not None:
            print(f"  TV 距離: {flow_result.topic_analysis.tv_distance:.4f} (>=0.6 代表強切換)")
            
        # 4. 決策與策略
        print("\n【4. 最終策略決策 (Policy Decision)】")
        print(f"  語義流程 (Semantic Flow): {flow_result.policy_decision.semantic_flow.upper()}")
        print(f"  建議檢索動作 (Action): {flow_result.policy_decision.retrieval_action}")
        print(f"  上下文等級 (Context Level): {flow_result.policy_decision.context_level}")
        print(f"  模糊狀態 (Ambiguous): {flow_result.policy_decision.is_ambiguous}")
        print(f"  策略代號 (Policy Case): {flow_result.policy_decision.policy_case}")
        print(f"{'='*80}\n")
        
        # 計算語義 Section 分類分數 (複用入口處的 query_vector)
        semantic_section_scores = self.retriever.strategy_mapper.ontology.get_section_matching_scores(query_vector, self.encoder)
        
        # [SAR #2] 加入領域熵到 Planning 使用的字典中
        semantic_section_scores["domain_entropy"] = flow_result.domain_analysis.entropy
        semantic_section_scores["task_label"] = intent
        
        # 更新 Session 狀態
        session_data['last_intent'] = intent
        session_data['current_domain'] = domain
        
        # 閒聊後備偵測（DST 判定，前面關鍵字快篩未命中的情況）
        if intent == 'CHITCHAT' or flow_result.policy_decision.policy_case == "CHITCHAT":
            print(f"[DialogueManager] DST 判定閒聊，跳過 RAG")
            chitchat_resp = self.generator.generate_chitchat(user_input)
            return chitchat_resp, None, {}, []

        # B2. Out-of-domain 偵測
        # 條件1：task cosine sim < 0.30（跟任何早療任務都相距甚遠）
        # 條件2：sim 稍低（< 0.40）且 softmax top-1 prob 也低（< 0.25）→ 分類器沒有把握
        _task_top_score = flow_result.task_top_score  # raw cosine similarity
        _task_top_prob  = max(flow_result.task_dist.values()) if flow_result.task_dist else 1.0
        _is_out_of_domain = (
            _task_top_score is not None
            and (
                _task_top_score < 0.30
                or (_task_top_score < 0.40 and _task_top_prob < 0.25)
            )
        )
        if _is_out_of_domain:
            print(f"[DialogueManager] Out-of-domain 偵測 (score={_task_top_score:.3f}, top_prob={_task_top_prob:.3f})，跳過 RAG")
            ood_system_prompt = (
                "使用者的問題可能超出早療輔助系統的服務範圍。"
                "請先簡短友善地回應使用者的問題（若有辦法回答），"
                "接著溫和地說明本系統專注於早期療育相關諮詢，"
                "並邀請使用者提出孩子發展或早療相關的問題。"
                "若問題隱含早療相關意涵（例如詢問費用、地點），請直接從早療角度切入回答。"
                "回答不超過三句話。"
            )
            from llm_generate_module.prompt_manager import LLMGenerationConfig
            ood_gen_config = LLMGenerationConfig(
                is_ambiguous=False,
                active_domains=[]
            )
            ood_response = self.generator.generate_response(
                user_query=user_input,
                retrieved_context=[],
                conversation_history=chat_history,
                system_prompt=ood_system_prompt,
                generation_config=ood_gen_config,
            )
            return ood_response, None, {"out_of_domain": True, "task_top_score": _task_top_score}, []

        # B5. Slot 檢查
        from dialogue_state_module.slot_extractors import extract_ability_focus, extract_school_type, extract_time_range
        _available_slot_values = {
            "region": flow_result.detected_region,
            "domain_focus": domain if domain != "整體概況" else None,
            "ability_focus": extract_ability_focus(user_input),
            "child_age": age_months,
            "school_type": extract_school_type(user_input),
            "time_range": extract_time_range(user_input),
            "report_range": "latest_2",  # 預設最近兩份
        }
        # Slot 回填時，把上輪已填的值也帶入
        if _is_slot_refill:
            for k, v in self.slot_tracker.state.filled_slots.items():
                if v is not None:
                    _available_slot_values[k] = v

        slot_result = self.slot_tracker.check_slots(
            task_label=intent,
            available_values=_available_slot_values,
            is_refill=_is_slot_refill,
        )

        if slot_result.slot_status == "has_missing":
            print(f"[SlotTracker] 缺槽位: {slot_result.missing_slots}（任務 {intent}）→ 寬範圍檢索 + 追問")
        elif slot_result.slot_status == "all_filled":
            print(f"[SlotTracker] 槽位已填滿: {slot_result.filled_slots}（任務 {intent}）→ 精確檢索")

        # C. 取得報告 doc_id (如果是進步查詢 N，則抓取最近兩份)
        doc_id = None
        if child_id:
            if intent == "N" and all_reports:
                # 取得最近兩份報告 (all_reports 已在外部按日期降序排列)
                target_reports = all_reports[:2]
                doc_id = [f"v7_report_{r.id}_{child_id}" for r in target_reports]
                print(f"[DialogueManager] 進步查詢觸發：檢索報告列表 {doc_id}")
            else:
                report_id = session_data.get('active_report_id')
                if report_id:
                    doc_id = f"v7_report_{report_id}_{child_id}"

        # D. 執行 RAG 檢索
        # 33 維特徵計算與 turn_state 構建 (對齊 v6)
        domain_dist = flow_result.domain_analysis.fused_distribution or flow_result.domain_analysis.distribution

        # fused_distribution 存在時，需區分兩種情境：
        # (1) 本輪 query 模糊被誤判為「整體概況」→ 用 fused 分布的 max 補強（v6 原意）
        # (2) STAY 沿用上一輪分布但本輪 top_domain 已明確 → 必須用本輪實際 top_domain
        #     否則會把 retrieval 拉回上一輪 domain，答非所問。
        _cur_top = flow_result.domain_analysis.top_domain
        if flow_result.domain_analysis.fused_distribution and _cur_top == "整體概況":
            fused_top_domain = max(domain_dist, key=domain_dist.get)
        else:
            fused_top_domain = _cur_top or domain

        turn_state = {
            "retrieval_action": flow_result.policy_decision.retrieval_action,
            "domain_distribution": domain_dist,
            "task_pred": intent,
            "task_dist": flow_result.task_dist or {},
            "secondary_tasks": flow_result.secondary_tasks,
            "scope_pred": flow_result.scope_label or "",
            "semantic_flow": flow_result.policy_decision.semantic_flow,
            "memory_action": flow_result.policy_decision.memory_action,
            "top_domain": fused_top_domain,
            "top_domain_prob": flow_result.domain_analysis.top_prob,
            "topic_overlap": flow_result.topic_analysis.overlap_score,
            "turn_index": flow_result.turn_index,
            "is_ambiguous": flow_result.policy_decision.is_ambiguous,
            "normalized_entropy": flow_result.domain_analysis.entropy,
            "active_domains": flow_result.domain_analysis.active_domains,
            "detected_region": flow_result.detected_region,
            "tv_distance": flow_result.topic_analysis.tv_distance,
            "context_sim": flow_result.context_analysis.similarity_score,  # Bug3 fix: Memory Agent 訓練特徵
            "is_multi_domain": flow_result.domain_analysis.is_multi_domain,
            # prev_action_norm / consecutive_stay_count 已移除（Phase 1 信號淨化）
            # 新設計：Clarify 屬性（從 action 變 attribute）
            "clarify_type": flow_result.policy_decision.clarify_type,
            "clarify_reason": flow_result.policy_decision.clarify_reason,
            "anchor_turn": flow_result.policy_decision.anchor_turn,
            "age_months": age_months,
            "task_entropy": _task_entropy,
            "slot_status": slot_result.slot_status,
            "is_slot_refill": _is_slot_refill,
        }
        
        # 2. RL 權重預測
        task_pred = flow_result.task_label or "A"
        main_domain = flow_result.domain_analysis.top_domain or "整體概況"
        scope_pred = flow_result.scope_label or "S_domain"
        
        rl_continuous = {
            "entropy": float(flow_result.domain_analysis.entropy),
            "top_prob": float(flow_result.domain_analysis.top_prob),
            "context_sim": float(flow_result.context_analysis.similarity_score),
            "topic_overlap": float(flow_result.topic_analysis.overlap_score),
            "turn_index_norm": min(float(flow_result.turn_index) / 10.0, 1.0),
            "raw_candidates": [
                {"id": r.node_id, "cat": r.properties.get("category", r.label), "text": r.text[:100]}
                for r in getattr(self.retriever, "last_debug_info", {}).get("raw_candidates", [])[:5]
            ]
        }
        
        try:
            w_semantic, w_structural, w_context = self.rl_agent.predict_weights(
                task_pred, main_domain, scope_pred, continuous=rl_continuous
            )
            rerank_config = {
                "semantic_weight": w_semantic,
                "structural_weight": w_structural,
                "context_weight": w_context
            }
            print(f"[DialogueManager] RL 預測權重: S={w_semantic:.2f}, T={w_structural:.2f}, C={w_context:.2f}")
            # 記錄實際使用的權重供離線訓練使用 (Bug4 fix)
            turn_state["rerank_w_semantic"] = w_semantic
            turn_state["rerank_w_structural"] = w_structural
            turn_state["rerank_w_context"] = w_context
        except Exception as e:
            print(f"[DialogueManager] RL 權重預測失敗，使用預設值: {e}")
            rerank_config = {"semantic_weight": 0.7, "structural_weight": 0.3}

        # 只有在有報告 ID 時才執行檢索，否則直接進入空上下文流程
        if doc_id:
            candidates = self.retriever.retrieve(
                user_query=user_input,
                turn_state=turn_state,
                doc_id=doc_id,
                rerank_config=rerank_config,
                enable_rewrite=True,
                enable_pcst=False,
                history=chat_history # 確保 QueryRewriter 具備歷史上下文
            )
        else:
            candidates = []
            print("[DialogueManager] 無有效報告 ID，跳過 RAG 檢索")

        # [DEBUG] 雙階段印出檢索結果 (供對比 Reranker 效果)
        raw_candidates = getattr(self.retriever, "last_debug_info", {}).get("raw_candidates", [])
        
        if raw_candidates or candidates:
            print("\n" + "="*80)
            print(" [A. Raw Retrieval] Top 8 (Vector Similarity Only)")
            print("-" * 80)
            for i, res in enumerate(raw_candidates[:8]):
                cat = res.properties.get("category", res.label)
                print(f"  {i+1}. [{res.score:.4f}] {cat}: {res.text[:100]}...")
            
            print("\n" + "-" * 80)
            print(" [B. Reranked Results] Top 8 (RL Agent Influenced)")
            print("-" * 80)
            for i, res in enumerate(candidates[:8]):
                cat = res.properties.get("category", res.label)
                sub = res.properties.get("subdomain", "N/A")
                print(f"  {i+1}. [{res.score:.4f}] {cat} ({sub}): {res.text[:100]}...")
            print("="*80 + "\n")
        
        # [NEW] 提取 Planning Agent 的決策日誌
        planning_info = getattr(self.retriever, "last_debug_info", {}).get("planning_info", {})
        if planning_info:
            print("\n【5. 檢索規劃 (Planning RL Agent) 決策】")
            print(f"  啟用區域: {planning_info.get('active', [])}")
            probs = planning_info.get('probs', {})
            prob_str = ", ".join([f"{k}: {v:.3f}" for k, v in probs.items()])
            print(f"  各類機率: {prob_str}")
        
        # E. 組合 Context (對齊 v6 格式，增加 path 字典以支持 PromptManager 渲染)
        retrieved_context = []
        for c in candidates[:20]:
            retrieved_context.append({
                "text": c.text,
                "score": c.score,
                "label": c.label,
                "id": c.node_id,
                "path": {
                    "subdomain": c.properties.get("subdomain", "General"),
                    "section_type": c.label,
                    "section_name": c.properties.get("category") or "N/A"
                },
                "metadata": c.metadata,
                "properties": c.properties # 保留原始屬性
            })
            
        # [NEW] 將規劃資訊併入 turn_state 以便 SQL 記錄 (供 RL 訓練)
        turn_state["planning_info"] = planning_info
        turn_state["semantic_section_scores"] = semantic_section_scores
        turn_state["num_candidates"] = len(candidates)

        # E2. 判斷是否帶入上一輪 context 作為參考背景
        #   - continue: 同主題延續 → 帶入 top-5, 權重作為參考
        #   - switch + context_sim >= 0.45: 跨主題但有語義關聯（如「這跟粗大動作有關嗎」）→ 帶入
        #   - switch + context_sim < 0.45: 乾淨切換 → 不帶
        #   - new: 全新話題 → 不帶
        prev_context_for_llm = None
        _semantic_flow = flow_result.policy_decision.semantic_flow
        _context_sim = flow_result.context_analysis.similarity_score
        if prev_retrieved_context:
            _carry_over = False
            if _semantic_flow == "continue":
                _carry_over = True
            elif _semantic_flow in ("switch", "shift_soft", "shift_hard") and _context_sim >= 0.45:
                _carry_over = True

            if _carry_over:
                # 去重：排除本輪已檢索到的相同節點
                current_ids = {c.get("id") for c in retrieved_context if c.get("id")}
                prev_context_for_llm = [
                    c for c in prev_retrieved_context[:3]
                    if c.get("id") not in current_ids
                ]
                if prev_context_for_llm:
                    print(f"[DialogueManager] 帶入上輪 context {len(prev_context_for_llm)} 筆 (flow={_semantic_flow}, ctx_sim={_context_sim:.3f})")
                else:
                    prev_context_for_llm = None

        # F. 生成回覆
        print(f"[DialogueManager] 準備生成回覆。Intent: {intent}, Domain: {domain}, Context 筆數: {len(retrieved_context)}")
        
        # [NEW] 構建 generation_config 以便處理模糊引導 (Ambiguity)
        from llm_generate_module.prompt_manager import LLMGenerationConfig
        is_ambiguous = flow_result.policy_decision.is_ambiguous

        # 如果策略建議 CLARIFY，強行標記為模糊（保留相容）
        if flow_result.policy_decision.retrieval_action in ("DUAL_OR_CLARIFY", "LOCAL_RESOURCE_CLARIFY"):
            is_ambiguous = True

        # 新設計 (Phase D)：傳遞 clarify_type / reason 供 prompt 層使用
        gen_config = LLMGenerationConfig(
            is_ambiguous=is_ambiguous,
            active_domains=flow_result.domain_analysis.active_domains,
            clarify_type=flow_result.policy_decision.clarify_type,
            clarify_reason=flow_result.policy_decision.clarify_reason,
        )

        # 決定系統提示詞 (根據資料缺失狀況動態調整)
        custom_system_prompt = None
        retrieval_action = flow_result.policy_decision.retrieval_action

        if not doc_id:
            custom_system_prompt = (
                "你是一位專業的早療系統助手。使用者目前尚未上傳或選取評估報告，請根據您的專業醫學知識回答使用者的一般性問題。"
                "在回答中請務必包含：1. 專業但易懂的知識補充 2. 溫馨提醒因為缺乏具體數據，您的建議僅供參考 3. 鼓勵使用者點擊上傳圖示或輸入存取碼，以獲得更精準的分析。"
            )
        elif retrieval_action == "LOCAL_RESOURCE_CLARIFY":
            # Task H / Task K：問到在地資源或補助但未偵測到縣市 → 主動詢問地區
            if intent == "K":
                custom_system_prompt = (
                    f"使用者詢問了早療補助或福利申請（『{user_input}』）。"
                    "各縣市補助方案和申請窗口不同，請先溫和地詢問家長所在的縣市或地區，"
                    "例如：『請問您目前在哪個縣市呢？這樣我可以提供當地的早療補助方案與申請辦法。』"
                    "如果系統已提供部分資訊，請一併整理呈現。"
                )
            else:
                custom_system_prompt = (
                    f"使用者詢問了在地早療或復健資源（『{user_input}』）。"
                    "請先溫和地詢問家長所在的縣市或地區，例如：『請問您目前在哪個縣市呢？這樣我可以幫您查詢附近的早療機構、物理治療所或相關資源。』"
                    "如果系統已提供部分資源清單，請一併整理呈現，並說明這是目前可找到的資訊，提供縣市後可以給出更精確的結果。"
                )
        elif is_ambiguous:
            # 模糊情況下的引導提示
            custom_system_prompt = (
                f"使用者詢問的『{user_input}』語意較為模糊或橫跨多個領域。"
                f"請在回答開頭先溫和地詢問家長：『您是指關於 {domain} 方面，還是其他部分（如：粗大動作、口語表達）？』"
                f"然後根據目前最可能的領域 {domain} 給予初步的專業解釋。"
            )
        elif not retrieved_context:
             custom_system_prompt = (
                f"你是一位專業的早療系統助手。使用者詢問了關於『{domain}』的問題，但您在目前的報告中並未發現相關紀錄。"
                "請根據您的醫學知識給予一般性建議，並主動詢問家長是否在日常生活中觀察到這方面的困難，引導家長提供更多生活細節以便分析。"
            )

        # Slot 追問：缺槽時附加追問提示（不覆蓋已有的 system_prompt，而是補充）
        if slot_result.followup_hint and slot_result.slot_status == "has_missing":
            _slot_instruction = (
                f"\n\n【追問指引】請在回答的結尾，用溫和自然的語氣追問以下資訊："
                f"\n{slot_result.followup_hint}"
            )
            if custom_system_prompt:
                custom_system_prompt += _slot_instruction
            else:
                custom_system_prompt = (
                    f"你是一位專業的早療系統助手，正在回答使用者關於『{domain}』的問題。"
                    f"請根據檢索到的資料回答，並在結尾自然地追問。"
                    f"{_slot_instruction}"
                )

        response = self.generator.generate_response(
            user_query=user_input,
            retrieved_context=retrieved_context,
            conversation_history=chat_history, # 傳遞給 Prompt 生成
            system_prompt=custom_system_prompt,
            generation_config=gen_config,
            prev_context=prev_context_for_llm,
            on_delta=on_delta,  # streaming callback (None=非 streaming)
        )
        
        # G. 更新並保存對話狀態 (為了下一輪的 ContextSimilarity)
        # 手動更新 Bot 回覆到狀態中
        self.flow_classifier.context_similarity.update(user_input, response)
        self.flow_classifier.save_state(u_id, c_id)

        # 更新 Slot 狀態（供下一輪回填偵測）
        self.slot_tracker.update_pending(intent, slot_result)

        # 更新並保存對話歷史
        msg_uuid = str(uuid.uuid4())
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response, "id": msg_uuid, "feedback": 0})
        if len(chat_history) > 10: # 保留最近 5 輪 (10 筆)
            chat_history = chat_history[-10:]

        # 保存本輪 top-5 retrieved_context 供下一輪參考
        save_context_snapshot = [
            {"id": c.get("id"), "text": c.get("text", "")[:300], "label": c.get("label", ""),
             "score": c.get("score", 0.0),
             "path": c.get("path", {})}
            for c in retrieved_context[:5]
        ] if retrieved_context else []

        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "messages": chat_history,
                    "last_retrieved_context": save_context_snapshot,
                    "slot_state": self.slot_tracker.save_state(),
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[DialogueManager] 儲存對話歷史失敗: {e}")
        
        return response, msg_uuid, turn_state, retrieved_context

    def reset_session(self, child_id, session_data):
        """重置對話狀態"""
        # 1. 清除 Session 資料
        session_data.pop('last_intent', None)
        session_data.pop('current_domain', None)
        
        # 2. 刪除持久化檔案
        from dialogue_state_module.state_persistence import delete_dialogue_state
        u_id = session_data.get('user_id', 0)
        c_id = child_id if child_id else 0
        delete_dialogue_state(u_id, c_id)
        
        # [NEW] 刪除對話歷史
        import os
        history_file = os.path.join("dialogue_states", f"user_{u_id}_child_{c_id}_history.json")
        if os.path.exists(history_file):
            try:
                os.remove(history_file)
            except Exception:
                pass
        
        # 3. 重置內存分類器狀態
        self.flow_classifier.reset()

        # 4. 重置 Slot 狀態
        self.slot_tracker.load_state(None)

        print(f"[DialogueManager] 已重置用戶 {u_id} 與兒童 {c_id} 的對話狀態")
        return True
