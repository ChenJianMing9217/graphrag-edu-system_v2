from typing import Dict, Any, List, Optional, Tuple
from .types import SearchStrategy, SearchOperation, SearchOperationType
from .topic_ontology import TopicOntology

class StrategyMapper:
    """
    Translates DST FlowResult signals into a SearchStrategy.
    """
    def __init__(self, ontology: TopicOntology, text_encoder: Any, planning_agent: Any = None):
        self.ontology = ontology
        self.text_encoder = text_encoder
        self.planning_agent = planning_agent

    def map_dst_to_strategy(self, turn_state: Dict[str, Any], user_query: str = "", query_vector: Optional[List[float]] = None, text_encoder: Any = None, chat_history: Optional[List[Dict]] = None) -> SearchStrategy:
        strategy = SearchStrategy()
        reasons = []
        
        retrieval_action = turn_state.get("retrieval_action", "")
        scope_label = turn_state.get("scope_pred", "")
        domain_distribution = turn_state.get("domain_distribution", {})
        task_label = turn_state.get("task_pred", "")
        active_domains = turn_state.get("active_domains", [])
        clarify_type = turn_state.get("clarify_type")  # 新設計：clarify 屬性

        reasons.append(f"DST Action: {retrieval_action}")
        reasons.append(f"Scope: {scope_label}")
        if clarify_type:
            reasons.append(f"ClarifyType: {clarify_type}")
        
        # 1. Summary Fetch (If Meta/Summary query or "整體概況" domain)
        is_overview_domain = "整體概況" in active_domains
        if is_overview_domain or scope_label == "S_overview" or task_label == "T_overview":
            strategy.operations.append(SearchOperation(
                op_type=SearchOperationType.SUMMARY_FETCH,
                params={"query_type": "summary", "limit": 10}
            ))
            reasons.append("Added SUMMARY_FETCH due to overview scope/task")

        # 2. Meta Fetch (If name/age/date mentioned)
        meta_keywords = ["姓名", "年齡", "日期", "性別", "個案"]
        if any(kw in user_query for kw in meta_keywords) or task_label == "T_meta_query":
            strategy.operations.append(SearchOperation(
                op_type=SearchOperationType.META_FETCH,
                params={}
            ))
            reasons.append("Added META_FETCH due to meta-related keywords or task")

        # 3. Subdomain Fetch (Based on domain distribution and task weights)
        
        # 1. Determine sections to fetch (Logic: RL Planning Agent vs. Static Ontology)
        if query_vector is None:
            query_vector = self.text_encoder.encode(user_query)
        semantic_section_scores = self.ontology.get_section_matching_scores(query_vector, self.text_encoder)
        
        use_sections = []
        planning_info = {}
        
        if self.planning_agent:
            # Use RL Planning Agent（含多任務 secondary_tasks + task_dist）
            planning_state = {f"sem_{k}": v for k, v in semantic_section_scores.items()}
            planning_state["task_label"]      = task_label
            planning_state["secondary_tasks"] = turn_state.get("secondary_tasks", [])
            planning_state["task_dist"]       = turn_state.get("task_dist", {})
            planning_state["domain_entropy"]  = float(turn_state.get("normalized_entropy", 0.0))
            # deterministic=True：線上推論固定用門檻決策，避免 Bernoulli 採樣造成兩次呼叫結果不一致
            planning_res = self.planning_agent.select_sections(planning_state, deterministic=True)
            use_sections = planning_res["active"]
            planning_info = planning_res
            reasons.append(f"Planning Agent selected: {use_sections}")
            
            # 處理新增的兩個額外模塊 (採用非破壞性檢查，避免影響原始 planning_res)
            if "community_resources" in use_sections:
                strategy.operations.append(SearchOperation(
                    op_type=SearchOperationType.MYSQL_RESOURCE_FETCH,
                    params={"query": user_query, "region": turn_state.get("detected_region", "")}
                ))
                reasons.append("Added MYSQL_RESOURCE_FETCH for community units")
                
            if "external_gpt" in use_sections:
                strategy.operations.append(SearchOperation(
                    op_type=SearchOperationType.GPT_FETCH,
                    params={"query": user_query, "chat_history": chat_history or []}
                ))
                reasons.append("Added GPT_FETCH based on Planning Agent decision")

            # 在傳遞給後續圖譜查詢 (Neo4j) 前，過濾掉非圖譜標籤
            graph_sections = [s for s in use_sections if s not in ["community_resources", "external_gpt"]]
            use_sections = graph_sections # 僅在本地作用域更新指標，不影響 planning_res['active']
        else:
            # Fallback to Static Ontology weights（支援多任務 section 合併）
            task_dist  = turn_state.get("task_dist", {})
            all_tasks  = [task_label] + [t for t in turn_state.get("secondary_tasks", []) if t != task_label]

            if len(all_tasks) > 1:
                # 多任務：以各 task 的 softmax 機率做加權平均
                total_w = sum(task_dist.get(t, 0.05) for t in all_tasks)
                merged: Dict[str, float] = {}
                for t in all_tasks:
                    w = task_dist.get(t, 0.05) / max(total_w, 1e-9)
                    for sec, sw in self.ontology.get_section_weights(t).items():
                        merged[sec] = merged.get(sec, 0.0) + sw * w
                use_sections = [sec for sec, w in merged.items() if w > 0.08]
                reasons.append(f"Multi-task Ontology merged {all_tasks} → sections: {use_sections}")
            else:
                section_weights = self.ontology.get_section_weights(task_label)
                use_sections = [sec for sec, weight in section_weights.items() if weight > 0]
                reasons.append(f"Ontology Weights selected: {use_sections}")

            # 保底：確保至少有基本四個 section
            if not use_sections or len(use_sections) < 3:
                use_sections = ["assessment", "observation", "training", "suggestion"]

        # Determine query_domains based on retrieval_action (Memory Agent breadth decision)
        top_domain = turn_state.get("top_domain", "")
        if retrieval_action == "NARROW_GRAPH":
            # 強延續：只抓最相關的 1 個 domain
            if top_domain and top_domain != "整體概況":
                query_domains = [top_domain]
            else:
                query_domains = active_domains[:1] if active_domains else []
        elif retrieval_action == "CONTEXT_FIRST":
            # 上下文優先：取機率最高的 2 個 domain
            sorted_d = sorted(domain_distribution.items(), key=lambda x: x[1], reverse=True)
            query_domains = [d for d, _ in sorted_d[:2] if d != "整體概況"]
            if not query_domains:
                query_domains = active_domains
        elif retrieval_action in ("DUAL_OR_CLARIFY", "LOCAL_RESOURCE_CLARIFY"):
            # 模糊/需澄清：全域搜尋（舊行為，保留作 fallback）
            query_domains = active_domains
        else:
            # WIDE_IN_DOMAIN 或其他：預設全域
            query_domains = active_domains

        # ========================================================================
        # Clarify Type 驅動的檢索策略調整（新設計）
        # 目的：根據「為什麼需要澄清」細緻調整檢索範圍，而非一刀切 DUAL_OR_CLARIFY
        # ========================================================================
        if clarify_type == "DOMAIN_HARD":
            # 極模糊：不做特別檢索（生成層會單純追問）
            # 保守作法：維持原 query_domains，避免 domain 猜錯
            reasons.append("DOMAIN_HARD: 極模糊 query，生成層純追問")
        elif clarify_type == "CONTEXT_MISSING":
            # T0 接續語但無歷史：放寬 domain 範圍，拉取前 2 個候選
            sorted_d = sorted(domain_distribution.items(), key=lambda x: x[1], reverse=True)
            expanded = [d for d, _ in sorted_d[:2] if d != "整體概況"]
            if expanded:
                query_domains = list(set(query_domains + expanded))
            reasons.append(f"CONTEXT_MISSING: 擴展 query_domains 至 {query_domains}")
        elif clarify_type == "SLOT_REGION":
            # 缺地區：提高 community_resources / external_gpt 相關性
            # 在 semantic_section_scores 上加權，讓 rerank 偏向通用資源
            if "community_resources" in semantic_section_scores:
                semantic_section_scores["community_resources"] = min(1.0, semantic_section_scores["community_resources"] * 1.3)
            if "external_gpt" in semantic_section_scores:
                semantic_section_scores["external_gpt"] = min(1.0, semantic_section_scores["external_gpt"] * 1.3)
            reasons.append("SLOT_REGION: 提升 community_resources + external_gpt 權重")
        elif clarify_type == "TASK_SOFT":
            # 多任務：確保 secondary_tasks 的 section 也在 use_sections 中
            secondary_tasks = turn_state.get("secondary_tasks", [])
            if secondary_tasks:
                try:
                    for t in secondary_tasks:
                        sec_w = self.ontology.get_section_weights(t)
                        for sec, w in sec_w.items():
                            if w > 0.1 and sec not in use_sections and sec not in ("community_resources", "external_gpt"):
                                use_sections.append(sec)
                    reasons.append(f"TASK_SOFT: 擴展 use_sections 涵蓋 {secondary_tasks} → {use_sections}")
                except Exception:
                    pass

        # 跳過 SUBDOMAIN_FETCH 的條件：
        #   1. retrieval_action == LOCAL_RESOURCE_SEARCH：純在地資源/補助查詢，不需要拉報告內容
        #   2. use_sections 為空：Planning Agent 只選了 community_resources/external_gpt，
        #      沒有任何 Neo4j 圖譜 section 需要檢索（拉了也會使用 fallback 全欄位，造成雜訊）
        _skip_subdomain = (
            retrieval_action == "LOCAL_RESOURCE_SEARCH"
            or len(use_sections) == 0
        )

        if query_domains and not _skip_subdomain:
            for domain in query_domains:
                # 跳過虛擬的「整體概況」領域，因為圖資中沒有對應標籤 (由 SUMMARY_FETCH 處理)
                if domain == "整體概況":
                    continue
                prob = domain_distribution.get(domain, 0.0)
                strategy.operations.append(SearchOperation(
                    op_type=SearchOperationType.SUBDOMAIN_FETCH,
                    params={"subdomain": domain, "sections": use_sections},
                    weight=prob if prob > 0 else 1.0
                ))
            reasons.append(f"retrieval_action={retrieval_action} → SUBDOMAIN_FETCH for domains: {query_domains} with sections: {use_sections}")
        elif _skip_subdomain:
            reasons.append(f"SUBDOMAIN_FETCH skipped (retrieval_action={retrieval_action}, graph_sections={use_sections})")

        # 4. MySQL Local Resource Fetch（若 Planning Agent 已加過 MYSQL_RESOURCE_FETCH 則跳過）
        _already_has_mysql = any(op.op_type == SearchOperationType.MYSQL_RESOURCE_FETCH for op in strategy.operations)
        if retrieval_action == "LOCAL_RESOURCE_SEARCH" and not _already_has_mysql:
            region = turn_state.get("detected_region")
            if region:
                keywords = None
                for kw in ["物理治療", "語言治療", "職能治療", "心理治療", "療育", "評估"]:
                    if kw in user_query:
                        keywords = kw
                        break

                strategy.operations.append(SearchOperation(
                    op_type=SearchOperationType.MYSQL_RESOURCE_FETCH,
                    params={"region": region, "keywords": keywords}
                ))
                reasons.append(f"Added MYSQL_RESOURCE_FETCH for region: {region}, keywords: {keywords}")

        # 5. LOCAL_RESOURCE_SEARCH 強制保底：必有 GPT_FETCH（external_gpt）+ 強制 web_search
        #    避免 planning agent 對短 query（如「台中市」）勾錯 sections，
        #    或 _should_use_web_search 因 query 太短無 resource keyword 而誤判，
        #    導致 LLM 只能靠記憶幻覺出假機構。
        if retrieval_action == "LOCAL_RESOURCE_SEARCH":
            _already_has_gpt = any(op.op_type == SearchOperationType.GPT_FETCH for op in strategy.operations)
            if not _already_has_gpt:
                strategy.operations.append(SearchOperation(
                    op_type=SearchOperationType.GPT_FETCH,
                    params={
                        "query": user_query,
                        "chat_history": chat_history or [],
                        "force_web_search": True,  # 強制走 web_search_preview
                    }
                ))
                reasons.append("LOCAL_RESOURCE_SEARCH 保底：強制 GPT_FETCH + web_search")
            else:
                # 已有 GPT_FETCH，把它升級為 force_web_search
                for op in strategy.operations:
                    if op.op_type == SearchOperationType.GPT_FETCH:
                        op.params["force_web_search"] = True
                        reasons.append("LOCAL_RESOURCE_SEARCH 保底：升級既有 GPT_FETCH 為 force_web_search")
                        break

        # 6. SLOT_REGION 防幻覺：H/K 任務缺地區時，移除 GPT_FETCH。
        #    chat.completions 即使 system prompt 禁止列機構，仍可能違規幻覺，
        #    缺地區直接讓 LLM 用個案資料 + ClinicalNorm 引導使用者補縣市。
        if clarify_type == "SLOT_REGION":
            n_before = len(strategy.operations)
            strategy.operations = [
                op for op in strategy.operations
                if op.op_type != SearchOperationType.GPT_FETCH
            ]
            n_removed = n_before - len(strategy.operations)
            if n_removed > 0:
                reasons.append(f"SLOT_REGION 防幻覺：移除 {n_removed} 個 GPT_FETCH（避免 LLM 編出假機構）")

        # Rerank Config
        strategy.rerank_config.update({
            "semantic_weight": 0.6,
            "structural_weight": 0.2,
            "context_weight": 0.2,
            "planning_info": planning_info
        })
        
        strategy.semantic_section_scores = semantic_section_scores
        strategy.reasons = reasons
        return strategy
