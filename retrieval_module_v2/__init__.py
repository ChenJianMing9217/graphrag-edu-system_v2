from .strategy_mapper import StrategyMapper
from .execution_engine import ExecutionEngine
from .reranker import Reranker
from .query_rewriter import QueryRewriter
from .pcst_solver import PCSTSolver
from .types import SearchStrategy, SearchOperation, SearchOperationType, CandidateNode
from typing import List, Dict, Any

class RetrievalModuleV2:
    def __init__(self, graph_client, sql_db=None, text_encoder=None, llm_generator=None, planning_agent=None):
        from .topic_ontology import default_ontology
        self.graph_client = graph_client
        self.sql_db = sql_db
        self.text_encoder = text_encoder
        self.strategy_mapper = StrategyMapper(
            ontology=default_ontology, 
            text_encoder=text_encoder, 
            planning_agent=planning_agent
        )
        self.execution_engine = ExecutionEngine(graph_client, sql_db=sql_db, text_encoder=text_encoder)
        self.reranker = Reranker(text_encoder)
        self.query_rewriter = QueryRewriter(llm_generator=llm_generator)
        self.pcst_solver = PCSTSolver(graph_client)

    def retrieve(self, user_query: str, turn_state: Dict[str, Any], doc_id: str = None, rerank_config: Dict[str, float] = None, enable_rewrite: bool = True, enable_pcst: bool = True, history: List[Dict[str, str]] = None) -> List[CandidateNode]:
        """
        Retrieves candidate nodes and optionally finds a connected sub-graph, 
        using dialogue history for contextualization.
        """
        print(f"\n[RetrievalModuleV2] === 開始檢視流程 (Query: {user_query}) ===")
        
        # 1. Query Rewriting (Reciprocal Rank Fusion - RRF)
        queries = [user_query]
        condensed = user_query
        if enable_rewrite:
            condensed, rewritten = self.query_rewriter.rewrite(user_query, turn_state, history=history)
            for q in rewritten:
                if q not in queries:
                    queries.append(q)
            print(f"[RetrievalModuleV2] 改寫後的查詢列表: {queries}")

        # --- 第一階段：初步檢索 (Initial Report/Resource Retrieval) ---
        all_results_lists = []
        primary_strategy = None  # 保留第一次（原始 query）的 strategy 供 enrichment 複用
        for i, q in enumerate(queries):
            q_vector = self.text_encoder.encode(q)
            strategy = self.strategy_mapper.map_dst_to_strategy(
                turn_state,
                user_query=q,
                query_vector=q_vector,
                text_encoder=self.text_encoder,
                chat_history=history,
            )
            if i == 0:
                primary_strategy = strategy

            # 僅抓取基礎個案資料
            candidates = self.execution_engine.execute_initial(strategy, doc_id)
            for c in candidates:
                c.metadata["semantic_section_scores"] = strategy.semantic_section_scores

            all_results_lists.append(candidates)
            print(f"[RetrievalModuleV2] 查詢 {i} ({q}) 找到了 {len(candidates)} 個基礎候選節點")

        if len(queries) > 1:
            fused_candidates = self._reciprocal_rank_fusion(all_results_lists)
        else:
            fused_candidates = all_results_lists[0] if all_results_lists else []

        # --- 稀疏檢索擴展 (Sparse Retrieval Fallback) ---
        # 當 CLARIFY / 模糊延續鎖定的 domain 結果太少時，擴展到 domain_distribution 的 top-2
        _SPARSE_THRESHOLD = 3
        _is_sparse = len(fused_candidates) < _SPARSE_THRESHOLD
        _memory_action = turn_state.get("memory_action", "")
        if _is_sparse and _memory_action in ("CLARIFY", "STAY"):
            domain_dist = turn_state.get("domain_distribution", {})
            current_domains = set(turn_state.get("active_domains", []))
            # 從 domain_distribution 取 top-2 且不在現有 active_domains 中的 domain
            sorted_domains = sorted(domain_dist.items(), key=lambda x: x[1], reverse=True)
            expand_domains = [d for d, _ in sorted_domains if d not in current_domains and d != "整體概況"][:2]
            if expand_domains:
                print(f"[RetrievalModuleV2] 稀疏擴展：原始 {len(fused_candidates)} 筆不足，補充 domain: {expand_domains}")
                for eq in queries[:1]:  # 只用原始 query 做擴展，避免重複
                    eq_vector = self.text_encoder.encode(eq)
                    expand_state = dict(turn_state)
                    expand_state["active_domains"] = expand_domains
                    expand_state["retrieval_action"] = "WIDE_IN_DOMAIN"
                    expand_strategy = self.strategy_mapper.map_dst_to_strategy(
                        expand_state, user_query=eq, query_vector=eq_vector,
                        text_encoder=self.text_encoder, chat_history=history,
                    )
                    expand_candidates = self.execution_engine.execute_initial(expand_strategy, doc_id)
                    # 用較低的 base score 標記擴展結果
                    for c in expand_candidates:
                        c.score *= 0.8
                        c.metadata["semantic_section_scores"] = expand_strategy.semantic_section_scores
                        c.metadata["from_sparse_expansion"] = True
                    fused_candidates.extend(expand_candidates)
                print(f"[RetrievalModuleV2] 稀疏擴展後共 {len(fused_candidates)} 筆候選節點")

        # --- 第一點五階段：中間篩選 (Intermediate Rerank & Diversity Selection) ---
        task_label = turn_state.get("task_pred")
        domain_dist = turn_state.get("domain_distribution")
        initial_rerank_config = {"semantic_weight": 0.7, "structural_weight": 0.3}
        
        ranked_initial = self.reranker.rerank(
            fused_candidates, 
            user_query, 
            initial_rerank_config,
            task_label=task_label,
            domain_distribution=domain_dist
        )
        
        # [NEW] 分類萃取：確保行為、目標、建議都能參與臨床對接
        def get_top_by_label(nodes, label, top_n=4):
            return [n for n in nodes if n.label == label][:top_n]
        
        obs_context = get_top_by_label(ranked_initial, "Observation", 4)
        train_context = get_top_by_label(ranked_initial, "TrainingDirection", 4)
        rec_context = get_top_by_label(ranked_initial, "Recommendation", 4) + get_top_by_label(ranked_initial, "GeneralRecommendation", 2)
        
        top_k_for_enrichment = obs_context + train_context + rec_context
        print(f"[RetrievalModuleV2] 選取 {len(top_k_for_enrichment)} 筆多維度觀察進行臨床對話...")

        # --- 第二階段：知識增強 (Enrichment Stage) --- (複用第一階段的 primary_strategy)
        if "planning_info" in primary_strategy.rerank_config:
            turn_state["planning_info"] = primary_strategy.rerank_config["planning_info"]

        enriched_candidates = self.execution_engine.execute_enrichment(
            primary_strategy,
            context_nodes=top_k_for_enrichment,
            user_query=user_query,
            age_months=turn_state.get("age_months"),
        )
        
        # --- 第三階段：最終整合 (Final Integration) ---
        # ExternalGPT 節點強制置頂（必須傳給 LLM generator，不受排序影響）
        # 其餘 enriched（ClinicalNorm 等）接在後面，最後是個案資料
        gpt_nodes = [n for n in enriched_candidates if n.label == "ExternalGPT"]
        other_enriched = [n for n in enriched_candidates if n.label != "ExternalGPT"]
        final_results = gpt_nodes + other_enriched + ranked_initial
        
        print(f"[RetrievalModuleV2] 最終整合完成 (共 {len(final_results)} 個節點)，專家知識已優先排序。")
        
        # 3. PCST Sub-graph Discovery (Contextual Connection)
        if enable_pcst and final_results:
            try:
                subgraph_edges = self.pcst_solver.find_subgraph(final_results, doc_id)
                if subgraph_edges:
                    # Attach subgraph info to the top node's metadata
                    final_results[0].metadata["subgraph_context"] = subgraph_edges
                    print(f"[RetrievalModuleV2] Found subgraph with {len(subgraph_edges)} relationships.")
            except Exception as e:
                print(f"[RetrievalModuleV2] PCST Error: {e}")
        # Store debug info for offline analysis
        self.last_debug_info = {
            "user_query": user_query,
            "condensed_query": condensed,
            "rewritten_queries": queries if enable_rewrite else [user_query],
            "num_candidates": len(final_results),
            "rerank_config": initial_rerank_config,
            "planning_info": primary_strategy.rerank_config.get("planning_info", {}),
            "raw_candidates": fused_candidates # 暫存重排前的結果
        }
        
        return final_results

    def _reciprocal_rank_fusion(self, results_lists: List[List[CandidateNode]], k: int = 60) -> List[CandidateNode]:
        """
        Implements Reciprocal Rank Fusion (RRF) to merge multiple result lists.
        """
        fused_scores = {}  # node_id -> score
        node_map = {}      # node_id -> CandidateNode (sample)

        for results in results_lists:
            for rank, node in enumerate(results):
                node_id = node.node_id
                if node_id not in fused_scores:
                    fused_scores[node_id] = 0.0
                    node_map[node_id] = node
                
                # RRF Formula: 1 / (k + rank)
                fused_scores[node_id] += 1.0 / (k + rank)

        # Create new list of candidates with fused scores
        fused_candidates = []
        for node_id, score in fused_scores.items():
            node = node_map[node_id]
            # Create a shallow copy to avoid modifying original node scores if they are shared
            new_node = CandidateNode(
                node_id=node.node_id,
                label=node.label,
                text=node.text,
                properties=node.properties,
                score=score,
                metadata=node.metadata
            )
            fused_candidates.append(new_node)

        # Sort by fused score
        fused_candidates.sort(key=lambda x: x.score, reverse=True)
        return fused_candidates
