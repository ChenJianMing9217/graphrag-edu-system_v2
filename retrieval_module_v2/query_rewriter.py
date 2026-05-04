from typing import List, Dict, Any
from llm_generate_module.llm_generator import LLMGenerator, LLMConfig
from llm_generate_module.prompt_manager import LLMPromptManager

class QueryRewriter:
    """
    QueryRewriter uses an LLM to generate multiple versions of a user query
    to improve retrieval recall and precision.
    """
    def __init__(self, llm_generator: LLMGenerator = None):
        if llm_generator is None:
            config = LLMConfig()
            self.llm_generator = LLMGenerator(config=config)
        else:
            self.llm_generator = llm_generator
        
        self.prompt_manager = LLMPromptManager()

    def rewrite(self, query: str, turn_state: Dict[str, Any], history: List[Dict[str, str]] = None) -> tuple[str, List[str]]:
        """
        Rewrites a query into multiple variations, optionally using dialogue history.
        
        Args:
            query: The original user query.
            turn_state: Current DST state and dialogue context.
            history: List of previous dialogue turns [{"role": "user/assistant", "content": "..."}].
            
        Returns:
            A tuple of (condensed_query, rewritten_queries_list).
        """
        # 1. Query Condensation (Contextualization)
        # If the flow is 'continue' and we have history, we first condense the query to a standalone version.
        semantic_flow = turn_state.get("semantic_flow", "shift_soft")
        working_query = query
        
        if history and semantic_flow == "continue":
            working_query = self._condense_query(query, history)
            print(f"[QueryRewriter] Condensed Query: {working_query}")

        # 2. Prepare context for rewriting
        context_str = self._format_context(turn_state)
        domain = turn_state.get("top_domain", "未知")
        task = turn_state.get("task_pred", "一般查詢")
        
        # 3. Build Rewrite Prompt
        prompt = self.prompt_manager.QUERY_REWRITE_PROMPT.format(
            query=working_query,
            context=context_str,
            domain=domain,
            task=task
        )
        
        # 4. Call LLM for Rewriting
        try:
            messages = [
                {"role": "system", "content": "你是一個專業的查詢改寫助手。"},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_generator.client.chat.completions.create(
                model=self.llm_generator.config.model,
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            
            raw_text = response.choices[0].message.content.strip()
            
            # 5. Parse results
            rewritten_queries = [line.strip() for line in raw_text.split("\n") if line.strip()]
            
            # Add the (potentially condensed) working query to the list
            if working_query not in rewritten_queries:
                rewritten_queries.insert(0, working_query)
            
            # Also add the original raw query if it's different and meaningful
            if query != working_query and query not in rewritten_queries:
                rewritten_queries.append(query)
                
            return working_query, rewritten_queries[:2]
            
        except Exception as e:
            print(f"[QueryRewriter] Rewrite Error: {e}")
            return working_query, [query]

    def _condense_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """Condenses the user query and history into a standalone query."""
        try:
            # Format history for the prompt
            history_str = ""
            for turn in history[-5:]:  # Only take last 5 turns for context
                role = "家長" if turn["role"] == "user" else "助手"
                history_str += f"{role}: {turn['content']}\n"
            
            prompt = self.prompt_manager.QUERY_CONDENSE_PROMPT.format(
                history=history_str,
                query=query
            )
            
            messages = [
                {"role": "system", "content": "你是一個專業的對話脈絡分析助手。"},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_generator.client.chat.completions.create(
                model=self.llm_generator.config.model,
                messages=messages,
                temperature=0.2,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[QueryRewriter] Condense Error: {e}")
            return query

    def _format_context(self, turn_state: Dict[str, Any]) -> str:
        """Formats turn_state into a brief context string for the LLM."""
        return f"回合數: {turn_state.get('turn_index', 0)} | 意圖流向: {turn_state.get('semantic_flow', 'N/A')}"
