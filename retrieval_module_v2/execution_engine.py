from typing import List, Dict, Any
from .types import SearchOperation, SearchOperationType, CandidateNode
from .mysql_client import MySQLResourceClient
import json

# 安全載入 ClinicalBridgeService
try:
    import sys
    import os
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.join(base_path, 'knowledge_graph_extra'))
    from clinical_api import ClinicalBridgeService  # type: ignore
    HAS_CLINICAL_API = True
except Exception as e:
    print(f"[ExecutionEngine] Warning: ClinicalBridgeService not available. fallback to safe mode. Error: {e}")
    HAS_CLINICAL_API = False

class ExecutionEngine:
    """
    Executes SearchOperations against Neo4j.
    """
    def __init__(self, graph_client, sql_db=None, text_encoder=None):
        self.graph_client = graph_client
        self.mysql_client = MySQLResourceClient(sql_db) if sql_db else None
        self.text_encoder = text_encoder
        
        # 安全初始化 Clinical API
        self.clinical_service = None
        if HAS_CLINICAL_API:
            try:
                # [NEW] 傳入整合過的 TextEncoder
                self.clinical_service = ClinicalBridgeService(encoder=text_encoder)
            except Exception as e:
                print(f"[ExecutionEngine] Failed to init ClinicalBridgeService: {e}")

    def execute_strategy(self, strategy, doc_id_or_ids: Any) -> List[CandidateNode]:
        """
        [DEPRECATED] Use execute_initial and execute_enrichment for better precision.
        """
        initial = self.execute_initial(strategy, doc_id_or_ids)
        enriched = self.execute_enrichment(strategy, context_nodes=initial)
        return initial + enriched

    def execute_initial(self, strategy, doc_id_or_ids: Any) -> List[CandidateNode]:
        """
        第一階段：僅從 Neo4j / MySQL 抓取個案原始資料。
        """
        all_candidates = []
        
        # 統一轉換為列表格式
        if isinstance(doc_id_or_ids, list):
            doc_ids = [str(x) for x in doc_id_or_ids]
        else:
            doc_ids = [str(doc_id_or_ids)]

        # 僅過濾 Neo4j (個案報告) 與 MySQL (資源)
        neo4j_ops = [op for op in strategy.operations if op.op_type not in [SearchOperationType.MYSQL_RESOURCE_FETCH, SearchOperationType.CLINICAL_FETCH, SearchOperationType.GPT_FETCH]]
        mysql_ops = [op for op in strategy.operations if op.op_type == SearchOperationType.MYSQL_RESOURCE_FETCH]
        
        # 1. 執行 Neo4j 基本操作 (個案報告)
        if neo4j_ops:
            with self.graph_client.driver.session(database=self.graph_client.database) as session:
                for op in neo4j_ops:
                    if op.op_type == SearchOperationType.SUBDOMAIN_FETCH:
                        all_candidates.extend(self._fetch_subdomain(session, op.params, doc_ids))
                    elif op.op_type == SearchOperationType.SUMMARY_FETCH:
                        all_candidates.extend(self._fetch_summary(session, op.params, doc_ids))
                    elif op.op_type == SearchOperationType.META_FETCH:
                        all_candidates.extend(self._fetch_meta(session, op.params, doc_ids))
        
        # 2. 執行 MySQL 操作
        for op in mysql_ops:
            all_candidates.extend(self._fetch_mysql_resources(op.params))
            
        return all_candidates

    def execute_enrichment(self, strategy, context_nodes: List[CandidateNode] = None, user_query: str = "", age_months: int = None) -> List[CandidateNode]:
        """
        第二階段：知識增強 (Enrichment Stage)
        [CHANGE] 臨床常模現在作為「Baseline 常駐」，不再依賴 Agent 勾選。

        Args:
            strategy: 檢索策略
            context_nodes: 第一階段篩選後的候選節點
            user_query: 使用者原始查詢（供臨床對接使用）
            age_months: 兒童月齡（供常模/里程碑查詢）
        """
        enriched_results = []

        # 1. 執行基礎臨床對接 (Clinical Enrichment) - 作為每一輪的常駐背景
        clinical_params = {"query": user_query, "age_months": age_months}
        # 只有在有 Clinical Service 的情況下執行
        if self.clinical_service:
            enriched_results.extend(self._fetch_clinical_norms(clinical_params, context_nodes=context_nodes))

        # 2. 執行 GPT 或其它顯式增強
        gpt_ops = [op for op in strategy.operations if op.op_type == SearchOperationType.GPT_FETCH]
        for op in gpt_ops:
            enriched_results.extend(self._fetch_gpt_knowledge(op.params))
            
        return enriched_results

    def _fetch_subdomain(self, session, params: Dict[str, Any], doc_ids: List[str]) -> List[CandidateNode]:
        subdomain = params.get("subdomain", "").strip()
        print(f"[Engine] Fetching Subdomain: '{subdomain}' for docs: {doc_ids}")
        
        # Map logical sections to Neo4j relationship types
        sections = params.get("sections", [])
        if not isinstance(sections, list):
            sections = [sections] if sections else []
            
        rel_map = {
            "assessment": "HAS_ASSESSMENT_TOOLS|HAS_SCORES|HAS_FORMAL_ASSESSMENTS",
            "observation": "HAS_OBSERVATIONS",
            "training": "HAS_TRAINING_PLAN",
            "suggestion": "HAS_RECOMMENDATIONS"
        }
        
        target_rels = []
        for sec in sections:
            if sec in rel_map:
                target_rels.append(rel_map[sec])
        
        if target_rels:
            rel_type = "|".join(target_rels)
        else:
            # Fallback to all if empty
            rel_type = "HAS_ASSESSMENT_TOOLS|HAS_OBSERVATIONS|HAS_RECOMMENDATIONS|HAS_TRAINING_PLAN|HAS_SCORES|HAS_FORMAL_ASSESSMENTS"
        
        # Use IN $doc_ids to fetch from multiple reports
        cypher = f"""
        MATCH (r:Report)-[:HAS_DOMAIN]->(d:Domain)-[:HAS_SUBDOMAIN]->(sd:Subdomain)
        WHERE r.id IN $doc_ids AND trim(sd.name) = trim($subdomain)
        MATCH (sd)-[:{rel_type}]->(h:CategoryHub)
        MATCH (h)-[:USED_TOOL|OBSERVED|RECOMMENDED|TRAINED_BY|HAS_VALUE]->(item)
        OPTIONAL MATCH (item)-[:HAS_SUB_ITEM]->(sub:SubItem)
        MATCH (r)-[:HAS_META]->(m:Meta)
        RETURN labels(item)[0] as label, item.text as text, item.raw_text as raw_text, item.id as id, 
               h.name as category, collect(sub.text) as sub_items, m.report_complete_date as date, r.id as rid
        """
        result = session.run(cypher, doc_ids=doc_ids, subdomain=subdomain)
        candidates = []
        for record in result:
            text = record["text"]
            if record["sub_items"]:
                text += "\n  - " + "\n  - ".join(record["sub_items"])
                
            candidates.append(CandidateNode(
                node_id=record["id"],
                label=record["label"],
                text=text,
                properties={
                    "raw_text": record["raw_text"], 
                    "category": record["category"],
                    "subdomain": subdomain,
                    "report_date": record["date"],
                    "report_id": record["rid"]
                }
            ))
            
        if not candidates:
            print(f"[Engine][Warning] No items found for subdomain '{subdomain}'. Checking available subdomains...")
            check_cypher = "MATCH (r:Report {id: $doc_id})-[:HAS_DOMAIN]->(d)-[:HAS_SUBDOMAIN]->(sd) RETURN sd.name as name"
            # 針對第一個報告進行診斷
            available = [r["name"] for r in session.run(check_cypher, doc_id=doc_ids[0])]
            print(f"[Engine][Debug] Available subdomains in graph: {available}")
            
        return candidates

    def _fetch_summary(self, session, params: Dict[str, Any], doc_ids: List[str]) -> List[CandidateNode]:
        print(f"[Engine] Fetching Summary for docs: {doc_ids}")
        cypher = """
        MATCH (r:Report)-[:HAS_SUMMARY]->(s:Summary)
        WHERE r.id IN $doc_ids
        MATCH (s)-[:HAS_CATEGORY]->(h:CategoryHub)-[:HAS_CONTENT|DIAGNOSED_AS|HAS_RESULT|HAS_SUGGESTION]->(item)
        MATCH (r)-[:HAS_META]->(m:Meta)
        RETURN labels(item)[0] as label, item.text as text, item.raw_text as raw_text, item.id as id, 
               h.name as category, [] as sub_items, m.report_complete_date as date, r.id as rid
        """
        result = session.run(cypher, doc_ids=doc_ids)
        candidates = []
        for record in result:
            label = record["label"] if record["label"] else "Summary"
            category = record["category"] if record["category"] else "綜合建議"
            candidates.append(CandidateNode(
                node_id=record["id"],
                label=label,
                text=f"[{category}] {record['text']}",
                properties={
                    "category": category,
                    "subdomain": "整體概況",
                    "section_type": label,
                    "section_name": category,
                    "report_date": record["date"],
                    "report_id": record["rid"]
                }
            ))
        return candidates

    def _fetch_meta(self, session, params: Dict[str, Any], doc_ids: List[str]) -> List[CandidateNode]:
        print(f"[Engine] Fetching Meta for docs: {doc_ids}")
        cypher = """
        MATCH (r:Report)-[:HAS_META]->(m:Meta)
        WHERE r.id IN $doc_ids
        RETURN m, r.id as rid
        """
        result = session.run(cypher, doc_ids=doc_ids)
        candidates = []
        for record in result:
            m = record["m"]
            rid = record["rid"]
            meta_text = (
                f"個案姓名: {m.get('patient_name')}\n"
                f"性別: {m.get('gender')}\n"
                f"年齡: {m.get('age')}\n"
                f"就診日期: {m.get('doctor_visit_date')}\n"
                f"報告完成日期: {m.get('report_complete_date')}"
            )
            props = dict(m)
            props.update({
                "subdomain": "基本資料",
                "section_type": "Meta",
                "section_name": "個案資訊",
                "report_date": m.get('report_complete_date'),
                "report_id": rid
            })
            candidates.append(CandidateNode(
                node_id=f"{rid}_meta",
                label="Meta",
                text=meta_text,
                properties=props
            ))
        return candidates

    def _fetch_mysql_resources(self, params: Dict[str, Any]) -> List[CandidateNode]:
        if not self.mysql_client:
            return []

        query = params.get("query", "")
        # LOCAL_RESOURCE_SEARCH 傳 keywords（已抽取），Planning Agent 不傳 keywords 只傳 query。
        # 修正：以前把整句 query 當 keywords 傳給 SQL LIKE，導致 unit_name/category 永遠不命中。
        # 改成：Planning Agent 路徑時主動從 query 抽取早療相關詞，沒抽到則 None（純地區查詢）。
        keywords = params.get("keywords")
        if keywords is None and query:
            for kw in ["物理治療", "語言治療", "職能治療", "心理治療",
                       "感覺統合", "感統", "早期療育", "早療", "療育",
                       "復健", "評估", "兒童發展"]:
                if kw in query:
                    keywords = kw
                    break
            # 沒抽到任何關鍵字 → 維持 None，SQL 會用 "%%" pattern 純地區查詢

        # 優先使用 DST 偵測到的 region（更準確），fallback 到關鍵字比對
        region = params.get("region", "")
        if not region:
            regions = ["台北", "新北", "桃園", "台中", "台南", "高雄", "基隆", "新竹", "苗栗", "彰化", "南投", "雲林", "嘉義", "屏東", "宜蘭", "花蓮", "台東", "澎湖", "金門", "馬祖", "連江"]
            for r in regions:
                if r in query:
                    region = r
                    break

        candidates = []

        # 1. 查機構/據點資源（sfaa_units + community_intervention_units）
        candidates.extend(self.mysql_client.fetch_resources_by_region(region, keywords=keywords))

        # 2. 查補助方案（subsidy_program）— 有地區時自動查
        if region:
            candidates.extend(self.mysql_client.fetch_subsidy_by_region(region))

        return candidates

    def _fetch_clinical_norms(self, params: Dict[str, Any], context_nodes: List[CandidateNode] = None) -> List[CandidateNode]:
        """安全呼叫 ClinicalBridgeService 獲取常模與里程碑"""
        if not self.clinical_service:
            print("[ExecutionEngine] ClinicalBridgeService is missing, skipping CLINICAL_FETCH.")
            return []
            
        query = params.get("query", "")
        age_months = params.get("age_months")
        
        print(f"[Engine] Fetching Clinical Norms for query: '{query}', age_months: {age_months}")
        
        if not self.clinical_service:
            return []
            
        try:
            # 1. 提取個案報告中的關鍵文本 (作為多跳關聯的起點)
            obs_texts = []
            train_texts = []
            rec_texts = []
            
            if context_nodes:
                for c in context_nodes:
                    # 根據 neo4j_importer 的標籤進行過濾
                    if c.label == "Observation":
                        obs_texts.append(c.text)
                    elif c.label == "TrainingDirection":
                        train_texts.append(c.text)
                    elif c.label == "Recommendation" or c.label == "GeneralRecommendation":
                        rec_texts.append(c.text)
            
            # 2. 調用臨床圖譜服務進行「語義對接」
            # 如果有上下文，優先使用 get_llm_payload (這會做向量比對找出最相關的常模)
            if obs_texts or train_texts or rec_texts:
                print(f"[ExecutionEngine] 執行臨床增強：對接 {len(obs_texts)} 筆觀察, {len(train_texts)} 筆訓練...")
                payload = self.clinical_service.get_llm_payload(
                    observations=" ".join(obs_texts),
                    training_goals=" ".join(train_texts),
                    recommendations=" ".join(rec_texts),
                    with_milestones=True
                )
                
                # 將豐富的 Payload 封裝成一個特殊的 CandidateNode
                return [CandidateNode(
                    node_id=f"clinical_bridge_enriched_{hash(query)}",
                    label="ClinicalNorm",
                    text=json.dumps(payload, ensure_ascii=False) if isinstance(payload, dict) else str(payload),
                    properties={"category": "發展關聯分析", "subdomain": "臨床常模"},
                    score=0.95, # 給予高分確保排在前面
                    metadata={"source": "ClinicalBridgeService"}
                )]
            
            # 3. 如果沒有上下文 (例如這是第一輪提問)，則回退到單純按年齡抓取里程碑
            print(f"[ExecutionEngine] 無個案上下文，依年齡 {age_months} 抓取里程碑...")
            milestones = self.clinical_service.get_developmental_map(age_months=age_months)
            if milestones and not milestones.get("error"):
                return [CandidateNode(
                    node_id=f"clinical_milestone_{age_months}",
                    label="ClinicalNorm",
                    text=json.dumps(milestones, ensure_ascii=False),
                    properties={"category": f"{age_months}個月發展里程碑", "subdomain": "臨床常模"},
                    score=0.85,
                    metadata={"source": "ClinicalBridgeService"}
                )]
                
        except Exception as e:
            print(f"[ExecutionEngine] Clinical Enrichment Error: {e}")
            
        return []

    def _fetch_gpt_knowledge(self, params: Dict[str, Any]) -> List[CandidateNode]:
        """
        呼叫外部 OpenAI API 取得通用早療/育兒知識，作為 RAG 語料注入 LLM context。
        路由策略：
          - 在地資源類 query（地區 + 機構/補助關鍵字）→ Responses API + web_search_preview（即時上網）
          - 其他通用知識 query → chat.completions（快、便宜）
        env 控制：USE_WEB_SEARCH=0 強制關閉上網
        """
        import os
        query = params.get("query", "")
        chat_history = params.get("chat_history", [])
        api_key = os.environ.get("OPENAI_API_KEY", "")

        if not api_key:
            print(f"[Engine] GPT Fetch: 未設定 OPENAI_API_KEY，略過 external_gpt 查詢")
            return []

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
        except Exception as e:
            print(f"[Engine] GPT Fetch: openai client 初始化失敗：{e}")
            return []

        enable_ws = os.environ.get("USE_WEB_SEARCH", "1") == "1"
        force_ws = bool(params.get("force_web_search"))
        use_ws = enable_ws and (force_ws or self._should_use_web_search(query))
        if force_ws:
            print(f"[Engine] GPT Fetch: LOCAL_RESOURCE_SEARCH 強制 web_search")

        if use_ws:
            try:
                return self._fetch_with_web_search(client, query, chat_history)
            except Exception as e:
                print(f"[Engine] GPT Web Search 失敗：{e}，fallback chat.completions")

        return self._fetch_chat_completion(client, query, chat_history)

    @staticmethod
    def _should_use_web_search(query: str) -> bool:
        """判斷是否啟動 web_search（在地資源類 query）"""
        if not query:
            return False
        regions = ("台北", "新北", "桃園", "台中", "台南", "高雄", "基隆", "新竹",
                   "苗栗", "彰化", "南投", "雲林", "嘉義", "屏東", "宜蘭", "花蓮",
                   "台東", "澎湖", "金門", "馬祖", "連江",
                   "臺北", "臺中", "臺南", "臺東")
        resource_kws = ("機構", "診所", "治療所", "醫院", "中心", "復健", "資源",
                        "去哪", "哪裡", "推薦", "據點", "申請", "補助", "窗口")
        has_region = any(r in query for r in regions)
        has_resource = any(kw in query for kw in resource_kws)
        return has_region and has_resource

    def _fetch_with_web_search(self, client, query: str, chat_history: List[Dict]) -> List[CandidateNode]:
        """用 Responses API + web_search_preview 真實上網查詢"""
        system_prompt = (
            "你是一位台灣早期療育（早療）與兒童發展的專業顧問，會即時查詢網路上的官方資訊。"
            "回答台灣家長/治療師關於在地機構、補助方案、申請流程的問題。"
            "重要規則："
            "\n1. 必須優先引用官方來源（衛福部、各縣市社會局、社家署）。"
            "\n2. 列出機構名單時要明確註明來源網站。"
            "\n3. 補助金額/條件請註明「以最新公告為準」。"
            "\n4. 全程使用繁體中文與台灣用語。"
            "\n5. 回答限 400 字以內，條列式為主。"
        )
        recent_history = chat_history[-4:] if chat_history else []
        input_msgs = [{"role": "system", "content": system_prompt}]
        for msg in recent_history:
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                input_msgs.append({"role": msg["role"], "content": msg["content"]})
        input_msgs.append({"role": "user", "content": query})

        print(f"[Engine] GPT Web Search 啟動（query={query[:30]}...）")
        response = client.responses.create(
            model="gpt-4o-mini",
            input=input_msgs,
            tools=[{"type": "web_search_preview"}],
        )

        gpt_text = getattr(response, "output_text", None)
        if not gpt_text:
            try:
                out_chunks = []
                for o in (response.output or []):
                    for c in getattr(o, "content", []) or []:
                        t = getattr(c, "text", None)
                        if t:
                            out_chunks.append(t)
                gpt_text = "\n".join(out_chunks).strip()
            except Exception:
                gpt_text = ""

        if not gpt_text:
            raise RuntimeError("Empty web_search response")

        print(f"[Engine] GPT Web Search 成功：{len(gpt_text)} 字")
        gpt_text = self._maybe_summarize(gpt_text, query)
        return [CandidateNode(
            node_id=f"external_gpt_ws_{hash(query) & 0xFFFFFF}",
            label="ExternalGPT",
            text=gpt_text,
            properties={"subdomain": "外部知識", "category": "在地資源（即時）"},
            score=0.85,
            metadata={"source": "OpenAI Responses + web_search", "query": query}
        )]

    def _fetch_chat_completion(self, client, query: str, chat_history: List[Dict]) -> List[CandidateNode]:
        """通用知識 query 走 chat.completions（不上網，便宜快速）"""
        try:
            system_prompt = (
                "你是一位早期療育（早療）與兒童發展的專業顧問。"
                "請針對照顧者或治療師的問題，提供準確、簡潔的通用專業知識。"
                "包含：發展里程碑、DSM-5 診斷規範、補助申請流程、衛教常識、學校合作策略等。"
                "回答限 300 字以內，不需要針對特定個案，提供通用的專業建議即可。"
                "\n\n【重要】"
                "\n1. 不要列出具體機構名稱、地址、電話（這類資訊請改建議查詢來源）。"
                "\n2. 涉及在地資源時，建議使用者「至衛福部社家署網站或各縣市社會局查詢」。"
                "\n3. 補助條件僅給通用方向，具體金額/年度請說「以該縣市最新公告為準」。"
            )

            recent_history = chat_history[-6:] if chat_history else []
            messages = [{"role": "system", "content": system_prompt}]
            for msg in recent_history:
                if msg.get("role") in ("user", "assistant") and msg.get("content"):
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": query})

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=400,
                temperature=0.3,
            )
            gpt_text = response.choices[0].message.content.strip()
            print(f"[Engine] GPT Fetch 成功（通用知識）：{len(gpt_text)} 字")
            gpt_text = self._maybe_summarize(gpt_text, query)

            return [CandidateNode(
                node_id=f"external_gpt_{hash(query) & 0xFFFFFF}",
                label="ExternalGPT",
                text=gpt_text,
                properties={"subdomain": "外部知識", "category": "通用早療知識"},
                score=0.75,
                metadata={"source": "OpenAI GPT", "query": query}
            )]

        except Exception as e:
            print(f"[Engine] GPT Fetch 失敗：{e}，略過 external_gpt")
            return []

    def _maybe_summarize(self, text: str, query: str) -> str:
        """
        若 GPT 回的內容太長，用本地 Qwen 摘要，避免下游 LLM context 被截斷。

        env 控制：
          SUMMARIZE_GPT=0           關閉摘要（預設開）
          SUMMARIZE_THRESHOLD=500   超過 N 字才摘要（預設 500）
          SUMMARIZE_MAX_TOKENS=300  摘要最大 token（預設 300）
        """
        import os
        if not text:
            return text

        enable = os.environ.get("SUMMARIZE_GPT", "1") == "1"
        threshold = int(os.environ.get("SUMMARIZE_THRESHOLD", "500"))
        if not enable or len(text) <= threshold:
            return text

        try:
            from openai import OpenAI
            try:
                from config import LLM_CONFIG
            except ImportError:
                # execution_engine 在 retrieval_module_v2 子模組，sys.path 可能未含 root
                import sys, os as _os
                _root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
                if _root not in sys.path:
                    sys.path.insert(0, _root)
                from config import LLM_CONFIG

            client = OpenAI(
                base_url=LLM_CONFIG['base_url'],
                api_key=LLM_CONFIG['api_key'],
                timeout=30,
            )
            max_tok = int(os.environ.get("SUMMARIZE_MAX_TOKENS", "300"))

            sys_msg = (
                "你是文件摘要專家，幫我把外部檢索結果壓縮成精簡重點。"
                "規則："
                "\n1. 全程使用繁體中文與台灣用語。"
                "\n2. 保留所有具體名稱、地址、電話、金額、條件等關鍵細節。"
                "\n3. 用條列式輸出，最多 5 點，每點不超過 40 字。"
                "\n4. 刪除冗餘介紹文字、客套話。"
                "\n5. 若有引用來源，保留（網站名/出處），不要保留長 URL。"
                "\n6. 整體字數控制在 200 字內。"
            )
            user_msg = (
                f"使用者問題：{query}\n\n"
                f"原始檢索結果（請壓縮成 200 字內條列重點）：\n{text}"
            )
            response = client.chat.completions.create(
                model=LLM_CONFIG['model'],
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=max_tok,
                temperature=0.1,
            )
            summary = (response.choices[0].message.content or "").strip()
            if summary and len(summary) < len(text):
                print(f"[Engine] 本地摘要：{len(text)} → {len(summary)} 字")
                return summary
            return text

        except Exception as e:
            print(f"[Engine] 本地摘要失敗：{e}，使用原文")
            return text
