import requests
import json
import re
import os
from neo4j import GraphDatabase

URI = 'bolt://192.168.150.136:7687'
AUTH = ('neo4j', 'password')

class ClinicalCorrespondenceEngine:
    def __init__(self, encoder=None):
        self.driver = GraphDatabase.driver(URI, auth=AUTH)
        self.encoder = encoder # [NEW] 允許從外部注入 app_v7 的 TextEncoder
        
        # 取得目前檔案路徑，確保能找到 JSON (現在位於 ../json)
        base_path = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(base_path), 'json')
        
        # 加載本地臨床字典 (用於自動化加權)
        dict_path = os.path.join(data_dir, 'clinical_dictionary.json')
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                self.dictionary = json.load(f)
        except Exception:
            self.dictionary = []
            
        # 加載自動識別的超級節點 (Sink Nodes)
        sink_path = os.path.join(data_dir, 'sink_nodes.json')
        try:
            with open(sink_path, 'r', encoding='utf-8') as f:
                self.sink_nodes = set(json.load(f))
        except Exception:
             self.sink_nodes = {"持續注意", "性遊戲", "動作計畫", "執行功能", "規則理解"}

    def close(self):
        self.driver.close()

    def get_embedding(self, text):
        """
        獲取向量：優先使用注入的 encoder，否則回退到本地 Qwen3-Embedding 伺服器
        """
        if not text:
            return None
            
        # 優先使用 app_v7 的 encoder
        if self.encoder:
            try:
                # 假設 encoder 有 encode 方法且回傳 np.ndarray，轉為 list
                emb = self.encoder.encode(text)
                if hasattr(emb, "tolist"):
                    return emb.tolist()
                return list(emb)
            except Exception as e:
                print(f"[ClinicalEngine] External Encoder error: {e}")

        # 回退到本地 8001 伺服器
        url = "http://localhost:8001/v1/embeddings"
        data = {
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "input": text
        }
        
        try:
            response = requests.post(url, json=data, timeout=5)
            response.raise_for_status()
            res_data = response.json()
            if "data" in res_data and len(res_data["data"]) > 0:
                return res_data["data"][0]["embedding"]
            return None
        except Exception as e:
            print(f"[ClinicalEngine] Local Embedding error: {e}")
            return None

    def _sanitize_lucene(self, text):
        """
        轉義 Lucene 特殊字元，避免語法錯誤
        """
        # 移除或轉義： + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /
        special_chars = r'[+\-&&||!(){}\[\]\^"~*?:\\]'
        return re.sub(special_chars, ' ', text)

    def _extract_boost_terms(self, text):
        """
        全量字典比對：掃描文字中出現的所有臨床專用語，並給予搜尋權重
        """
        boosts = []
        for term in self.dictionary:
            if term in text:
                boosts.append(f"{term}^10")
        # 限制加權詞的數量，避免 Lucene 查詢過長
        return " ".join(boosts[:20]) if boosts else ""

    def find_multi_party_correspondence(self, query_text, limit=3, context_domain=None):
        """
        混合搜尋 (Hybrid Search)：結合 Lucene 關鍵字與 Vector 語意向量
        """
        with self.driver.session() as session:
            # --- PART 1: 傳統 Lucene 搜尋 ---
            clean_text = self._sanitize_lucene(query_text)
            boost_text = self._extract_boost_terms(clean_text)
            full_search_text = f"{clean_text} {boost_text}".strip()

            lucene_query = """
            CALL db.index.fulltext.queryNodes("abilityIndex", $text) YIELD node AS n, score AS s
            RETURN elementId(n) AS elementId, n.name AS name, s AS score, "Ability" AS label, "keyword" AS type
            UNION ALL
            CALL db.index.fulltext.queryNodes("obsIndex", $text) YIELD node AS n, score AS s
            RETURN elementId(n) AS elementId, n.content AS name, s AS score, "ObservationIndicator" AS label, "keyword" AS type
            UNION ALL
            CALL db.index.fulltext.queryNodes("strategyIndex", $text) YIELD node AS n, score AS s
            RETURN elementId(n) AS elementId, n.name AS name, s AS score, "TrainingStrategy" AS label, "keyword" AS type
            ORDER BY score DESC LIMIT $limit
            """
            lucene_res = session.run(lucene_query, text=full_search_text, limit=limit)
            
            # --- PART 2: 向量語意搜尋 ---
            vector_res_data = []
            embedding = self.get_embedding(query_text)
            if embedding:
                vector_query = """
                CALL db.index.vector.queryNodes("abilityVectorIndex", $limit, $emb) YIELD node AS n, score AS s
                RETURN elementId(n) AS elementId, n.name AS name, s AS score, "Ability" AS label, "vector" AS type
                UNION ALL
                CALL db.index.vector.queryNodes("obsVectorIndex", $limit, $emb) YIELD node AS n, score AS s
                RETURN elementId(n) AS elementId, n.content AS name, s AS score, "ObservationIndicator" AS label, "vector" AS type
                ORDER BY score DESC LIMIT $limit
                """
                vector_res = session.run(vector_query, emb=embedding, limit=limit)
                vector_res_data = [dict(r) for r in vector_res]

            # --- PART 3: 混合與評分綜效 ---
            all_entries = [dict(r) for r in lucene_res] + vector_res_data
            
            # 使用簡單的去重與權重合併
            seen = {}
            for e in all_entries:
                eid = e['elementId']
                if eid not in seen:
                    seen[eid] = e
                    # 向量分數通常較低 (0.5-0.9)，我們給與加權
                    if e['type'] == 'vector': e['score'] *= 10
                else:
                    # 如果同時命中，大幅加分
                    seen[eid]['score'] *= 2.0
                    seen[eid]['type'] = 'hybrid'

            all_entries = sorted(seen.values(), key=lambda x: x['score'], reverse=True)
            if not all_entries: return []
            
            # --- PART 4: 嚴格篩選 (Strict Filtering) ---
            # 只保留最高分的一定比例 (例如前 50%)，且絕對分數不能太低
            max_score = all_entries[0]['score']
            # 目前 Vector 分數約在 0.7-0.9 之間，Lucene 則可能很大。
            # 我們保留前 3 個「真正相關」的點
            entry_points = [r for r in all_entries if r['score'] >= max_score * 0.7][:3]
            
            if not entry_points:
                return {"message": "No matches found in clinical index.", "results": []}

            # Step 2: For each entry point, expand to find related "Parties"
            # We trace through the Ability Hub to find siblings and recommendations
            correspondences = []
            for entry in entry_points:
                # Find connected clinical nodes within 2 hops
                # 嚴格路徑：保留 Hub (Ability) 本身作為對接錨點
                expand_query = """
                MATCH (start) WHERE elementId(start) = $eid
                OPTIONAL MATCH (start)-[:INDICATES_ABILITY|TARGETS_ABILITY|HAS_ABILITY]-(a:Ability)
                WITH start, a
                // 找出相關的能力 (包含 a 自身)、策略、活動、里程碑
                MATCH (related)
                WHERE (related = a OR (a)-[:HAS_MILESTONE|TARGETS_ABILITY|HAS_ABILITY|IMPLEMENTS_STRATEGY]-(related))
                  AND labels(related)[0] IN ['Ability', 'TrainingStrategy', 'TrainingActivity', 'Milestone']
                  AND NOT related.name IN $sinks
                // 同時回傳領域標籤以供分級
                OPTIONAL MATCH (related)-[:HAS_ABILITY|TARGETS_ABILITY|BELONGS_TO_SUBDOMAIN*1..2]-(s:Subdomain)
                RETURN DISTINCT labels(related)[0] AS label, 
                                related.name AS name, 
                                related.expected_behavior AS behavior,
                                related.description AS desc,
                                collect(DISTINCT s.name) AS subdomains
                LIMIT 15
                """
                expanded = session.run(expand_query, eid=entry['elementId'], sinks=list(self.sink_nodes))
                
                # 收集結果，包含領域屬性
                party_map = {}
                for row in expanded:
                    lbl = row['label']
                    if lbl not in party_map: party_map[lbl] = []
                    item_name = row['name'] or row['behavior'] or row['desc']
                    if item_name not in party_map[lbl]:
                        party_map[lbl].append({
                            "name": item_name,
                            "subdomains": row['subdomains']
                        })
                
                correspondences.append({
                    "enrty": entry,
                    "related_parties": party_map
                })
                
            return correspondences

def run_sample_test():
    engine = ClinicalCorrespondenceEngine()
    
    # Test text from User
    test_text = "上樓梯時可不扶扶手、一腳一階行走，會將雙手平舉於身體兩側以協助維持平衡。"
    print(f"\n--- Testing Search for: '{test_text[:30]}...' ---")
    
    results = engine.find_multi_party_correspondence(test_text)
    
    for idx, res in enumerate(results):
        entry = res['enrty']
        print(f"\n[{idx+1}] Entry Point: {entry['name']} ({entry['label']}) [Score: {entry['score']:.2f}]")
        print("    Related Items Found:")
        for label, items in res['related_parties'].items():
            print(f"      - {label}: {', '.join(items[:5])}...")

    engine.close()

if __name__ == "__main__":
    run_sample_test()
