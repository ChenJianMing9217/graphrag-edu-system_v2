#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graph_client.py
負責 Neo4j 查詢（用現有 Neo4j driver）
"""

from typing import Dict, List, Optional, Any
from neo4j import GraphDatabase
import sys
import os

# 添加父目錄到路徑以導入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_neo4j_uri, get_neo4j_auth

# Domain 映射（避免循環導入，直接定義在這裡）
SUBDOMAIN_TO_DOMAIN = {
    "粗大動作": "知覺動作功能",
    "精細動作": "知覺動作功能",
    "感覺統合": "知覺動作功能",
    "口腔動作": "吞嚥/口腔功能",
    "吞嚥功能": "吞嚥/口腔功能",
    "吞嚥反射": "吞嚥/口腔功能",
    "口語理解": "口語溝通功能",
    "口語表達": "口語溝通功能",
    "說話": "口語溝通功能",
    "認知功能": "認知功能",
    "情緒行為與社會適應功能": "社會情緒功能",
}

DIALOGUE_STATE_DOMAINS = list(SUBDOMAIN_TO_DOMAIN.keys())

def _is_subdomain_name(name: str) -> bool:
    """判斷是否為 subdomain 名稱"""
    return name in DIALOGUE_STATE_DOMAINS

def _map_subdomain_to_domain(subdomain_name: str) -> str:
    """將 subdomain 映射到 domain"""
    return SUBDOMAIN_TO_DOMAIN.get(subdomain_name, subdomain_name)

def _find_matching_domain(query_domain: str, available_domains: list[str]) -> Optional[str]:
    """在可用 domains 中尋找匹配的 domain"""
    # 1. 如果 query_domain 是 subdomain，映射到 domain
    if _is_subdomain_name(query_domain):
        mapped_domain = _map_subdomain_to_domain(query_domain)
        if mapped_domain in available_domains:
            return mapped_domain
    
    # 2. 直接匹配
    if query_domain in available_domains:
        return query_domain
    
    # 3. 模糊匹配
    for available_domain in available_domains:
        if query_domain in available_domain or available_domain in query_domain:
            return available_domain
    
    return None


class GraphClient:
    """Neo4j 圖查詢客戶端"""
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j"
    ):
        """
        初始化 Neo4j 客戶端
        
        Args:
            uri: Neo4j URI（預設從 config 讀取）
            user: Neo4j 使用者（預設從 config 讀取）
            password: Neo4j 密碼（預設從 config 讀取）
            database: Neo4j 資料庫名稱
        """
        if uri is None:
            uri = get_neo4j_uri()
        if user is None or password is None:
            user, password = get_neo4j_auth()
        
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """關閉連接"""
        if self.driver:
            self.driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def list_domains(self, doc_id: str) -> List[Dict[str, str]]:
        """
        列出某報告的所有 domains
        
        Args:
            doc_id: 文件 ID
        
        Returns:
            List[Dict]: [{"domain_id": "...", "name": "..."}, ...]
        """
        cypher = """
        MATCH (r:Report {doc_id: $doc_id})-[:COVERS_DOMAIN]->(d:Domain)
        RETURN d.domain_id AS domain_id, d.name AS name
        ORDER BY d.name
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, doc_id=doc_id)
            return [{"domain_id": record["domain_id"], "name": record["name"]} 
                    for record in result]
    
    def list_subdomains(
        self, 
        doc_id: str, 
        domain_name: str,
        allow_map_to_parent_domain: bool = True
    ) -> List[Dict[str, str]]:
        """
        列出某 domain 下的所有 subdomains
        
        Args:
            doc_id: 文件 ID
            domain_name: Domain 名稱（可能是 subdomain 名稱）
            allow_map_to_parent_domain: 如果 domain_name 是 subdomain，是否允許映射到母 domain 再展開（預設 True，向後兼容）
        
        Returns:
            List[Dict]: [{"subdomain_id": "...", "name": "..."}, ...]
        """
        # 如果 domain_name 是 subdomain 且不允許映射，直接查詢該 subdomain
        if _is_subdomain_name(domain_name) and not allow_map_to_parent_domain:
            cypher = """
            MATCH (sd:Subdomain {name: $subdomain_name})
            WHERE sd.doc_id = $doc_id
            RETURN sd.subdomain_id AS subdomain_id, sd.name AS name
            LIMIT 1
            """
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, doc_id=doc_id, subdomain_name=domain_name)
                record = result.single()
                if record:
                    return [{"subdomain_id": record["subdomain_id"], "name": record["name"]}]
                return []
        
        # 先取得所有可用的 domains
        available_domains = [d["name"] for d in self.list_domains(doc_id)]
        
        # 如果 domain_name 是 subdomain，嘗試映射到 domain（僅當 allow_map_to_parent_domain=True）
        actual_domain_name = domain_name
        if _is_subdomain_name(domain_name) and allow_map_to_parent_domain:
            mapped_domain = _map_subdomain_to_domain(domain_name)
            if mapped_domain in available_domains:
                actual_domain_name = mapped_domain
            else:
                # 如果映射失敗，嘗試直接匹配或模糊匹配
                matched = _find_matching_domain(domain_name, available_domains)
                if matched:
                    actual_domain_name = matched
                else:
                    # 如果還是找不到，嘗試直接查詢（可能 domain_name 就是實際的 domain）
                    pass
        
        cypher = """
        MATCH (r:Report {doc_id: $doc_id})-[:COVERS_DOMAIN]->(d:Domain {name: $domain_name})
        MATCH (d)-[:HAS_SUBDOMAIN]->(sd:Subdomain)
        RETURN sd.subdomain_id AS subdomain_id, sd.name AS name
        ORDER BY sd.name
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, doc_id=doc_id, domain_name=actual_domain_name)
            subdomains = [{"subdomain_id": record["subdomain_id"], "name": record["name"]} 
                         for record in result]
            
            # 如果查不到結果，且 domain_name 是 subdomain，嘗試直接查詢該 subdomain
            if not subdomains and _is_subdomain_name(domain_name):
                # 直接查詢 subdomain（可能 subdomain 直接連到 sections，沒有經過 domain）
                cypher2 = """
                MATCH (sd:Subdomain {name: $subdomain_name})
                WHERE sd.doc_id = $doc_id
                RETURN sd.subdomain_id AS subdomain_id, sd.name AS name
                LIMIT 1
                """
                result2 = session.run(cypher2, doc_id=doc_id, subdomain_name=domain_name)
                record = result2.single()
                if record:
                    subdomains = [{"subdomain_id": record["subdomain_id"], "name": record["name"]}]
            
            return subdomains
    
    def fetch_sections_by_subdomain(
        self, 
        doc_id: str, 
        subdomain_id: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        取得某 subdomain 下的四種 section 節點
        
        Args:
            doc_id: 文件 ID
            subdomain_id: Subdomain ID
        
        Returns:
            Dict: {
                "assessment": [Assessment nodes],
                "observation": [Observation nodes],
                "training": [Training nodes],
                "suggestion": [Suggestion nodes]
            }
        """
        cypher = """
        MATCH (sd:Subdomain {subdomain_id: $subdomain_id})
        OPTIONAL MATCH (sd)-[:HAS_ASSESSMENT]->(a:Assessment)
        OPTIONAL MATCH (sd)-[:HAS_OBSERVATION]->(o:Observation)
        OPTIONAL MATCH (sd)-[:HAS_TRAINING]->(t:Training)
        OPTIONAL MATCH (sd)-[:HAS_SUGGESTION]->(s:Suggestion)
        RETURN 
            collect(DISTINCT a) AS assessments,
            collect(DISTINCT o) AS observations,
            collect(DISTINCT t) AS trainings,
            collect(DISTINCT s) AS suggestions
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, subdomain_id=subdomain_id)
            record = result.single()
            
            def node_to_dict(node):
                if node is None:
                    return None
                return dict(node)
            
            return {
                "assessment": [node_to_dict(n) for n in record["assessments"] if n is not None],
                "observation": [node_to_dict(n) for n in record["observations"] if n is not None],
                "training": [node_to_dict(n) for n in record["trainings"] if n is not None],
                "suggestion": [node_to_dict(n) for n in record["suggestions"] if n is not None],
            }
    
    def fetch_items(
        self,
        section_id: str,
        section_type: str,  # "assessment", "observation", "training", "suggestion"
        limit: int = 10,
        include_subitems: bool = False
    ) -> List[Dict[str, Any]]:
        """
        取得某 section 的 items（只取 level=1 的主項）
        
        Args:
            section_id: Section 節點的 id
            section_type: Section 類型
            limit: 限制數量
            include_subitems: 是否包含子項（透過 HAS_SUBITEM 展開 1 層）
        
        Returns:
            List[Dict]: Item 節點列表
        """
        # 決定 item label
        item_label_map = {
            "assessment": "AssessmentItem",
            "observation": "ObservationItem",
            "training": "TrainingItem",
            "suggestion": "SuggestionItem",
        }
        item_label = item_label_map.get(section_type)
        if not item_label:
            return []
        
        if include_subitems:
            # 包含子項（展開 1 層）
            cypher = f"""
            MATCH (sec {{id: $section_id}})-[:HAS_ITEM]->(it:{item_label})
            WHERE it.level = 1
            OPTIONAL MATCH (it)-[:HAS_SUBITEM]->(sub:{item_label})
            RETURN it, collect(DISTINCT sub) AS subitems
            ORDER BY it.seq
            LIMIT $limit
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, section_id=section_id, limit=limit)
                items = []
                for record in result:
                    item_dict = dict(record["it"])
                    subitems = [dict(sub) for sub in record["subitems"] if sub is not None]
                    item_dict["subitems"] = subitems
                    items.append(item_dict)
                return items
        else:
            # 只取主項
            cypher = f"""
            MATCH (sec {{id: $section_id}})-[:HAS_ITEM]->(it:{item_label})
            WHERE it.level = 1
            RETURN it
            ORDER BY it.seq
            LIMIT $limit
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, section_id=section_id, limit=limit)
                return [dict(record["it"]) for record in result]
    
    def expand_from_item(
        self,
        item_id: str,
        item_label: str,  # "AssessmentItem", "ObservationItem", etc.
        hops: int = 2,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        從某 item 展開（可選功能，用於圖遍歷）
        
        Args:
            item_id: Item 節點的 id
            item_label: Item label
            hops: 展開的跳數
            limit: 限制數量
        
        Returns:
            List[Dict]: 展開的節點列表
        """
        cypher = f"""
        MATCH (start:{item_label} {{id: $item_id}})
        MATCH path = (start)-[*1..{hops}]-(related)
        WHERE related.id IS NOT NULL
        RETURN DISTINCT related, length(path) AS distance
        ORDER BY distance, related.seq
        LIMIT $limit
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, item_id=item_id, limit=limit)
            return [dict(record["related"]) for record in result]
    
    def search_fallback_vector(
        self,
        doc_id: str,
        query_embedding: Optional[List[float]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        向量搜尋後備方案（若專案有向量索引才實作，目前先 stub）
        
        Args:
            doc_id: 文件 ID
            query_embedding: 查詢向量（可選）
            limit: 限制數量
        
        Returns:
            List[Dict]: 搜尋結果
        """
        # TODO: 實作向量搜尋（如果有向量索引）
        return []
    
    def get_report_overview_sections(
        self,
        doc_id: str,
        domain_name: Optional[str] = None,
        k_per_subdomain: int = 1
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        取得報告概覽所需的 sections（REPORT_OVERVIEW mode 使用）
        
        Args:
            doc_id: 文件 ID
            domain_name: Domain 名稱（可選，若指定則只查該 domain，可能是 subdomain 名稱）
            k_per_subdomain: 每個 subdomain 取幾個 section
        
        Returns:
            Dict: {
                "domain_name": {
                    "subdomain_name": {
                        "assessment": [...],
                        "observation": [...],
                        "training": [...],
                        "suggestion": [...]
                    }
                }
            }
        """
        result = {}
        
        if domain_name:
            # 只查指定 domain（可能是 subdomain 名稱）
            # 策略 1: 如果 domain_name 是 subdomain，直接查詢該 subdomain
            if _is_subdomain_name(domain_name):
                # 直接查詢 subdomain 的 sections
                cypher = """
                MATCH (sd:Subdomain {name: $subdomain_name})
                WHERE sd.doc_id = $doc_id
                OPTIONAL MATCH (sd)-[:HAS_ASSESSMENT]->(a:Assessment)
                OPTIONAL MATCH (sd)-[:HAS_OBSERVATION]->(o:Observation)
                OPTIONAL MATCH (sd)-[:HAS_TRAINING]->(t:Training)
                OPTIONAL MATCH (sd)-[:HAS_SUGGESTION]->(s:Suggestion)
                RETURN 
                    collect(DISTINCT a) AS assessments,
                    collect(DISTINCT o) AS observations,
                    collect(DISTINCT t) AS trainings,
                    collect(DISTINCT s) AS suggestions
                """
                with self.driver.session(database=self.database) as session:
                    result_query = session.run(cypher, doc_id=doc_id, subdomain_name=domain_name)
                    record = result_query.single()
                    if record:
                        def node_to_dict(node):
                            if node is None:
                                return None
                            return dict(node)
                        
                        sections_dict = {
                            "assessment": [node_to_dict(n) for n in record["assessments"] if n is not None],
                            "observation": [node_to_dict(n) for n in record["observations"] if n is not None],
                            "training": [node_to_dict(n) for n in record["trainings"] if n is not None],
                            "suggestion": [node_to_dict(n) for n in record["suggestions"] if n is not None],
                        }
                        
                        if any(sections_dict.values()):
                            limited_sections = {}
                            for sec_type, sec_list in sections_dict.items():
                                limited_sections[sec_type] = sec_list[:k_per_subdomain]
                            
                            result[domain_name] = {
                                domain_name: limited_sections
                            }
                            return result
            
            # 策略 2: 嘗試映射到實際的 domain
            available_domains = [d["name"] for d in self.list_domains(doc_id)]
            actual_domain_name = _find_matching_domain(domain_name, available_domains)
            
            if actual_domain_name:
                domains = [{"name": actual_domain_name}]
            else:
                # 如果找不到對應的 domain，嘗試直接查詢該名稱（可能是實際的 domain）
                domains = [{"name": domain_name}]
        else:
            # 查所有 domains
            domains = self.list_domains(doc_id)
        
        for domain in domains:
            domain_name = domain["name"]
            
            # 正常流程：通過 domain 查詢 subdomains
            subdomains = self.list_subdomains(doc_id, domain_name)
            
            # 如果查不到 subdomains，跳過
            if not subdomains:
                continue
            
            # 正常流程：處理找到的 subdomains
            domain_result = {}
            for subdomain in subdomains:
                subdomain_id = subdomain["subdomain_id"]
                subdomain_name = subdomain["name"]
                
                # 取得該 subdomain 的所有 sections
                sections = self.fetch_sections_by_subdomain(doc_id, subdomain_id)
                
                # 限制每個 section 類型的數量
                limited_sections = {}
                for sec_type, sec_list in sections.items():
                    limited_sections[sec_type] = sec_list[:k_per_subdomain]
                
                domain_result[subdomain_name] = limited_sections
            
            if domain_result:  # 只有當有結果時才添加
                result[domain_name] = domain_result
        
        return result
    
    def get_domain_sections(
        self,
        doc_id: str,
        domain_names: List[str],
        section_types: List[str],
        k_per_subdomain: int = 2,
        allow_parent_domain_expand: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        取得指定 domains 的 sections（DOMAIN mode 使用）
        
        Args:
            doc_id: 文件 ID
            domain_names: Domain 名稱列表（可能是 subdomain 名稱）
            section_types: 要查詢的 section 類型列表
            k_per_subdomain: 每個 subdomain 取幾個 section
            allow_parent_domain_expand: 是否允許 subdomain 映射到母 domain 再展開（預設 False）
        
        Returns:
            Dict: {
                "domain_name": {
                    "subdomain_name": {
                        "assessment": [...],
                        ...
                    }
                }
            }
        """
        result = {}
        
        for domain_name in domain_names:
            # 如果 domain_name 是 subdomain 且不允許展開，直接查詢該 subdomain
            if _is_subdomain_name(domain_name) and not allow_parent_domain_expand:
                # 直接查詢 subdomain 的 sections（不經過 domain）
                cypher = """
                MATCH (sd:Subdomain {name: $subdomain_name})
                WHERE sd.doc_id = $doc_id
                OPTIONAL MATCH (sd)-[:HAS_ASSESSMENT]->(a:Assessment)
                OPTIONAL MATCH (sd)-[:HAS_OBSERVATION]->(o:Observation)
                OPTIONAL MATCH (sd)-[:HAS_TRAINING]->(t:Training)
                OPTIONAL MATCH (sd)-[:HAS_SUGGESTION]->(s:Suggestion)
                RETURN 
                    collect(DISTINCT a) AS assessments,
                    collect(DISTINCT o) AS observations,
                    collect(DISTINCT t) AS trainings,
                    collect(DISTINCT s) AS suggestions
                """
                with self.driver.session(database=self.database) as session:
                    result_query = session.run(cypher, doc_id=doc_id, subdomain_name=domain_name)
                    record = result_query.single()
                    if record:
                        def node_to_dict(node):
                            if node is None:
                                return None
                            return dict(node)
                        
                        sections_dict = {
                            "assessment": [node_to_dict(n) for n in record["assessments"] if n is not None],
                            "observation": [node_to_dict(n) for n in record["observations"] if n is not None],
                            "training": [node_to_dict(n) for n in record["trainings"] if n is not None],
                            "suggestion": [node_to_dict(n) for n in record["suggestions"] if n is not None],
                        }
                        
                        # 如果有找到 sections，使用該 subdomain 作為 key
                        if any(sections_dict.values()):
                            limited_sections = {}
                            for sec_type in section_types:
                                if sec_type in sections_dict:
                                    limited_sections[sec_type] = sections_dict[sec_type][:k_per_subdomain]
                                else:
                                    limited_sections[sec_type] = []
                            
                            result[domain_name] = {
                                domain_name: limited_sections  # 使用 domain_name 作為 subdomain_name
                            }
                            continue
            
            # 正常流程：通過 domain 查詢 subdomains（允許映射）
            subdomains = self.list_subdomains(doc_id, domain_name, allow_map_to_parent_domain=allow_parent_domain_expand)
            
            # 如果查不到 subdomains，嘗試其他策略
            if not subdomains:
                # 策略 1: 如果 domain_name 是 subdomain，嘗試直接查詢該 subdomain 的 sections
                if _is_subdomain_name(domain_name):
                    # 直接查詢 subdomain 的 sections（不經過 domain）
                    cypher = """
                    MATCH (sd:Subdomain {name: $subdomain_name})
                    WHERE sd.doc_id = $doc_id
                    OPTIONAL MATCH (sd)-[:HAS_ASSESSMENT]->(a:Assessment)
                    OPTIONAL MATCH (sd)-[:HAS_OBSERVATION]->(o:Observation)
                    OPTIONAL MATCH (sd)-[:HAS_TRAINING]->(t:Training)
                    OPTIONAL MATCH (sd)-[:HAS_SUGGESTION]->(s:Suggestion)
                    RETURN 
                        collect(DISTINCT a) AS assessments,
                        collect(DISTINCT o) AS observations,
                        collect(DISTINCT t) AS trainings,
                        collect(DISTINCT s) AS suggestions
                    """
                    with self.driver.session(database=self.database) as session:
                        result_query = session.run(cypher, doc_id=doc_id, subdomain_name=domain_name)
                        record = result_query.single()
                        if record:
                            def node_to_dict(node):
                                if node is None:
                                    return None
                                return dict(node)
                            
                            sections_dict = {
                                "assessment": [node_to_dict(n) for n in record["assessments"] if n is not None],
                                "observation": [node_to_dict(n) for n in record["observations"] if n is not None],
                                "training": [node_to_dict(n) for n in record["trainings"] if n is not None],
                                "suggestion": [node_to_dict(n) for n in record["suggestions"] if n is not None],
                            }
                            
                            # 如果有找到 sections，使用該 subdomain 作為 key
                            if any(sections_dict.values()):
                                limited_sections = {}
                                for sec_type in section_types:
                                    if sec_type in sections_dict:
                                        limited_sections[sec_type] = sections_dict[sec_type][:k_per_subdomain]
                                    else:
                                        limited_sections[sec_type] = []
                                
                                result[domain_name] = {
                                    domain_name: limited_sections  # 使用 domain_name 作為 subdomain_name
                                }
                                continue
            
            # 正常流程：處理找到的 subdomains
            domain_result = {}
            for subdomain in subdomains:
                subdomain_id = subdomain["subdomain_id"]
                subdomain_name = subdomain["name"]
                
                # 取得該 subdomain 的所有 sections
                all_sections = self.fetch_sections_by_subdomain(doc_id, subdomain_id)
                
                # 只保留需要的 section 類型，並限制數量
                limited_sections = {}
                for sec_type in section_types:
                    if sec_type in all_sections:
                        limited_sections[sec_type] = all_sections[sec_type][:k_per_subdomain]
                    else:
                        limited_sections[sec_type] = []
                
                domain_result[subdomain_name] = limited_sections
            
            if domain_result:  # 只有當有結果時才添加
                result[domain_name] = domain_result
        
        return result
    
    def get_score_sections(
        self,
        doc_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        取得分數相關的 Assessment sections（SCORE mode 使用）
        
        Args:
            doc_id: 文件 ID
            limit: 限制數量
        
        Returns:
            List[Dict]: Assessment 節點列表（包含 result/tools/text 欄位）
        """
        cypher = """
        MATCH (r:Report {doc_id: $doc_id})-[:COVERS_DOMAIN]->(d:Domain)
        MATCH (d)-[:HAS_SUBDOMAIN]->(sd:Subdomain)
        MATCH (sd)-[:HAS_ASSESSMENT]->(a:Assessment)
        WHERE a.result IS NOT NULL OR a.tools IS NOT NULL
        RETURN a
        ORDER BY a.group_id, a.id
        LIMIT $limit
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, doc_id=doc_id, limit=limit)
            return [dict(record["a"]) for record in result]
