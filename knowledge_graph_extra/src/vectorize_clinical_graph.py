import requests
import numpy as np
from neo4j import GraphDatabase
import time

# 配置
URI = 'bolt://192.168.150.136:7687'
AUTH = ('neo4j', 'password')
EMBED_SERVER = 'http://192.168.150.136:8080/embed'

class ClinicalVectoriser:
    def __init__(self):
        self.driver = GraphDatabase.driver(URI, auth=AUTH)

    def close(self):
        self.driver.close()

    def get_embedding(self, text):
        """
        透過新的 Qwen3-Embedding 本地伺服器獲取向量
        """
        if not text:
            return [0.0] * 1024
        
        url = "http://localhost:8001/v1/embeddings"
        data = {
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "input": text
        }
        
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            res_data = response.json()
            if "data" in res_data and len(res_data["data"]) > 0:
                return res_data["data"][0]["embedding"]
            return [0.0] * 1024
        except Exception as e:
            print(f"Error fetching embedding for '{text[:20]}...': {e}")
            return [0.0] * 1024

    def clear_all_embeddings(self):
        """
        清空圖譜中所有節點的向量 (用於模型更換時的重新對齊)
        """
        print("\nClearing all existing embeddings for re-alignment...")
        labels = ["Ability", "ObservationIndicator", "TrainingStrategy", "Milestone"]
        with self.driver.session() as session:
            for lbl in labels:
                session.run(f"MATCH (n:{lbl}) SET n.embedding = NULL")
        print("Embeddings cleared.")

    def vectorize_all_nodes(self, force=False):
        if force:
            self.clear_all_embeddings()
            
        labels = ["Ability", "ObservationIndicator", "TrainingStrategy", "Milestone"]
        
        for label in labels:
            print(f"\nProcessing Label: [{label}]")
            with self.driver.session() as session:
                # 找出尚未向量化的節點
                text_prop = "content" if label == "ObservationIndicator" else ("expected_behavior" if label == "Milestone" else "name")
                
                query = f"MATCH (n:{label}) WHERE n.embedding IS NULL RETURN elementId(n) as eid, n.{text_prop} as text"
                nodes = session.run(query)
                
                count = 0
                batch = []
                for node in nodes:
                    eid = node['eid']
                    text = node['text']
                    if not text: continue
                    
                    embedding = self.get_embedding(text)
                    if embedding:
                        batch.append({"eid": eid, "emb": embedding})
                        count += 1
                    
                    if len(batch) >= 20:
                        self._update_batch(label, batch)
                        batch = []
                        print(f"  Progress: Updated {count} nodes of type {label}...")

                if batch:
                    self._update_batch(label, batch)
                print(f"Finished {label}: Total {count} nodes updated.")

    def _update_batch(self, label, batch):
        with self.driver.session() as session:
            query = f"""
            UNWIND $batch AS item
            MATCH (n) WHERE elementId(n) = item.eid
            SET n.embedding = item.emb
            """
            session.run(query, batch=batch)

    def create_vector_indexes(self):
        """
        建立向量索引 (如果已存在則不重複建立)
        """
        print("\nCreating Neo4j Vector Indexes (1024-dim, COSINE)...")
        with self.driver.session() as session:
            indices = [
                ("abilityVectorIndex", "Ability"),
                ("obsVectorIndex", "ObservationIndicator"),
                ("strategyVectorIndex", "TrainingStrategy"),
                ("milestoneVectorIndex", "Milestone")
            ]
            for idx_name, label in indices:
                try:
                    # 確保索引配置為 1024 維度
                    query = f"""
                    CREATE VECTOR INDEX `{idx_name}` IF NOT EXISTS
                    FOR (n:{label})
                    ON (n.embedding)
                    OPTIONS {{
                      indexConfig: {{
                        `vector.dimensions`: 1024,
                        `vector.similarity_function`: 'cosine'
                      }}
                    }}
                    """
                    session.run(query)
                    print(f"  Index Checked/Created: {idx_name}")
                except Exception as e:
                    print(f"  Index Error {idx_name}: {e}")

if __name__ == "__main__":
    vectorizer = ClinicalVectoriser()
    # 執行全量重新向量化 (以適應新模型 Qwen3)
    vectorizer.vectorize_all_nodes(force=True)
    # 建立/更新索引
    vectorizer.create_vector_indexes()
    vectorizer.close()
