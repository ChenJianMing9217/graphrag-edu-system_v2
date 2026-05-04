import json
import os
from neo4j import GraphDatabase

URI = 'bolt://192.168.150.136:7687'
AUTH = ('neo4j', 'password')

def export_clinical_metadata():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    
    # 取得目前檔案路徑，確保能找到 JSON (現在位於 ../json)
    base_path = os.path.dirname(os.path.abspath(__file__))
    json_dir = os.path.join(os.path.dirname(base_path), 'json')

    with driver.session() as session:
        # 1. 導出臨床詞彙字典 (用於搜尋加權)
        print("Exporting clinical dictionary...")
        res = session.run("MATCH (n) WHERE n:Ability OR n:Subdomain OR n:Domain RETURN DISTINCT n.name as name")
        names = [r['name'] for r in res if r['name'] and len(r['name']) >= 2]
        dict_path = os.path.join(json_dir, 'clinical_dictionary.json')
        with open(dict_path, 'w', encoding='utf-8') as f:
            json.dump(names, f, ensure_ascii=False)
        print(f"  Saved {len(names)} entities to {dict_path}")

        # 2. 自動識別超級節點 (Sink Nodes)
        print("\nIdentifying sink nodes (Hubs)...")
        res = session.run("""
            MATCH (n) WHERE n:Ability OR n:Subdomain OR n:Domain
            MATCH (n)-[r]-()
            WITH n, count(r) as dc
            WHERE dc > 15
            RETURN n.name as name, dc
            ORDER BY dc DESC
        """)
        sink_nodes = [r['name'] for r in res]
        sink_path = os.path.join(json_dir, 'sink_nodes.json')
        with open(sink_path, 'w', encoding='utf-8') as f:
            json.dump(sink_nodes, f, ensure_ascii=False)
        print(f"  Identified {len(sink_nodes)} sink nodes to {sink_path}")

    driver.close()

if __name__ == "__main__":
    export_clinical_metadata()
