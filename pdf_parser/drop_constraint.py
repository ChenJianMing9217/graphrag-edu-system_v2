from neo4j import GraphDatabase
import sys
import os

# 將父目錄加入 sys.path 以便導入 config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import get_neo4j_uri, get_neo4j_auth

def drop_domain_constraint():
    uri = get_neo4j_uri()
    user, password = get_neo4j_auth()
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session() as session:
        print("正在查詢 Neo4j 中的約束條件...")
        # 查詢所有約束
        result = session.run("SHOW CONSTRAINTS")
        constraints = []
        for record in result:
            # Neo4j 5.x 欄位名稱可能是 name, type, entityType, labelsOrTypes, properties
            # 或者是用舊版的 'description'
            constraints.append(record)
            
        target_constraints = []
        for c in constraints:
            # 檢查是否為 Domain 或 Subdomain 標籤且包含 name 屬性的唯一性項目
            c_name = c.get('name')
            c_labels = c.get('labelsOrTypes')
            c_props = c.get('properties')
            
            if c_labels and ('Domain' in c_labels or 'Subdomain' in c_labels) and \
               c_props and 'name' in c_props:
                target_constraints.append(c_name)
        
        if target_constraints:
            for tc in target_constraints:
                print(f"找到目標約束: {tc}，正在刪除...")
                session.run(f"DROP CONSTRAINT {tc}")
            print("✓ 約束已成功刪除！現在相同名稱的領域與子領域可以共存於不同報告中了。")
        else:
            print("未能自動找到 Domain(name) 或 Subdomain(name) 的唯一性約束。")
            print("如果您仍看到 ConstraintError，請手動在 Neo4j Console 執行：")
            print("SHOW CONSTRAINTS")
            print("然後根據名稱執行 DROP CONSTRAINT <name>")

    driver.close()

if __name__ == "__main__":
    drop_domain_constraint()
