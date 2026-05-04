import json
import os
from neo4j import GraphDatabase

URI = 'bolt://192.168.150.136:7687'
AUTH = ('neo4j', 'password')
# 取得目前檔案路徑，確保能找到 JSON (現在位於 ../json)
base_path = os.path.dirname(os.path.abspath(__file__))
json_dir = os.path.join(os.path.dirname(base_path), 'json')

def load(filename):
    filepath = os.path.join(json_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_import():
    driver = GraphDatabase.driver(URI, auth=AUTH)

    # Load V3 abilities for audit
    abls = load('abilities_v3.json')
    abl_names = {a['name'] for a in abls}

    with driver.session() as session:
        def safe_run(query, data, name):
            if not data: return
            try:
                res = session.run(query, batch=data)
                summary = res.consume()
                nc = summary.counters.nodes_created
                rc = summary.counters.relationships_created
                print(f"  {name}: {len(data)} items -> Nodes:{nc}, Rels:{rc}")
            except Exception as e:
                print(f"  Error {name}: {e}")

        # ==========================================
        # 1. TRAINING STRATEGIES (training_directions)
        # ==========================================
        print("=== Loading Training Strategies ===")

        # 1a. Constraint
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (ts:TrainingStrategy) REQUIRE ts.id IS UNIQUE")

        # 1b. Strategy Nodes
        strategies = load('training_strategies.json')
        safe_run("""
            UNWIND $batch AS row
            MERGE (ts:TrainingStrategy {id: row.id})
            SET ts.name = row.name,
                ts.subdomain_name = row.subdomain_name,
                ts.domain_name = row.domain_name,
                ts.description = row.description
        """, strategies, "TrainingStrategy nodes")

        # 1c. Strategy -> Subdomain
        rel_ts_sub = load('rel_training_strategy_subdomain.json')
        safe_run("""
            UNWIND $batch AS row
            MATCH (ts:TrainingStrategy {id: row.from})
            MATCH (s:Subdomain {name: row.to})
            MERGE (ts)-[:BELONGS_TO_SUBDOMAIN]->(s)
        """, rel_ts_sub, "Strategy->Subdomain")

        # 1d. Strategy -> Ability (filter out missing abilities)
        rel_ts_abl = load('rel_training_strategy_ability.json')
        valid_ts_abl = [r for r in rel_ts_abl if r['to'] in abl_names]
        missing_ts_abl = [r for r in rel_ts_abl if r['to'] not in abl_names]
        print(f"  Strategy->Ability: {len(valid_ts_abl)} valid, {len(missing_ts_abl)} skipped (missing ability)")
        if missing_ts_abl:
            print(f"    Missing: {list({r['to'] for r in missing_ts_abl})}")
        safe_run("""
            UNWIND $batch AS row
            MATCH (ts:TrainingStrategy {id: row.from})
            MATCH (a:Ability {name: row.to})
            MERGE (ts)-[:TARGETS_ABILITY]->(a)
        """, valid_ts_abl, "Strategy->Ability")

        # ==========================================
        # 2. TRAINING ACTIVITIES (recommandation)
        # ==========================================
        print("\n=== Loading Training Activities ===")

        # 2a. Constraint
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (ta:TrainingActivity) REQUIRE ta.id IS UNIQUE")

        # 2b. Activity Nodes
        activities = load('training_activities.json')
        safe_run("""
            UNWIND $batch AS row
            MERGE (ta:TrainingActivity {id: row.id})
            SET ta.name = row.name,
                ta.strategy_name = row.strategy_name,
                ta.description = row.description
        """, activities, "TrainingActivity nodes")

        # 2c. Activity -> Strategy
        rel_ta_ts = load('rel_training_activity_strategy.json')
        safe_run("""
            UNWIND $batch AS row
            MATCH (ta:TrainingActivity {id: row.from})
            MATCH (ts:TrainingStrategy {name: row.to})
            MERGE (ta)-[:IMPLEMENTS_STRATEGY]->(ts)
        """, rel_ta_ts, "Activity->Strategy")

        # 2d. Activity -> Ability (filter out missing abilities)
        rel_ta_abl = load('rel_training_activity_ability.json')
        valid_ta_abl = [r for r in rel_ta_abl if r['to'] in abl_names]
        missing_ta_abl = [r for r in rel_ta_abl if r['to'] not in abl_names]
        print(f"  Activity->Ability: {len(valid_ta_abl)} valid, {len(missing_ta_abl)} skipped (missing ability)")
        if missing_ta_abl:
            print(f"    Missing: {list({r['to'] for r in missing_ta_abl})}")
        safe_run("""
            UNWIND $batch AS row
            MATCH (ta:TrainingActivity {id: row.from})
            MATCH (a:Ability {name: row.to})
            MERGE (ta)-[:TARGETS_ABILITY]->(a)
        """, valid_ta_abl, "Activity->Ability")

        print("\nFinished importing Training Strategies & Activities!")

    driver.close()

if __name__ == '__main__':
    run_import()
