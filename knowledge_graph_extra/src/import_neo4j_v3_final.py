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
    if not os.path.exists(filepath):
        print(f"Skipping {filename}: File not found.")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_import():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    
    with driver.session() as session:
        # 0. Clear DB
        print("Clearing database...")
        session.run("MATCH (n) DETACH DELETE n")

        # 1. Setup Constraints
        print("Setting up constraints...")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Subdomain) REQUIRE s.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Ability) REQUIRE a.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Milestone) REQUIRE m.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (src:Source) REQUIRE src.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (obs:ObservationIndicator) REQUIRE obs.indicator_id IS UNIQUE")

        def safe_run(query, data, name):
            if not data: return
            try:
                res = session.run(query, batch=data)
                created_nodes = res.consume().counters.nodes_created
                created_rels = res.consume().counters.relationships_created
                print(f"Loaded {name}: {len(data)} items processed. (Nodes: {created_nodes}, Rels: {created_rels})")
            except Exception as e:
                print(f"Error loading {name}: {e}")

        # 2. Domains
        domains = load('domains.json')
        safe_run("""
            UNWIND $batch AS row
            MERGE (d:Domain {name: row.name})
            SET d.description = row.description
        """, domains, "Domains")

        # 3. Subdomains
        subdomains = load('subdomains.json')
        safe_run("""
            UNWIND $batch AS row
            MERGE (s:Subdomain {name: row.name})
            SET s.description = row.description
        """, subdomains, "Subdomains")

        # 4. Domain -> Subdomain
        rel_dom_sub = load('rel_domain_subdomain.json')
        safe_run("""
            UNWIND $batch AS row
            MATCH (d:Domain {name: row.from})
            MATCH (s:Subdomain {name: row.to})
            MERGE (d)-[:HAS_SUBDOMAIN]->(s)
        """, rel_dom_sub, "Domain->Subdomain")

        # 5. Abilities
        abilities = load('abilities_v3.json')
        safe_run("""
            UNWIND $batch AS row
            MERGE (a:Ability {id: row.id})
            SET a.name = row.name,
                a.aliases = row.aliases,
                a.description = row.description,
                a.source_type = row.source
        """, abilities, "Abilities")

        # 6. Subdomain -> Ability
        rel_sub_abl = load('rel_subdomain_ability_v3.json')
        safe_run("""
            UNWIND $batch AS row
            MATCH (s:Subdomain {name: trim(row.from)})
            MATCH (a:Ability {name: trim(row.to)})
            MERGE (s)-[:HAS_ABILITY]->(a)
        """, rel_sub_abl, "Subdomain->Ability")

        # 7. Milestones
        milestones = load('milestones_v3_fixed.json')
        safe_run("""
            UNWIND $batch AS row
            MERGE (m:Milestone {id: row.id})
            SET m.expected_behavior = row.expected_behavior,
                m.age_min_month = toInteger(row.age_min_month),
                m.age_max_month = toInteger(row.age_max_month),
                m.note = row.note,
                m.needs_review = row.needs_review
        """, milestones, "Milestones")

        # 8. Ability -> Milestone
        rel_abl_mil = load('rel_ability_milestone_v3.json')
        safe_run("""
            UNWIND $batch AS row
            MATCH (a:Ability {name: trim(row.from)})
            MATCH (m:Milestone {id: row.to})
            MERGE (a)-[:HAS_MILESTONE]->(m)
        """, rel_abl_mil, "Ability->Milestone")

        # 9. Sources
        sources = load('sources.json')
        safe_run("""
            UNWIND $batch AS row
            MERGE (src:Source {id: row.id})
            SET src.name = row.name,
                src.origin = row.origin,
                src.url = row.url
        """, sources, "Sources")

        rel_mil_src = load('rel_milestone_source.json')
        safe_run("""
            UNWIND $batch AS row
            MATCH (m:Milestone {id: row.from})
            MATCH (src:Source {id: row.to})
            MERGE (m)-[:SOURCED_FROM]->(src)
        """, rel_mil_src, "Milestone->Source")

        # 10. Observation Indicators (Fix: use row.id and row.name mapping correctly)
        observations = load('observation_indicators_v3.json')
        safe_run("""
            UNWIND $batch AS row
            MERGE (obs:ObservationIndicator {indicator_id: row.id})
            SET obs.domain = row.primary_domain,
                obs.subdomain = row.primary_subdomain,
                obs.content = row.name,
                obs.age_min = toInteger(row.age_min),
                obs.age_max = toInteger(row.age_max),
                obs.guidance = row.guidance,
                obs.needs_review = row.needs_review
        """, observations, "Observations")

        # 11. Observation -> Ability
        rel_obs_abl = load('rel_observation_ability_v3.json')
        safe_run("""
            UNWIND $batch AS row
            MATCH (obs:ObservationIndicator {indicator_id: row.from})
            MATCH (a:Ability {name: trim(row.to)})
            MERGE (obs)-[:INDICATES_ABILITY]->(a)
        """, rel_obs_abl, "Obs->Ability")

        # 12. Observation -> Subdomain/Domain
        rel_obs_sub = load('rel_observation_subdomain_v3.json')
        safe_run("""
            UNWIND $batch AS row
            MATCH (obs:ObservationIndicator {indicator_id: row.from})
            MATCH (s:Subdomain {name: trim(row.to)})
            MERGE (obs)-[:RELATES_TO_SUBDOMAIN]->(s)
        """, rel_obs_sub, "Obs->Subdomain")

        rel_obs_dom = load('rel_observation_domain_v3.json')
        safe_run("""
            UNWIND $batch AS row
            MATCH (obs:ObservationIndicator {indicator_id: row.from})
            MATCH (d:Domain {name: trim(row.to)})
            MERGE (obs)-[:RELATES_TO_DOMAIN]->(d)
        """, rel_obs_dom, "Obs->Domain")

        print("Finished building V3 Graph successfully in one pass!")

    driver.close()

if __name__ == '__main__':
    run_import()
