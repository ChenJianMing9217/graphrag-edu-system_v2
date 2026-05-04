from neo4j import GraphDatabase

URI = 'bolt://192.168.150.136:7687'
AUTH = ('neo4j', 'password')

def setup_indexes():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver.session() as session:
        print("--- Setting up Full-Text Indexes in Neo4j ---")
        
        # 1. Drop existing indexes if any (optional, but safer)
        # Note: In 5.x, you drop by name, but we don't know the names yet.
        # We'll just use CREATE OR REPLACE if the version supports it, or handle errors.

        indexes = [
            ("abilityIndex", "Ability", ["name"]),
            ("obsIndex", "ObservationIndicator", ["content"]),
            ("strategyIndex", "TrainingStrategy", ["name", "description"]),
            ("activityIndex", "TrainingActivity", ["name", "description"]),
            ("milestoneIndex", "Milestone", ["expected_behavior"]),
            ("subdomainIndex", "Subdomain", ["name"])
        ]


        for name, label, props in indexes:
            try:
                # 5.x Syntax
                query = f"CREATE FULLTEXT INDEX {name} IF NOT EXISTS FOR (n:{label}) ON EACH [{', '.join([f'n.{p}' for p in props])}]"
                session.run(query)
                print(f"  Created Index: {name} on {label}({', '.join(props)})")
            except Exception as e:
                print(f"  Error creating index {name}: {e}")

        # Verification
        print("\n--- Current Full-Text Indexes ---")
        res = session.run("SHOW FULLTEXT INDEXES")
        for r in res:
            print(f"  - {r['name']} (State: {r['state']})")

    driver.close()

if __name__ == '__main__':
    setup_indexes()
