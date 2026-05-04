// 1. Setup Constraints
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (s:Subdomain) REQUIRE s.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (a:Ability) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (m:Milestone) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (src:Source) REQUIRE src.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (obs:ObservationIndicator) REQUIRE obs.indicator_id IS UNIQUE;

// 2. Load Domains
CALL apoc.load.json("file:///v3/domains.json") YIELD value
MERGE (d:Domain {name: value.name})
SET d.description = value.description;

// 3. Load Subdomains
CALL apoc.load.json("file:///v3/subdomains.json") YIELD value
MERGE (s:Subdomain {name: value.name})
SET s.description = value.description;

// 4. Load Domain -> Subdomain Relations
CALL apoc.load.json("file:///v3/rel_domain_subdomain.json") YIELD value
MATCH (d:Domain {name: value.from})
MATCH (s:Subdomain {name: value.to})
MERGE (d)-[:HAS_SUBDOMAIN]->(s);

// 5. Load Abilities (v3 - Only Standardized/Stable Concepts)
CALL apoc.load.json("file:///v3/abilities_v3.json") YIELD value
MERGE (a:Ability {id: value.id})
SET a.name = value.name,
    a.aliases = value.aliases,
    a.description = value.description,
    a.source_type = value.source;

// 6. Load Subdomain -> Ability Relations
CALL apoc.load.json("file:///v3/rel_subdomain_ability_v3.json") YIELD value
MATCH (s:Subdomain {name: value.from})
MATCH (a:Ability {name: value.to})  // Assuming v3 rels use 'to' = ability.name
MERGE (s)-[:HAS_ABILITY]->(a);

// 7. Load Milestones (v3 - Fixed linkages)
CALL apoc.load.json("file:///v3/milestones_v3_fixed.json") YIELD value
MERGE (m:Milestone {id: value.id})
SET m.expected_behavior = value.expected_behavior,
    m.age_min_month = toInteger(value.age_min_month),
    m.age_max_month = toInteger(value.age_max_month),
    m.note = value.note,
    m.needs_review = value.needs_review;

// 8. Load Ability -> Milestone Relations
CALL apoc.load.json("file:///v3/rel_ability_milestone_v3.json") YIELD value
MATCH (a:Ability {name: value.from})
MATCH (m:Milestone {id: value.to})
MERGE (a)-[:HAS_MILESTONE]->(m);

// 9. Load Sources and Milestone -> Source
CALL apoc.load.json("file:///v3/sources.json") YIELD value
MERGE (src:Source {id: value.id})
SET src.name = value.name,
    src.origin = value.origin,
    src.url = value.url;

CALL apoc.load.json("file:///v3/rel_milestone_source.json") YIELD value
MATCH (m:Milestone {id: value.from})
MATCH (src:Source {id: value.to})
MERGE (m)-[:SOURCED_FROM]->(src);

// 10. Load Observation Indicators
CALL apoc.load.json("file:///v3/observation_indicators_v3.json") YIELD value
MERGE (obs:ObservationIndicator {indicator_id: value.indicator_id})
SET obs.domain = value.domain,
    obs.subdomain = value.subdomain,
    obs.content = value.content,
    obs.age_min = toInteger(value.age_min),
    obs.age_max = toInteger(value.age_max),
    obs.guidance = value.guidance,
    obs.needs_review = value.needs_review;

// 11. Load Observation -> Ability Relations
CALL apoc.load.json("file:///v3/rel_observation_ability_v3.json") YIELD value
MATCH (obs:ObservationIndicator {indicator_id: value.from})
MATCH (a:Ability {name: value.to})
MERGE (obs)-[:INDICATES_ABILITY]->(a);

// 12. Load Observation -> Subdomain/Domain
CALL apoc.load.json("file:///v3/rel_observation_subdomain_v3.json") YIELD value
MATCH (obs:ObservationIndicator {indicator_id: value.from})
MATCH (s:Subdomain {name: value.to})
MERGE (obs)-[:RELATES_TO_SUBDOMAIN]->(s);

CALL apoc.load.json("file:///v3/rel_observation_domain_v3.json") YIELD value
MATCH (obs:ObservationIndicator {indicator_id: value.from})
MATCH (d:Domain {name: value.to})
MERGE (obs)-[:RELATES_TO_DOMAIN]->(d);
