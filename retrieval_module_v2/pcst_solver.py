from typing import List, Dict, Any, Set
from .types import CandidateNode

class PCSTSolver:
    """
    Sub-graph discovery tool using Steiner Tree heuristics (Prize-Collecting approach).
    Connects isolated candidate nodes to provide more coherent context.
    """
    def __init__(self, graph_client):
        self.graph_client = graph_client

    def find_subgraph(
        self, 
        terminals: List[CandidateNode], 
        doc_id: str, 
        max_hops: int = 2,
        max_total_nodes: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Finds a connected sub-graph that includes as many terminals as possible.
        
        Returns:
            A list of relationship triples (node1, relation, node2) representing the context graph.
        """
        if not terminals or len(terminals) < 2:
            return []

        # 1. Selection of top terminals (limit to avoid combinatorial explosion)
        top_terminals = sorted(terminals, key=lambda x: x.score, reverse=True)[:5]
        terminal_names = [(t.properties.get("name") or t.text[:20], f"{t.score:.4f}", t.label) for t in top_terminals]
        print(f"[PCST] 選定前 {len(top_terminals)} 個終端節點進行連通: {terminal_names}")
        terminal_ids = {t.node_id for t in top_terminals}
        
        # 2. Find paths between pairs of terminals
        # Cypher query for shortest paths
        cypher = """
        MATCH (n1 {id: $id1}), (n2 {id: $id2})
        MATCH p = shortestPath((n1)-[*1..%d]-(n2))
        WHERE ALL(node IN nodes(p) WHERE coalesce(node.doc_id, $doc_id) = $doc_id)
        RETURN p
        """ % max_hops

        edges = []
        visited_nodes = set()
        
        with self.graph_client.driver.session(database=self.graph_client.database) as session:
            for i in range(len(top_terminals)):
                for j in range(i + 1, len(top_terminals)):
                    res = session.run(
                        cypher, 
                        id1=top_terminals[i].node_id, 
                        id2=top_terminals[j].node_id,
                        doc_id=doc_id
                    )
                    for record in res:
                        path = record["p"]
                        self._extract_relationships(path, edges, visited_nodes)

        # 3. Limit total nodes if necessary (optional pruning)
        # If the graph is too big, it might overwhelm the LLM context.
        return edges[:max_total_nodes]

    def _extract_relationships(self, path, edges_list, visited_nodes):
        """Extracts relationship triples from a Neo4j path object."""
        nodes = path.nodes
        relationships = path.relationships
        
        for rel in relationships:
            # Get start and end nodes of this relationship
            start_node = rel.start_node
            end_node = rel.end_node
            
            # Create a serializable triple
            triple = {
                "source": {
                    "id": start_node.get("id"),
                    "name": start_node.get("name") or start_node.get("text", "")[:30],
                    "label": list(start_node.labels)[0] if start_node.labels else "Unknown"
                },
                "relation": rel.type,
                "target": {
                    "id": end_node.get("id"),
                    "name": end_node.get("name") or end_node.get("text", "")[:30],
                    "label": list(end_node.labels)[0] if end_node.labels else "Unknown"
                }
            }
            
            # Deduplicate by source_id - relation - target_id
            triple_key = f"{triple['source']['id']}-{triple['relation']}-{triple['target']['id']}"
            if triple_key not in visited_nodes:
                edges_list.append(triple)
                visited_nodes.add(triple_key)
