# graph density for each source
"""MATCH (n)-[r]-(m)
WHERE r.source CONTAINS 'sherpa'
WITH count(DISTINCT n) + count(DISTINCT m) AS total_nodes, count(r) AS total_relationships
WITH total_nodes, total_relationships,
     CASE
         WHEN total_nodes > 1 THEN total_relationships * 1.0 / (total_nodes * (total_nodes - 1) / 2)
         ELSE 0
     END AS density
RETURN total_nodes, total_relationships, density
"""