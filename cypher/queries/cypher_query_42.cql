"""// Assign source from relationships to connected nodes
MATCH (n)-[r]->(m)
WHERE r.source IS NOT NULL  // Only consider relationships with a source property
SET n.source = r.source,
    m.source = r.source
RETURN DISTINCT n, m"""