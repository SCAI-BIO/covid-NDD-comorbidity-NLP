"""  
WITH ['Sherpa', 'PubTator', 'CBM', 'scai', 'opentargets', 'disgenet', 'indra', 'drugbank', 'kegg'] AS sources
UNWIND sources AS source
MATCH (n)-[r]-(m)
WHERE toLower(r.source) CONTAINS toLower(source)
WITH source, 
     count(DISTINCT id(n) + id(r) + id(m)) AS unique_triples, 
     count(r) AS total_triples, 
     count(DISTINCT n) AS unique_nodes, 
     count(DISTINCT m) AS unique_target_nodes
WITH source, unique_nodes + unique_target_nodes AS total_nodes, unique_nodes, unique_triples, total_triples
WITH source, total_nodes, unique_nodes, unique_triples, total_triples,
     CASE
         WHEN total_nodes > 1 THEN unique_triples * 1.0 / (total_nodes * (total_nodes - 1) / 2)
         ELSE 0
     END AS density
RETURN source, total_nodes, unique_nodes, unique_triples, total_triples, density
ORDER BY source

"""

#for primekg
"""  
MATCH (n)-[r]-(m)
WITH 
     count(DISTINCT id(n) + id(r) + id(m)) AS unique_triples, 
     count(r) AS total_triples, 
     count(DISTINCT n) AS unique_nodes, 
     count(DISTINCT m) AS unique_target_nodes
WITH unique_nodes + unique_target_nodes AS total_nodes, unique_nodes, unique_triples, total_triples
WITH total_nodes, unique_nodes, unique_triples, total_triples,
     CASE
         WHEN total_nodes > 1 THEN unique_triples * 1.0 / (total_nodes * (total_nodes - 1) / 2)
         ELSE 0
     END AS density
RETURN total_nodes, unique_nodes, unique_triples, total_triples, density
ORDER BY total_nodes DESC

"""