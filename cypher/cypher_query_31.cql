#correct general statistics above for undirected graphs
""" 
WITH ['Sherpa', 'PubTator', 'CBM', 'scai', 'opentargets', 'disgenet', 'indra', 'drugbank', 'kegg'] AS sources
UNWIND sources AS source
MATCH (n)-[r]-(m)
WHERE toLower(r.source) CONTAINS toLower(source) AND id(n) < id(m)
WITH source, 
     count(DISTINCT id(n) + id(r) + id(m)) AS unique_triples, 
     count(r) AS total_triples, 
     collect(DISTINCT n) + collect(DISTINCT m) AS unique_nodes,
     collect(n) + collect(m) AS total_nodes
WITH source, 
     unique_triples, 
     total_triples,
     size(apoc.coll.toSet(unique_nodes)) AS unique_node_count,
     size(total_nodes) AS total_node_count
WITH source, unique_node_count, total_node_count, unique_triples, total_triples,
     CASE
         WHEN unique_node_count > 1 THEN unique_triples * 2.0 / (unique_node_count * (unique_node_count - 1))
         ELSE 0
     END AS density
RETURN source, unique_node_count, total_node_count, unique_triples, total_triples, density
ORDER BY source

""" 

#this is for directed graphs (SHERPA, CBM, SCAI, INDRA, KEGG (pathology))
""" 
WITH ['Sherpa', 'PubTator', 'CBM', 'scai', 'opentargets', 'disgenet', 'indra', 'drugbank', 'kegg'] AS sources
UNWIND sources AS source
MATCH (n)-[r]->(m)
WHERE toLower(r.source) CONTAINS toLower(source)
WITH source, 
     count(DISTINCT id(n) + id(r) + id(m)) AS unique_triples, 
     count(r) AS total_triples, 
     collect(DISTINCT n) + collect(DISTINCT m) AS unique_nodes,
     collect(n) + collect(m) AS total_nodes
WITH source, 
     unique_triples, 
     total_triples,
     size(apoc.coll.toSet(unique_nodes)) AS unique_node_count,
     size(total_nodes) AS total_node_count
WITH source, unique_node_count, total_node_count, unique_triples, total_triples,
     CASE
         WHEN unique_node_count > 1 THEN unique_triples * 1.0 / (unique_node_count * (unique_node_count - 1))
         ELSE 0
     END AS density
RETURN source, unique_node_count, total_node_count, unique_triples, total_triples, density
ORDER BY source

""" 

#For primekg (old incorrect)
"""  


MATCH (n)-[r]->(m)
WITH 
     count(DISTINCT id(n) + id(r) + id(m)) AS unique_triples, 
     count(r) AS total_triples, 
     collect(DISTINCT n) + collect(DISTINCT m) AS unique_nodes,
     collect(n) + collect(m) AS total_nodes
WITH 
     unique_triples, 
     total_triples,
     size(apoc.coll.toSet(unique_nodes)) AS unique_node_count,
     size(total_nodes) AS total_node_count
WITH unique_node_count, total_node_count, unique_triples, total_triples,
     CASE
         WHEN unique_node_count > 1 THEN unique_triples * 1.0 / (unique_node_count * (unique_node_count - 1))
         ELSE 0
     END AS density
RETURN unique_node_count, total_node_count, unique_triples, total_triples, density
ORDER BY unique_node_count DESC


"""

#For primekg corrected

"""  
MATCH (n)-[r]->(m)
WHERE any(term IN ["covid", "alzheimer", "neurodegeneration", "parkinson"] 
           WHERE n.name =~ ('(?i).*' + term + '.*') 
              OR m.name =~ ('(?i).*' + term + '.*'))
WITH 
     count(DISTINCT id(n) + id(r) + id(m)) AS unique_triples, 
     count(r) AS total_triples, 
     collect(DISTINCT n) + collect(DISTINCT m) AS unique_nodes,
     collect(n) + collect(m) AS total_nodes
WITH 
     unique_triples, 
     total_triples,
     size(apoc.coll.toSet(unique_nodes)) AS unique_node_count,
     size(total_nodes) AS total_node_count
WITH unique_node_count, total_node_count, unique_triples, total_triples,
     CASE
         WHEN unique_node_count > 1 THEN unique_triples * 1.0 / (unique_node_count * (unique_node_count - 1))
         ELSE 0
     END AS density
RETURN unique_node_count, total_node_count, unique_triples, total_triples, density
ORDER BY unique_node_count DESC
"""