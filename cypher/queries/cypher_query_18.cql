## get as alist of nds and rels the paths of SCAI graphs of comorbidity
"""
WITH ['alzheimer', 'parkinson'] AS targetKeywords, ['covid'] AS covidKeyword
MATCH (d1), (d2)
WHERE 
  any(keyword IN targetKeywords WHERE toLower(d1.name) CONTAINS toLower(keyword)) 
  AND any(keyword IN covidKeyword WHERE toLower(d2.name) CONTAINS toLower(keyword)) 
  AND toLower(d1.source) CONTAINS "scai"
  AND toLower(d2.source) CONTAINS "scai"

// Step 3: Compute shortest path
MATCH path = shortestPath((d1)-[*..3]-(d2)) // Find shortest path within 3 hops

// Step 4: Extract node IDs and relationship types from the path
WITH DISTINCT [node IN nodes(path) | id(node)] AS pathNodes, 
               [rel IN relationships(path) | type(rel)] AS pathRelationships

// Step 5: Retrieve node details based on node IDs
UNWIND pathNodes AS nodeId
MATCH (n)
WHERE id(n) = nodeId
WITH pathNodes, pathRelationships, n, 
     CASE WHEN 'name' IN keys(n) THEN n.name ELSE n.symbol END AS nodeName

// Step 6: Collect node names back into a list
WITH DISTINCT pathNodes, pathRelationships, collect(nodeName) AS nodeNames

// Step 7: Return the lists of node names and relationship types
RETURN nodeNames AS pathNodes, pathRelationships
"""