""" 
WITH ['alzheimer', 'parkinson', 'neurodegenerative', 'covid','neurodegeneration'] AS keywords
MATCH (d)
WHERE any(keyword IN keywords WHERE toLower(d.name) CONTAINS keyword)
WITH collect(d) AS diseases

// Step 2: Generate unique pairs
UNWIND diseases AS d1
UNWIND diseases AS d2
WITH d1, d2
WHERE id(d1) < id(d2)  // Ensure unique pairs and avoid self-matches

// Step 3: Compute shortest path
MATCH path = shortestPath((d1)-[*..3]-(d2)) // Find shortest path within 3 hops
RETURN DISTINCT path
"""


#above but defining where source is coming from
"""

WITH ['alzheimer', 'parkinson', 'neurodegenerative', 'covid', 'neurodegeneration'] AS keywords
MATCH (d)
WHERE any(keyword IN keywords WHERE toLower(d.name) CONTAINS keyword)
WITH collect(d) AS diseases

// Step 2: Generate unique pairs
UNWIND diseases AS d1
UNWIND diseases AS d2
WITH d1, d2
WHERE id(d1) < id(d2)  // Ensure unique pairs and avoid self-matches

// Step 3: Compute shortest path with filtered relationships
MATCH path = shortestPath((d1)-[r*..3]-(d2)) // Find shortest path within 3 hops
WHERE all(rel IN relationships(path) WHERE 
    any(source_keyword IN ['opentargets', 'disgenet', 'drugbank', 'indra'] 
        WHERE toLower(rel.source) CONTAINS source_keyword)) // Check if source contains any of the target keywords
RETURN DISTINCT path


"""


# above but get paths as list of ndoeds and rels when defining source type
"""
WITH ['alzheimer', 'parkinson', 'neurodegenerative', 'covid', 'neurodegeneration'] AS keywords
MATCH (d)
WHERE any(keyword IN keywords WHERE toLower(d.name) CONTAINS keyword)
WITH collect(d) AS diseases

// Step 2: Generate unique pairs
UNWIND diseases AS d1
UNWIND diseases AS d2
WITH d1, d2 WHERE id(d1) < id(d2)  // Ensure unique pairs and avoid self-matches

// Step 3: Compute shortest path with filtered relationships
MATCH path = shortestPath((d1)-[r*..3]-(d2)) // Find shortest path within 3 hops
WHERE all(rel IN relationships(path) WHERE 
    any(source_keyword IN ['opentargets', 'disgenet', 'drugbank', 'indra'] 
        WHERE toLower(rel.source) CONTAINS source_keyword)) // Filter by source

// Step 4: Extract path nodes and relationships
WITH DISTINCT [node IN nodes(path) | id(node)] AS pathNodes, 
                [rel IN relationships(path) | type(rel)] AS pathRelationships

// Step 5: Retrieve node details based on ID
UNWIND pathNodes AS nodeId
MATCH (n)
WHERE id(n) = nodeId
WITH pathNodes, pathRelationships, n, 
     CASE WHEN 'name' IN keys(n) THEN n.name ELSE n.symbol END AS nodeName

// Collect the node names back into a list
WITH DISTINCT pathNodes, pathRelationships, collect(nodeName) AS nodeNames

// Return the lists of node names and relationship types
RETURN nodeNames AS pathNodes, pathRelationships
"""