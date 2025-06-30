""""
WITH ['alzheimer', 'parkinson', 'neurodegenerative', 'covid', 'neurodegeneration'] AS keywords

// Step 1: Group similar nodes
MATCH (d)
WHERE any(keyword IN keywords WHERE toLower(d.name) CONTAINS keyword)
WITH d.name AS name, collect(d) AS similar_nodes, keywords
WITH apoc.text.levenshteinSimilarity(name, reduce(s = '', k IN keywords | s + k)) AS similarity, name, similar_nodes
WITH similarity, name, similar_nodes
ORDER BY similarity DESC
WITH collect({similarity: similarity, name: name, nodes: similar_nodes}) AS grouped_nodes

// Step 2: Generate unique pairs of grouped nodes
UNWIND range(0, size(grouped_nodes)-2) AS i
UNWIND range(i+1, size(grouped_nodes)-1) AS j
WITH grouped_nodes[i] AS group1, grouped_nodes[j] AS group2

// Step 3: Compute shortest path with filtered relationships
UNWIND group1.nodes AS n1
UNWIND group2.nodes AS n2
MATCH path = shortestPath((n1)-[r*..3]-(n2)) // Find shortest path within 3 hops
WHERE all(rel IN relationships(path) WHERE 
    any(source_keyword IN ['opentargets', 'disgenet', 'drugbank', 'indra'] 
        WHERE toLower(rel.source) CONTAINS source_keyword)) // Check if source contains any of the target keywords

// Step 4: Extract and format path information
WITH 
    group1.name AS disease1,
    group2.name AS disease2,
    [node IN nodes(path) | node.name] AS nodeNames,
    [rel IN relationships(path) | type(rel)] AS relTypes,
    [rel IN relationships(path) | rel.source] AS relSources,
    length(path) AS pathLength

// Step 5: Format data for CSV
RETURN 
    disease1 AS Disease1,
    disease2 AS Disease2,
    pathLength AS PathLength,
    apoc.text.join(nodeNames, ' -> ') AS PathNodes,
    apoc.text.join(relTypes, ' -> ') AS RelationshipTypes,
    apoc.text.join(relSources, ' -> ') AS RelationshipSources
ORDER BY PathLength"""