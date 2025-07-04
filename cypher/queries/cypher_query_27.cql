"""  
WITH ['alzheimer', 'parkinson', 'neurodegenerative', 'covid', 'neurodegeneration'] AS keywords,
     ['disease', 'susceptibility', 'familial', 'with', 'to', 'and', 'type', 'number'] AS commonWords // Common words to ignore

// Step 1: Match nodes and clean up names by removing punctuation and common words
MATCH (d)
WHERE any(keyword IN keywords WHERE toLower(d.name) CONTAINS keyword)
WITH d, 
     apoc.text.regreplace(toLower(d.name), "[^a-zA-Z0-9\\s]", "") AS cleanName, // Remove punctuation and convert to lowercase
     commonWords

// Step 2: Remove common words like 'disease' and numbers
WITH d, trim(reduce(name = cleanName, word IN commonWords | apoc.text.replace(name, word, ''))) AS finalName // Remove common words

// Step 3: Group nodes by the cleaned and normalized name
WITH finalName, collect(d) AS similar_nodes

// Step 4: Generate unique pairs of grouped nodes
WITH collect({name: finalName, nodes: similar_nodes}) AS grouped_nodes
UNWIND range(0, size(grouped_nodes)-2) AS i
UNWIND range(i+1, size(grouped_nodes)-1) AS j
WITH grouped_nodes[i] AS group1, grouped_nodes[j] AS group2

// Step 5: Compute shortest path with filtered relationships
UNWIND group1.nodes AS n1
UNWIND group2.nodes AS n2
MATCH path = shortestPath((n1)-[r*..3]-(n2)) // Find shortest path within 3 hops
WHERE all(rel IN relationships(path) WHERE 
    any(source_keyword IN ['opentargets', 'disgenet', 'drugbank', 'indra'] 
        WHERE toLower(rel.source) CONTAINS source_keyword)) // Check if source contains any of the target keywords

// Step 6: Filter for paths where one node is related to COVID
WITH path, nodes(path) AS pathNodes, relationships(path) AS pathRels
WHERE any(node IN pathNodes WHERE toLower(node.name) CONTAINS 'covid') // Ensure one node contains 'covid'

// Step 7: Extract node and relationship information for the paths
WITH 
    [node IN pathNodes | COALESCE(node.name, node.symbol, toString(id(node)))] AS nodeNames, // Fallback: name -> symbol -> id
    [rel IN pathRels | type(rel)] AS relTypes,
    [rel IN pathRels | rel.source] AS relSources,
    length(path) AS pathLength

// Step 8: Return the lists of nodes, relationships, path length, and sources
RETURN 
    nodeNames AS Nodes, 
    relTypes AS Relationships, 
    pathLength AS PathLength,
    relSources AS RelationshipSources
ORDER BY pathLength ASC // Sort by path length in ascending order
//LIMIT 200

"""