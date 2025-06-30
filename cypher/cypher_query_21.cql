""""
WITH ['alzheimer', 'parkinson', 'neurodegenerative', 'covid', 'neurodegeneration'] AS keywords

// Step 1: Find disease-related nodes
MATCH (d)
WHERE any(keyword IN keywords WHERE toLower(d.name) CONTAINS keyword)
WITH collect(d) AS disease_nodes

// Step 2: Generate unique pairs of disease nodes
UNWIND disease_nodes AS d1
UNWIND disease_nodes AS d2
WITH d1, d2
WHERE id(d1) < id(d2)  // Ensure unique pairs and avoid self-matches

// Step 3: Compute shortest path with filtered relationships
MATCH path = shortestPath((d1)-[r*..3]-(d2)) // Find shortest path within 3 hops
WHERE all(rel IN relationships(path) WHERE 
    any(source_keyword IN ['opentargets', 'disgenet', 'drugbank', 'indra'] 
        WHERE toLower(rel.source) CONTAINS source_keyword)) // Check if source contains any of the target keywords

// Step 4: Return paths for visualization
RETURN path
LIMIT 200  // Adjust this limit based on your visualization needs and graph size
"""