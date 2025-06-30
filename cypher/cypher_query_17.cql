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
RETURN DISTINCT path, length(path) AS pathLength

"""