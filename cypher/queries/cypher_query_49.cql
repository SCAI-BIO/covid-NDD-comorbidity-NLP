"""
WITH ["neuroinflammation", "oxidative stress", "mitochondrial dysfunction", "blood-brain barrier disruption", "amyloid", "tau pathology", "autonomic dysfunction", "neurotropism"] AS keywords
     
MATCH p=(n)-[*1..5]-(m) 
WHERE ANY(keyword IN keywords WHERE toLower(n.name) CONTAINS toLower(keyword) OR toLower(m.name) CONTAINS toLower(keyword))
RETURN p
LIMIT 50

"""
#efficient: mainly gives text minign triples

"""
WITH ["neuroinflammation", "inflammation", "neuroimmune response", "neuroimmunology", "brain inflammation",
      "oxidative stress", "free radicals", "reactive oxygen species", "ROS", "oxidative damage",
      "mitochondrial", "mitochondrial dysfunction", "mitochondrial damage", "mitochondrial health", "energy production",
      "blood-brain barrier", "brain barrier", "BBB", "neurovascular unit", "blood-brain permeability",
      "amyloid", "amyloid plaques", "beta-amyloid", "amyloid beta", "plaques in Alzheimer's",
      "tau", "tau protein", "tau tangles", "tauopathies", "tau neurofibrillary tangles",
      "autonomic dysfunction", "autonomic nervous system", "ANS", "autonomic regulation", "autonomic imbalance",
      "neurotropism", "neuronal targeting", "neurotropic", "neurotropic viruses", "neurotropic diseases"]AS keywords
UNWIND keywords AS keyword
WITH DISTINCT toLower(keyword) AS lower_keyword
MATCH (n)-[r*1..3]-(m)  // Depth limited to 1..3 for better performance
WHERE (toLower(n.name) CONTAINS lower_keyword OR toLower(m.name) CONTAINS lower_keyword)
WITH n, m
LIMIT 5000  // Limit the number of node pairs
MATCH p = shortestPath((n)-[*1..3]-(m))  // Use shortestPath to get only the shortest path
RETURN p


"""
# get path nodes

"""  
  
WITH ["neuroinflammation", "inflammation", "neuroimmune response", "neuroimmunology", "brain inflammation",
      "oxidative stress", "free radicals", "reactive oxygen species", "ROS", "oxidative damage",
      "mitochondrial", "mitochondrial dysfunction", "mitochondrial damage", "mitochondrial health", "energy production",
      "blood-brain barrier", "brain barrier", "BBB", "neurovascular unit", "blood-brain permeability",
      "amyloid", "amyloid plaques", "beta-amyloid", "amyloid beta", "plaques in Alzheimer's",
      "tau", "tau protein", "tau tangles", "tauopathies", "tau neurofibrillary tangles",
      "autonomic dysfunction", "autonomic nervous system", "ANS", "autonomic regulation", "autonomic imbalance",
      "neurotropism", "neuronal targeting", "neurotropic", "neurotropic viruses", "neurotropic diseases"] AS keywords
UNWIND keywords AS keyword
WITH DISTINCT toLower(keyword) AS lower_keyword
MATCH (n)-[r*1..3]-(m)  // Set the relationship depth to 1..3 for better performance
WHERE (toLower(n.name) CONTAINS lower_keyword OR toLower(m.name) CONTAINS lower_keyword)
WITH n, m
LIMIT 5000  // Limit the number of node pairs considered
MATCH p = shortestPath((n)-[*1..3]-(m))  // Use shortestPath to find the shortest path
WITH n, m, p, relationships(p) AS relTypes, length(p) AS pathLength
UNWIND relTypes AS rel  // Unwind relationships to extract the types and properties of relationships
WITH n, m, rel, pathLength,
     CASE 
        WHEN rel.source IS NOT NULL AND rel.source <> '' THEN rel.source 
        ELSE null 
     END AS validSource  // Only include source if it is not null or empty
WHERE validSource IS NOT NULL  // Filter out paths where source is null or empty
RETURN 
    n.name AS `Start Node`,  // Start Node (name of the first node in the path)
    type(rel) AS `REL_type`,  // Relationship type (for each relationship in the path)
    m.name AS `End Node`,
    rel.pmid AS PMID,  // End Node (name of the second node in the path)
    rel.evidence AS `Evidences`,  // Evidence (assuming Evidence is a property of the relationship)
    validSource AS `Source`,  // Only include Source if it is not null or empty
    pathLength  // Path Length (optional, shows the length of the path)
ORDER BY pathLength  // Sort by path length
LIMIT 500  // Limit the number of paths returned

"""
##add cbm triple as well
"""WITH [
    // Neuroinflammatory mechanisms
    "neuroinflammation", "inflammation", "neuroimmune response", "neuroimmunology", "brain inflammation",
    "cytokine storm", "inflammatory mediators", "glial activation", "microglial activation",
    
    // Oxidative stress pathway
    "oxidative stress", "free radicals", "reactive oxygen species", "ROS", "oxidative damage",
    "antioxidant response", "redox signaling", "lipid peroxidation",
    
    // Mitochondrial dysfunction
    "mitochondrial", "mitochondrial dysfunction", "mitochondrial damage", "mitochondrial health",
    "energy production", "ATP synthesis", "electron transport chain",
    
    // Blood-brain barrier disruption
    "blood-brain barrier", "brain barrier", "BBB", "neurovascular unit", "blood-brain permeability",
    "tight junction proteins", "endothelial dysfunction",
    
    // Protein aggregation
    "amyloid", "amyloid plaques", "beta-amyloid", "amyloid beta", "protein aggregation",
    "protein misfolding", "proteostasis", "protein degradation",
    
    // Tau pathology
    "tau", "tau protein", "tau tangles", "tauopathies", "tau phosphorylation",
    "microtubule dysfunction", "axonal transport",
    
    // Autonomic dysfunction
    "autonomic dysfunction", "autonomic nervous system", "ANS", "autonomic regulation",
    "sympathetic activation", "parasympathetic dysfunction",
    
    // Neurotropism
    "neurotropism", "neuronal targeting", "neurotropic", "viral neuroinvasion",
    "neural spread", "synaptic dysfunction"
] AS keywords

// Process keywords once
UNWIND keywords AS keyword
WITH DISTINCT toLower(keyword) AS lower_keyword
WITH collect(lower_keyword) AS processed_keywords

// Main query with optimizations
MATCH (n)
WHERE any(keyword IN processed_keywords WHERE toLower(n.name) CONTAINS keyword)
MATCH (m)
WHERE id(m) > id(n) AND 
      any(keyword IN processed_keywords WHERE toLower(m.name) CONTAINS keyword)
WITH n, m
CALL apoc.path.expandConfig(n, {
    minLevel: 1,
    maxLevel: 3,
    uniqueness: "NODE_PATH",
    targetNode: m,
    limit: 1
}) YIELD path

WITH n, m, path,
     [rel in relationships(path) | rel] AS rels,
     length(path) AS pathLength
UNWIND rels AS rel

WITH n, m, rel, pathLength,
     CASE 
        WHEN rel.source IS NOT NULL AND trim(rel.source) <> '' 
        THEN rel.source 
        ELSE null 
     END AS validSource,
     CASE
        WHEN rel.evidence IS NOT NULL AND trim(rel.evidence) <> ''
        THEN rel.evidence
        ELSE 'No evidence provided'
     END AS evidence
WHERE validSource IS NOT NULL

RETURN DISTINCT
    n.name AS `Start Node`,
    type(rel) AS `Relationship`,
    rel.pmid as PMID,
    m.name AS `End Node`,
    evidence AS `Evidence`,
    validSource AS `Source`,
    pathLength AS `Path Length`,
    CASE
        WHEN n.type IS NOT NULL THEN n.type
        ELSE 'Unknown'
    END AS `Start Node Type`,
    CASE
        WHEN m.type IS NOT NULL THEN m.type
        ELSE 'Unknown'
    END AS `End Node Type`
ORDER BY pathLength, `Start Node`
LIMIT 1000
"""