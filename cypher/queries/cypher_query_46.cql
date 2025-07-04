""" Only triples about COVID AD PD and NDD

WITH ["covid", "sars", "alzheimer", "parkinson", "neuro", "dementia"] AS keywords
MATCH (n)-[r]->(m)
WHERE ANY(keyword IN keywords WHERE toLower(n.name) CONTAINS keyword)
   OR ANY(keyword IN keywords WHERE toLower(m.name) CONTAINS keyword)
RETURN n.name AS Subject, type(r) AS Relation, m.name AS Object


"""


#improve above and extend keyword search (old not in paper)

""" 
WITH [
    "covid", "sars", "coronavirus", "post-covid", "long covid", "SARS-CoV-2",
    "encephalopathy", "myopathy", "brain fog", "fatigue", "hypoxia", "cytokine storm",
    "stroke", "guillain-barré", "neuroinflammation", "viral infection", "ARDS",
    "pasc", "neurodegeneration", "neurodegenerative", "alzheimer", "parkinson",
    "dementia", "cognitive decline", "memory loss", "ALS", "amyotrophic lateral sclerosis",
    "huntington", "multiple sclerosis", "lewy body dementia", "frontotemporal dementia",
    "PSP", "progressive supranuclear palsy", "CBD", "corticobasal degeneration",
    "CJD", "creutzfeldt-jakob", "motor neuron disease", "ataxia", "prion disease",
    "chorea", "myoclonus", "viral-induced neurodegeneration", "neurological decline",
    "neurotropism", "neuroinflammatory response", "axonal degeneration"
] AS keywords
MATCH (n)-[r]-(m)
WHERE ANY(keyword IN keywords WHERE toLower(n.name) CONTAINS keyword)
   OR ANY(keyword IN keywords WHERE toLower(m.name) CONTAINS keyword)
RETURN n.name AS Subject, type(r) AS Relation, m.name AS Object


"""

#get covid-NDD triples only direct interaction: final in paper
"""  

WITH [
   // Core COVID-19 terms
   "SARS-CoV-2", "COVID-19", "2019-nCoV", "coronavirus", "SARS", 
   "post-COVID", "long COVID", "PASC",
   
   // COVID-19 molecular & mechanism
   "ACE2", "TMPRSS2", "spike protein", "viral pneumonia",
   "cytokine storm", "viral infection", "ARDS",
   "respiratory infection", "hypoxia",
   
   // Neurological manifestations of COVID
   "brain fog", "fatigue", "encephalopathy",
   "guillain-barré", "stroke", "myopathy",
   "neuroinflammation", "neurological decline",
   "neuroinflammatory response", "neurotropism",
   
   // Core neurodegenerative diseases
   "Alzheimer", "Parkinson", "Huntington",
   "ALS", "amyotrophic lateral sclerosis",
   "multiple sclerosis", "prion disease",
   "dementia", "neurodegeneration", "neurodegenerative",
   
   // Specific neurodegenerative conditions
   "Lewy body dementia", "frontotemporal dementia",
   "PSP", "progressive supranuclear palsy",
   "CBD", "corticobasal degeneration",
   "CJD", "Creutzfeldt-Jakob",
   "motor neuron disease",
   
   // Symptoms & manifestations
   "cognitive decline", "memory loss", "ataxia",
   "chorea", "myoclonus", "axonal degeneration",
   
   // Molecular mechanisms
   "tauopathy", "synucleinopathy", "proteinopathy"
] AS keywords

// First match potential nodes
MATCH (n)
WHERE ANY(keyword IN keywords WHERE toLower(n.name) CONTAINS toLower(keyword))

// Pass the matched nodes
WITH COLLECT(DISTINCT n) as nodes, keywords

// Then expand to relationships with matched nodes
MATCH (n)-[r]-(m)
WHERE n IN nodes
AND ANY(keyword IN keywords WHERE toLower(m.name) CONTAINS toLower(keyword))

WITH 
   COUNT(DISTINCT n) + COUNT(DISTINCT m) as total_nodes,
   COUNT(DISTINCT r) as total_edges,
   COLLECT(DISTINCT n) + COLLECT(DISTINCT m) as all_nodes,
   COLLECT(DISTINCT r) as all_edges

RETURN 
   total_nodes as TotalNodes,
   total_edges as TotalEdges,
   CASE 
       WHEN total_nodes <= 1 THEN 0
       ELSE toFloat(total_edges) / (total_nodes * (total_nodes - 1))
   END as GraphDensity
"""
#inderct interaction included as well to get covid-NDD tripels from primekg

"""  
WITH [
    // Core COVID-19 terms - reduced to most specific
    "SARS-CoV-2", "COVID-19", "coronavirus", "SARS", 
    
    // Core neurodegenerative diseases - reduced to most specific
    "Alzheimer", "Parkinson", "Huntington",
    "ALS", "multiple sclerosis", "dementia",
    "neurodegeneration"
] AS keywords

// First find nodes matching keywords
MATCH (n)
WHERE ANY(keyword IN keywords WHERE toLower(n.name) CONTAINS toLower(keyword))

// Collect these as starting points
WITH COLLECT(DISTINCT n) as starting_nodes, keywords

// Find direct and one-hop connections
MATCH path = (n)-[r]-(intermediate)-[r2]-(m)
WHERE 
    n IN starting_nodes
    AND m IN starting_nodes
    AND n <> m  // Ensure we don't match same node

RETURN 
    COUNT(DISTINCT n) + COUNT(DISTINCT m) as TotalNodes,
    COUNT(DISTINCT r) + COUNT(DISTINCT r2) as TotalEdges,
    CASE 
        WHEN COUNT(DISTINCT n) + COUNT(DISTINCT m) <= 1 THEN 0
        ELSE toFloat(COUNT(DISTINCT r) + COUNT(DISTINCT r2)) / ((COUNT(DISTINCT n) + COUNT(DISTINCT m)) * (COUNT(DISTINCT n) + COUNT(DISTINCT m) - 1))
    END as GraphDensity
LIMIT 1000

"""