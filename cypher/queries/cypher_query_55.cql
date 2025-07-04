#above but branch up till reach covid node (for PRIMEKG especially)
""" 
WITH [
    "epithelial", "immune cell types", "alveolar macrophages",
    "Monocytes", "myeloid cells recruited to tissues",
    "Neuronal cells", "immune cells",
    "Microglia", "astrocytes", "oligodendrocytes",
    "Microglia and hippocampal synapses", "inflammatory responses",
    "Glial cells", "microglia and astrocytes in brain inflammation",
    "Astrocytes", "microglia in cognitive impairment pathways",
    "Pyramidal neurons", "CA1 hippocampal region", "stress",
    "Microglia activation", "synapse loss",
    "inflammatory responses", "IL6", "IL1B", "TNF",
    "interferon responses",
    "CDC25C", "PLK1",
    "APOE", "TREM2", "TLR4 activation",
    "Genetic variants in TLR4","TLR4", "brain inflammation",
    "CDC25C", "PLK1",
    "Inflammatory gene responses", "cognitive decline",
    "Transcriptional changes",
    "TREM2", "APOE",
    "TLR4 genetic variant", "neuroinflammation",
    "Cytokine storms", "megakaryocytes", "monocytes",
    "Chemokine signaling", "cellular stress",
    "Neuroinflammatory processes", "cytokine","oxidative stress",
    "APP metabolism", "lipid metabolism", "TLR4",
    "TLR4-mediated", "neuroinflammation", "synaptic pruning", "activated microglia",
    "Neuroinflammation", "immune activation", "cognitive decline",
    "Persistent neuroinflammation", "blood-brain barrier disruption",
    "hippocampal sclerosis",
    "lipid metabolism", "protein aggregation", "TLR4 activation",
    "Microglial phagocytosis of synapses"
] AS keywords,
     ["scai"] AS sources
 
MATCH initial_path=(n)-[r]-(m)
WHERE ANY(keyword IN keywords WHERE apoc.text.levenshteinDistance(toLower(n.name), toLower(keyword)) <= 2 
      OR apoc.text.levenshteinDistance(toLower(m.name), toLower(keyword)) <= 2)
WITH DISTINCT nodes(initial_path) AS initial_nodes

// Expand and find the shortest path from each initial node to a COVID-related node
UNWIND initial_nodes AS start_node
MATCH sp=shortestPath((start_node)-[*]-(covid_node))
WHERE apoc.text.levenshteinDistance(toLower(covid_node.name), "covid") <= 2
RETURN 
    [node IN nodes(sp) | node] AS path_nodes,
    [rel IN relationships(sp) | rel] AS path_relationships,
    length(sp) AS path_length,
    covid_node.name AS covid_node_name
LIMIT 100
"""