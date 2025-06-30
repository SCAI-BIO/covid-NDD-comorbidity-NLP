#Use cell type biological process info as well
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
     ["sherpa"] AS sources
 
MATCH p=(n)-[r]-(m) 
WHERE ANY(source IN sources WHERE toLower(r.source) CONTAINS toLower(source)) 
  AND ANY(keyword IN keywords WHERE toLower(n.name) CONTAINS toLower(keyword) OR toLower(m.name) CONTAINS toLower(keyword))
RETURN p
limit 50

"""

#to get list of pahs
"""
WITH p, 
     [node IN nodes(p) | node.name] AS node_names,
     [rel IN relationships(p) | type(rel)] AS relationship_types,
     [rel IN relationships(p) | rel.source] AS relationship_sources,
     length(p) AS path_length
RETURN node_names, relationship_types, relationship_sources, path_length
""" 


#Above but using fuzzy matching (USED FOR PAPER!!!!!)
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
     ["sherpa"] AS sources
 
MATCH p=(n)-[r]-(m)
WHERE ANY(source IN sources WHERE apoc.text.levenshteinDistance(toLower(r.source), toLower(source)) <= 2)
  AND ANY(keyword IN keywords WHERE apoc.text.levenshteinDistance(toLower(n.name), toLower(keyword)) <= 2 
      OR apoc.text.levenshteinDistance(toLower(m.name), toLower(keyword)) <= 2)
RETURN p
LIMIT 50

""" 

#to get list of pahs
"""
WITH p, 
     [node IN nodes(p) | node.name] AS node_names,
     [rel IN relationships(p) | type(rel)] AS relationship_types,
     [rel IN relationships(p) | rel.source] AS relationship_sources,
     length(p) AS path_length
RETURN node_names, relationship_types, relationship_sources, path_length
""" 

#above expand paths to more than length one (not used for paper took much time, mainly for PrimeKG intended first!!)

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
     ["cbm"] AS sources

MATCH p=(n)-[r*1..5]-(m)
WHERE ANY(rel IN r WHERE ANY(source IN sources WHERE apoc.text.levenshteinDistance(toLower(rel.source), toLower(source)) <= 2))
  AND ANY(keyword IN keywords WHERE apoc.text.levenshteinDistance(toLower(n.name), toLower(keyword)) <= 2 
      OR apoc.text.levenshteinDistance(toLower(m.name), toLower(keyword)) <= 2)
RETURN 
    [node IN nodes(p) | node] AS nodes,
    [rel IN relationships(p) | rel] AS relationships,
    length(p) AS path_length
LIMIT 50

"""