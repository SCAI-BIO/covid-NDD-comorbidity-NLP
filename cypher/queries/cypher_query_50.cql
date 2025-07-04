"""  
MATCH (n)-[r]->(m)
WHERE toLower(r.source) CONTAINS "sherpa" 
  AND (
    toLower(n.name) CONTAINS "covid" OR 
    toLower(n.name) CONTAINS "sars-cov-2" OR
    toLower(n.name) CONTAINS "neuroinflammation" OR
    toLower(n.name) CONTAINS "oxidative stress" OR
    toLower(n.name) CONTAINS "immune response" OR
    toLower(n.name) CONTAINS "viral protein aggregation" OR
    toLower(n.name) CONTAINS "hypoxia" OR
    toLower(n.name) CONTAINS "amyloidogenesis" OR
    toLower(n.name) CONTAINS "neurotropism" OR
    toLower(n.name) CONTAINS "mitochondrial dysfunction" OR
    toLower(n.name) CONTAINS "cognitive decline" OR
    toLower(n.name) CONTAINS "parkinsonism" OR
    toLower(n.name) CONTAINS "fatigue" OR
    toLower(n.name) CONTAINS "mood disorders" OR
    toLower(n.name) CONTAINS "anosmia" OR
    toLower(n.name) CONTAINS "ageusia" OR
    toLower(n.name) CONTAINS "tau" OR
    toLower(n.name) CONTAINS "ace2" OR
    toLower(n.name) CONTAINS "snca" OR
    toLower(n.name) CONTAINS "lrk2" OR
    toLower(n.name) CONTAINS "amyloid beta" OR
    toLower(n.name) CONTAINS "glial cell activation" OR
    toLower(n.name) CONTAINS "cognitive dysfunction" OR
    toLower(n.name) CONTAINS "neurodegeneration" OR
    toLower(n.name) CONTAINS "synaptic dysfunction" OR
    toLower(n.name) CONTAINS "neurovascular dysfunction" OR
    toLower(n.name) CONTAINS "il-6" OR
    toLower(n.name) CONTAINS "tnf-α" OR
    toLower(n.name) CONTAINS "autophagy impairment" OR
    toLower(n.name) CONTAINS "motor symptoms"
  )
RETURN n.name as Start_Node, type(r) as REL_type, m.name as End_Node, r.evidence as Evidence, r.pmid as PMID

"""