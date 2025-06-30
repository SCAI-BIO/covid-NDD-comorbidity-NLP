#Find paths that have an occurence of phenotpyes (not fuzzy)
"""
WITH ["Memory Impairment", "memory loss", "memory deficit",
      "Psychomotor Deterioration", "motor skills", "motor impairment",
      "Cognitive Epileptic Aura", "cognitive aura", "seizure disturbance",
      "Dysdiadochokinesis", "alternating movement", "motor coordination",
      "Status Epilepticus", "seizure paralysis", "motor weakness",
      "Sensorineural Hearing Impairment", "hearing loss", "deafness",
      "Gaze Palsy", "eye movement", "gaze palsy",
      "Leukoencephalopathy", "white matter", "brain disease",
      "Daytime Somnolence", "drowsiness", "sleepiness",
      "Diabetes Insipidus", "ADH deficiency", "hormone disorder",
      "Sleep-Wake Rhythm", "sleep cycle", "circadian disturbance",
      "Motor Neuron Dysfunction", "spasticity", "hyperreflexia",
      "Memory Impairment", "past memory loss", "memory recall",
      "Sensory Seizure", "smell hallucinations", "odor perception",
      "CSF Pyridoxal", "Vitamin B6", "CSF B6 levels",
      "Demyelination", "nerve damage", "myelin sheath",
      "Gait Ataxia", "unsteady walk", "balance",
      "Cognitive Seizure", "memory issues", "focal seizure",
      "Neurofibrillary Tangles", "tau protein", "brain tangles",
      "Neuroinflammation", "brain inflammation", "CNS inflammation"
     ] AS keywords,
     ["disgenet", "opentarget", "drugbank", "indra"] AS sources
 
MATCH p=(n)-[r]-(m) 
WHERE ANY(source IN sources WHERE toLower(r.source) CONTAINS toLower(source)) 
  AND ANY(keyword IN keywords WHERE toLower(n.name) CONTAINS toLower(keyword) OR m.name CONTAINS toLower(keyword))
RETURN p
LIMIT 50

"""

#save above as list of nodeds and edges
""" 
WITH ["Memory Impairment", "memory loss", "memory deficit",
      "Psychomotor Deterioration", "motor skills", "motor impairment",
      "Cognitive Epileptic Aura", "cognitive aura", "seizure disturbance",
      "Dysdiadochokinesis", "alternating movement", "motor coordination",
      "Status Epilepticus", "seizure paralysis", "motor weakness",
      "Sensorineural Hearing Impairment", "hearing loss", "deafness",
      "Gaze Palsy", "eye movement", "gaze palsy",
      "Leukoencephalopathy", "white matter", "brain disease",
      "Daytime Somnolence", "drowsiness", "sleepiness",
      "Diabetes Insipidus", "ADH deficiency", "hormone disorder",
      "Sleep-Wake Rhythm", "sleep cycle", "circadian disturbance",
      "Motor Neuron Dysfunction", "spasticity", "hyperreflexia",
      "Memory Impairment", "past memory loss", "memory recall",
      "Sensory Seizure", "smell hallucinations", "odor perception",
      "CSF Pyridoxal", "Vitamin B6", "CSF B6 levels",
      "Demyelination", "nerve damage", "myelin sheath",
      "Gait Ataxia", "unsteady walk", "balance",
      "Cognitive Seizure", "memory issues", "focal seizure",
      "Neurofibrillary Tangles", "tau protein", "brain tangles",
      "Neuroinflammation", "brain inflammation", "CNS inflammation"
     ] AS keywords,
     ["disgenet", "opentarget", "drugbank", "indra"] AS sources

MATCH p=(n)-[r]-(m) 
WHERE ANY(source IN sources WHERE toLower(r.source) CONTAINS toLower(source)) 
  AND ANY(keyword IN keywords WHERE toLower(n.name) CONTAINS toLower(keyword) OR toLower(m.name) CONTAINS toLower(keyword))
WITH p, 
     [node IN nodes(p) | node.name] AS node_names,
     [rel IN relationships(p) | type(rel)] AS relationship_types,
     [rel IN relationships(p) | rel.source] AS relationship_sources,
     length(p) AS path_length
RETURN node_names, relationship_types, relationship_sources, path_length
LIMIT 50
"""

#above with fuzzy matching

"""  

WITH ["Memory Impairment", "memory loss", "memory deficit",
      "Psychomotor Deterioration", "motor skills", "motor impairment",
      "Cognitive Epileptic Aura", "cognitive aura", "seizure disturbance",
      "Dysdiadochokinesis", "alternating movement", "motor coordination",
      "Status Epilepticus", "seizure paralysis", "motor weakness",
      "Sensorineural Hearing Impairment", "hearing loss", "deafness",
      "Gaze Palsy", "eye movement", "gaze palsy",
      "Leukoencephalopathy", "white matter", "brain disease",
      "Daytime Somnolence", "drowsiness", "sleepiness",
      "Diabetes Insipidus", "ADH deficiency", "hormone disorder",
      "Sleep-Wake Rhythm", "sleep cycle", "circadian disturbance",
      "Motor Neuron Dysfunction", "spasticity", "hyperreflexia",
      "Memory Impairment", "past memory loss", "memory recall",
      "Sensory Seizure", "smell hallucinations", "odor perception",
      "CSF Pyridoxal", "Vitamin B6", "CSF B6 levels",
      "Demyelination", "nerve damage", "myelin sheath",
      "Gait Ataxia", "unsteady walk", "balance",
      "Cognitive Seizure", "memory issues", "focal seizure",
      "Neurofibrillary Tangles", "tau protein", "brain tangles",
      "Neuroinflammation", "brain inflammation", "CNS inflammation"
     ] AS keywords,
     ["disgenet", "opentarget", "drugbank", "indra"] AS sources
 
MATCH p=(n)-[r]-(m)
WHERE ANY(source IN sources WHERE apoc.text.levenshteinDistance(toLower(r.source), toLower(source)) <= 2) 
  AND ANY(keyword IN keywords WHERE apoc.text.levenshteinDistance(toLower(n.name), toLower(keyword)) <= 2 
      OR apoc.text.levenshteinDistance(toLower(m.name), toLower(keyword)) <= 2)
RETURN p
LIMIT 50

"""