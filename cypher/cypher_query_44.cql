#cypher queris to get common rels between two graphS:
#covid and ad
cypher_disgenet = """
MATCH (d1:Disease)-[r]->(g)<-[r2]-(d2:Disease)
WHERE d1.name CONTAINS 'COVID' AND d2.name CONTAINS 'Alzheimer'
RETURN d1 AS disease1, r AS relationship1, g AS gene, r2 AS relationship2, d2 AS disease2
"""
#covid and all diseases
"""
MATCH (d1:Disease)-[r]->(g)<-[r2]-(d2:Disease)
WHERE (d1.name CONTAINS 'COVID' OR d1.name CONTAINS 'Alzheimer' OR d1.name CONTAINS 'Parkinson' OR d1.name CONTAINS 'neurodegenerative')
  AND (d2.name CONTAINS 'COVID' OR d2.name CONTAINS 'Alzheimer' OR d2.name CONTAINS 'Parkinson' OR d2.name CONTAINS 'neurodegenerative')
  AND d1 <> d2  // Ensure that d1 and d2 are different nodes
RETURN d1 AS disease1, r AS relationship1, g AS gene, r2 AS relationship2, d2 AS disease2

"""

#cypher filter common genes based on score
"""MATCH (d1:Disease)-[r]->(g)<-[r2]-(d2:Disease)
WHERE (d1.name CONTAINS 'COVID' OR d1.name CONTAINS 'Alzheimer' OR d1.name CONTAINS 'Parkinson' OR d1.name CONTAINS 'neurodegenerative')
  AND (d2.name CONTAINS 'COVID' OR d2.name CONTAINS 'Alzheimer' OR d2.name CONTAINS 'Parkinson' OR d2.name CONTAINS 'neurodegenerative')
  AND d1 <> d2  // Ensure that d1 and d2 are different nodes
  AND r.score is not null  // Ensure the score property exists
  AND r.score > 0.75 // Filter by association score
RETURN d1 AS disease1, r AS relationship1, g AS gene, r2 AS relationship2, d2 AS disease2, r.score AS score
"""

cypher = """

MATCH (d1:Disease {name: 'Covid-19'})-[r1:RELATED_TO]->(g:Gene)<-[r2:RELATED_TO]-(d2:Disease {name: "Alzheimer's disease"})
RETURN d1, r1, g, r2, d2
"""

#get common rels between four disease
cypher = """MATCH (d1:Disease {name: 'Covid-19'})-[r1:RELATED_TO]->(g:Gene)
WHERE (d1)-[:RELATED_TO]->(g)
MATCH (d2:Disease {name: "Alzheimer's disease"})-[r2:RELATED_TO]->(g)
WHERE (d2)-[:RELATED_TO]->(g)
MATCH (d3:Disease {name: 'Parkinson Disease'})-[r3:RELATED_TO]->(g)
WHERE (d3)-[:RELATED_TO]->(g)
MATCH (d4:Disease {name: 'Neurodegenerative Diseases'})-[r4:RELATED_TO]->(g)
WHERE (d4)-[:RELATED_TO]->(g)
RETURN d1,d2,d3,d4,g,
       r1, r2, r3, r4"""

"""
MATCH (d:Disease)-[r]->(g:Gene)
WHERE d.name CONTAINS 'COVID'
RETURN count(distinct(g))
"""