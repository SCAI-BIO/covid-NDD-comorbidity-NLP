"""MATCH (d1:Disease)-[r]->(g)<-[r2]-(d2:Disease)
WHERE (d1.name CONTAINS 'COVID' OR d1.name CONTAINS 'Alzheimer' OR d1.name CONTAINS 'Parkinson' OR d1.name CONTAINS 'Neurodegenerative')
  AND (d2.name CONTAINS 'COVID' OR d2.name CONTAINS 'Alzheimer' OR d2.name CONTAINS 'Parkinson' OR d2.name CONTAINS 'Neurodegenerative')
RETURN d1 AS disease1, r AS relationship1, g AS gene, r2 AS relationship2, d2 AS disease2"""

###to delete everything in batch
"""CALL apoc.periodic.iterate(
  'MATCH (n) RETURN n',
  'DETACH DELETE n',
  {batchSize: 1000}
)"""