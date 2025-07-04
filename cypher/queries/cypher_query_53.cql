""" 
MATCH (d1:Disease)-[r]->(g:Gene)<-[r2]-(d2:Disease)
WHERE (
toLower(d1.name) CONTAINS 'covid'
AND (
toLower(d2.name) CONTAINS 'alzheimer' OR
toLower(d2.name) CONTAINS 'parkinson' OR
toLower(d2.name) CONTAINS 'neurodegenerative'
)
)
OR (
toLower(d2.name) CONTAINS 'covid'
AND (
toLower(d1.name) CONTAINS 'alzheimer' OR
toLower(d1.name) CONTAINS 'parkinson' OR
toLower(d1.name) CONTAINS 'neurodegenerative'
)
)
AND d1 <> d2 // Ensure that d1 and d2 are different nodes
RETURN
d1.name AS disease1,
TYPE(r) AS relationship1,
r.source as source_1,
g.name AS gene,
TYPE(r2) AS relationship2,
r2.source as source2,
d2.name AS disease2
"""