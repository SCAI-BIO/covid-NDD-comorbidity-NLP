### get triples by source
"""

MATCH (a)-[r]->(b)
WHERE r.source STARTS WITH 'iTextMine'
RETURN a.name AS Subject, type(r) AS Predicate, b.name AS Object, r.source AS Source, r.evidence AS Evidence

"""


##CBM or Sherpa triples

"""

MATCH (a)-[r]->(b)
WHERE r.filePath CONTAINS 'CBM'
RETURN a.name AS Subject, type(r) AS Predicate, b.name AS Object, r.filePath AS FilePath, r.evidence AS Evidence


"""

#get triples by frequency
"""
MATCH (a)-[r]->(b) where b.name is not NULL 
RETURN a.name AS Subject, type(r) AS Predicate, b.name AS Object, COUNT(*) AS Frequency
ORDER BY Frequency DESC
"""

# common nodes between KEGG and Sherpa
"""MATCH (n1)-[r1]->(m1), (n2)-[r2]->(m2)
WHERE apoc.text.distance(n1.name, n2.name) < 7
  AND r1.source = 'KEGG'
  AND "sherpa" in r2.annotationDatasource 
RETURN DISTINCT n1.name AS Common_Node_KEGG, n2.name AS Common_Node_Sherpa"""