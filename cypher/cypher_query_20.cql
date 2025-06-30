"""  For exact match of tuples and same names
MATCH (n1)-[r1]->(m1), (n2)-[r2]->(m2)
WHERE n1.name = n2.name
  AND m1.name = m2.name
  AND r1.source <> r2.source
RETURN 
  n1.name AS Common_Node,
  type(r1) AS Relationship_Type_1, 
  type(r2) AS Relationship_Type_2,
  r1.source AS Source_1, 
  r2.source AS Source_2, 
  m1.name AS Related_Node
LIMIT 100

"""

#For fuzzy match:
"""" 
MATCH (n1)-[r1]->(m1), (n2)-[r2]->(m2)
WHERE apoc.text.levenshteinDistance(n1.name, n2.name) < 3
  AND apoc.text.levenshteinDistance(m1.name, m2.name) < 3
  AND r1.source <> r2.source
RETURN 
  n1.name AS Common_Node_1,
  n2.name AS Common_Node_2,
  type(r1) AS Relationship_Type_1, 
  type(r2) AS Relationship_Type_2,
  r1.source AS Source_1, 
  r2.source AS Source_2, 
  m1.name AS Related_Node_1,
  m2.name AS Related_Node_2
LIMIT 100

"""