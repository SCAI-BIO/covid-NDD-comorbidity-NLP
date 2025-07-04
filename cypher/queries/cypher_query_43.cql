"""
MATCH (disease:Disease)-[r]->(related)
WHERE toLower(disease.name) CONTAINS 'alzheimer' OR 
     toLower(disease.name)  CONTAINS 'covid' OR 
      toLower(disease.name)  CONTAINS 'parkinson' OR 
      toLower(disease.name)  CONTAINS 'neurodegeneration'
RETURN disease.name AS Disease, type(r) AS RelationshipType, related AS RelatedNode

"""

"""
#rethink above?
MATCH (disease:Disease)-[r]->(related)
WHERE (toLower(disease.name) CONTAINS 'alzheimer' OR 
       toLower(disease.name) CONTAINS 'covid' OR 
       toLower(disease.name) CONTAINS 'parkinson' OR 
       toLower(disease.name) CONTAINS 'neurodegeneration') OR
      (toLower(related.name) CONTAINS 'alzheimer' OR 
       toLower(related.name) CONTAINS 'covid' OR 
       toLower(related.name) CONTAINS 'parkinson' OR 
       toLower(related.name) CONTAINS 'neurodegeneration')
RETURN disease.name AS Disease, type(r) AS RelationshipType, related
"""