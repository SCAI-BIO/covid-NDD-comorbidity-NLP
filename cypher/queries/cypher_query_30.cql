"""  
WITH ['alzheimer', 'parkinson', 'neurodegenerative', 'covid', 'sars', "cov", 'neurodegen', 'inflamamtion'] AS keywords
MATCH (d)
WHERE any(keyword IN keywords WHERE toLower(d.name) CONTAINS toLower(keyword))
WITH d
MATCH (d1)-[r*..1]-(d2)
WHERE id(d1) < id(d2)
AND d1.name IS NOT NULL 
AND d2.name IS NOT NULL
AND all(rel IN r WHERE toLower(rel.source) CONTAINS 'scai')
AND all(rel IN r WHERE rel.pmid IS NOT NULL)
AND all(rel IN r WHERE rel.evidence IS NOT NULL)
AND any(rel IN r WHERE 
    toLower(rel.evidence) CONTAINS 'covid' OR 
    toLower(rel.evidence) CONTAINS 'viral infection' OR
    toLower(rel.evidence) CONTAINS 'sars cov' OR
    toLower(rel.evidence) CONTAINS 'neuro')  // Filter for evidence containing relevant terms
UNWIND r AS rel
RETURN DISTINCT
   d1.name AS Subject,
   d2.name AS Object,
   type(rel) AS Relation,
   rel.pmid AS PMID,
   rel.evidence AS Evidence,
   rel.source AS Source

"""