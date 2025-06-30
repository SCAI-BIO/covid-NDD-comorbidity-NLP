"""
MATCH (gene:Gene)-[r1]-(d1:Disease)
MATCH (gene)-[r2]-(d2:Disease)
WHERE 
    toLower(r1.source) CONTAINS "opentarget" 
    AND toLower(r2.source) CONTAINS "opentarget"
    AND (toLower(d1.name) CONTAINS "covid" OR toLower(d1.name) CONTAINS "covid-19")
    AND (
        toLower(d2.name) CONTAINS "alzheimer" 
        OR toLower(d2.name) CONTAINS "parkinson"
        OR toLower(d2.name) CONTAINS "neuro"
    )
RETURN DISTINCT 
    gene.symbol as Gene,
    d1.name as Covid_Related,
    d2.name as Neurological_Disease
ORDER BY Gene
""""""