#general statistics
"""  
#unique nodes Sherpa


MATCH (n)-[r]->()
WHERE toLower(r.source) = toLower('sherpa')
RETURN COUNT(DISTINCT n) AS uniqueNodeCount;


#unique triples

first run:

:param source => 'sherpa'

then:

MATCH (n1)-[r]->(n2)
WHERE toLower(r.source) = toLower($source)
RETURN COUNT(DISTINCT id(n1) + id(r) + id(n2)) AS uniqueTripleCount;

"""