#Sherpa upload
import os
from bel_json_importer.n4j_meta import Neo4jClient
from bel_json_importer.n4j_bel import Neo4jBel
paths = []
for path, _, files in os.walk("data/Sherpa"): #substitute it with "data" to laod covid and NDD and sherpa triples only
    for file in files:
        print(file)
        if file.endswith(".json"):
            print(path)
            paths.append(os.path.join(path, file))
neo = Neo4jClient(
    uri="bolt://localhost:7687", database="neo4j", user="neo4j", password="12345678"
)
#Add all three graphs covid ad pd and comorbidity
n4jbel = Neo4jBel(client=neo)
for path in paths:
    n4jbel.import_json(input_path=path, update_from_protein2gene=False) #Maria added True

print("Done")

#remember to add this fr convininece:

'match(n)-[r]->(m) where "sherpa" in r.annotationDatasource set r.source = "sherpa"'