#above save data in csv
import pandas as pd
from neo4j import GraphDatabase

# Define connection parameters
uri = "bolt://localhost:7687"  # Update with your Neo4j Bolt URL
user = "neo4j"  # Replace with your Neo4j username
password = "12345678"  # Replace with your Neo4j password

# Create a Neo4j driver instance
driver = GraphDatabase.driver(uri, auth=(user, password))

# Queries with directed relationships (for sources containing 'sherpa')
query_directed_unique_nodes = """
MATCH (n)-[r]->()
WHERE toLower(r.source) CONTAINS toLower($source)
RETURN COUNT(DISTINCT n) AS uniqueNodeCount
"""

query_directed_unique_triples = """
MATCH (n1)-[r]->(n2)
WHERE toLower(r.source) CONTAINS toLower($source)
RETURN COUNT(DISTINCT id(n1) + id(r) + id(n2)) AS uniqueTripleCount
"""

# Queries with undirected relationships (for all other sources)
query_undirected_unique_nodes = """
MATCH (n)-[r]-()
WHERE toLower(r.source) CONTAINS toLower($source)
RETURN COUNT(DISTINCT n) AS uniqueNodeCount
"""

query_undirected_unique_triples = """
MATCH (n1)-[r]-(n2)
WHERE toLower(r.source) CONTAINS toLower($source)
RETURN COUNT(DISTINCT id(n1) + id(r) + id(n2)) AS uniqueTripleCount
"""

# List of sources to query for
sources = ['sherpa', 'opentargets', 'disgenet', 'indra', 'drugbank', 'kegg', 'cbm', 'scai','pubtator']
print("Note!!!!!For CBM and SCAI graphs and SHERPA, first choose where r.filePath is cbm or scai or sherpa, set r.source = ...")
# Function to execute the queries for each source
def get_stats_for_source(source):
    with driver.session() as session:
        if "sherpa" in source.lower():
            # Use directed relationships for sources containing 'sherpa'
            result_nodes = session.run(query_directed_unique_nodes, source=source)
            unique_node_count = result_nodes.single()["uniqueNodeCount"]

            result_triples = session.run(query_directed_unique_triples, source=source)
            unique_triple_count = result_triples.single()["uniqueTripleCount"]
        else:
            # Use undirected relationships for other sources
            result_nodes = session.run(query_undirected_unique_nodes, source=source)
            unique_node_count = result_nodes.single()["uniqueNodeCount"]

            result_triples = session.run(query_undirected_unique_triples, source=source)
            unique_triple_count = result_triples.single()["uniqueTripleCount"]

        # Return the results
        return unique_node_count, unique_triple_count

# Initialize an empty list to store the results
results = []

# Loop over each source and collect the results in the list
for source in sources:
    unique_nodes, unique_triples = get_stats_for_source(source)
    results.append({"Source": source, "Unique Nodes": unique_nodes, "Unique Triples": unique_triples})

# Create a pandas DataFrame from the results
df = pd.DataFrame(results)

# Display the DataFrame
print(df)

# Optionally, save the DataFrame to a CSV file
df.to_csv(r"C:\Users\nbabaiha\Documents\GitHub\COMMUTE\commute\neo4j-import-analysis\neo4j-analysis-results\source_statistics_recheck.csv", index=False)

# Close the Neo4j driver connection
driver.close()