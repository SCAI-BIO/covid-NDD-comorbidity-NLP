#COMMON GENE nodes between KEGG and sherpa
from neo4j import GraphDatabase
from matplotlib_venn import venn2
import matplotlib.pyplot as plt

# Neo4j connection details
uri = "bolt://localhost:7687"  # Update this with your Neo4j connection URI
user = "neo4j"  # Update this with your Neo4j username
password = "12345678"  # Update this with your Neo4j password

# Connect to Neo4j
driver = GraphDatabase.driver(uri, auth=(user, password))

# Function to execute a query and return a single result
def query_neo4j(query):
    with driver.session() as session:
        result = session.run(query)
        return result.single()[0]

# Query to get Sherpa node count (only Gene nodes)
sherpa_count_query = """
MATCH (n)
WHERE 'sherpa' IN n.source and n.namespace = "HGNC"
RETURN COUNT(DISTINCT n.name) AS Sherpa_Node_Count
"""

# Query to get KEGG node count (only Gene nodes)
kegg_count_query = """
MATCH (n:Gene)
WHERE n.source = 'KEGG'
RETURN COUNT(DISTINCT n.name) AS KEGG_Node_Count
"""

# Query to get intersection count (only Gene nodes)
intersection_query = """
MATCH (n)
WHERE 'sherpa' IN n.source and n.namespace = 'HGNC'
WITH COLLECT(DISTINCT n.name) AS sherpa_genes
MATCH (m:Gene)
WHERE m.source = 'KEGG'
WITH sherpa_genes, COLLECT(DISTINCT m.name) AS kegg_genes
RETURN SIZE(apoc.coll.intersection(sherpa_genes, kegg_genes)) AS Intersection_Count
"""

# Execute queries
sherpa_count = query_neo4j(sherpa_count_query)
kegg_count = query_neo4j(kegg_count_query)
intersection_count = query_neo4j(intersection_query)

# Create a Venn diagram
plt.figure(figsize=(8, 6))
venn2(subsets=(sherpa_count - intersection_count, kegg_count - intersection_count, intersection_count),
      set_labels=('Sherpa', 'KEGG'))

# Set the title
plt.title('Venn Diagram of Sherpa and KEGG Gene Node Counts with Intersection')

# Display the Venn diagram
plt.show()

# Close the Neo4j driver connection
driver.close()