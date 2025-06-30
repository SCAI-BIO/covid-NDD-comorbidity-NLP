#Draw bar chart for 10 top genes by node degree  

from neo4j import GraphDatabase
import pandas as pd
import matplotlib.pyplot as plt

# Neo4j connection details
uri = "bolt://localhost:7687"  # Update this with your Neo4j connection URI
user = "neo4j"  # Update this with your Neo4j username
password = "12345678"  # Update this with your Neo4j password

# Connect to Neo4j
driver = GraphDatabase.driver(uri, auth=(user, password))

# Cypher query to retrieve the genes and their degree
query = """
MATCH (g)-[r]->()
WHERE g.namespace = "HGNC" AND "sherpa" IN r.annotationDatasource
RETURN DISTINCT g.name AS GeneName, COUNT(r) AS Degree
ORDER BY Degree DESC
"""

# query = """MATCH (g:Gene)-[r]->()
# WHERE r.source = "KEGG"
# RETURN DISTINCT g.name AS GeneName, COUNT(r) AS Degree
# ORDER BY Degree DESC
# """

# Function to execute query and return data as DataFrame
def query_neo4j(query):
    with driver.session() as session:
        result = session.run(query)
        records = result.data()
        # Convert result to a pandas DataFrame
        df = pd.DataFrame(records)
    return df

# Execute the query and get the result as a DataFrame
df = query_neo4j(query)

# Sort the DataFrame and get the top 10 genes by degree
df_top10 = df.head(10)

# Plotting the bar chart
plt.figure(figsize=(10, 6))
plt.bar(df_top10['GeneName'], df_top10['Degree'], color='blue')

# Add labels and title
plt.xlabel('Gene Name')
plt.ylabel('Degree')
plt.title('Top 10 Genes by Degree (Source: Sherpa)')

# Rotate the x labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()

# Close Neo4j connection
driver.close()