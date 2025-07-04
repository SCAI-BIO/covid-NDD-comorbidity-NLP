# Centrality measure (various types) and bar charts (no data source given)
from neo4j import GraphDatabase
import pandas as pd
import matplotlib.pyplot as plt

# Define connection parameters
uri = "bolt://localhost:7687"
user = "neo4j"
password = "12345678"

# Create a Neo4j driver instance
driver = GraphDatabase.driver(uri, auth=(user, password))

def execute_query(query):
    with driver.session() as session:
        result = session.run(query)
        return [record for record in result]

def compute_centrality(graph_name, centrality_type):
    query = f"""
    CALL gds.{centrality_type}.stream('{graph_name}')
    YIELD nodeId, score
    RETURN gds.util.asNode(nodeId).name AS nodeName, score AS centrality
    ORDER BY centrality DESC;
    """
    return execute_query(query)

# Define centrality types
centrality_types = ['degree', 'closeness', 'betweenness', 'eigenvector']

# Compute and store centrality measures
results = {}
for centrality_type in centrality_types:
    data = compute_centrality('myGraph', centrality_type)
    df = pd.DataFrame(data)
    
    # Print the columns to check their names
    print(f"Original columns for {centrality_type}: {df.columns.tolist()}")
    
    # Rename columns
    df.rename(columns={0: 'node_id', 1: 'centrality_score'}, inplace=True)
    
    # Sort by centrality_score in descending order
    df.sort_values(by='centrality_score', ascending=False, inplace=True)
    
    # Select top 5
    df_top10 = df.head(10)
    
    # Store the DataFrame
    results[centrality_type] = df_top10

# Close the driver
driver.close()

# Define a function to plot centrality
def plot_centrality(df, centrality_type):
    plt.figure(figsize=(12, 8))
    plt.bar(df['node_id'], df['centrality_score'], color='skyblue')
    plt.xlabel('Node Name')
    plt.ylabel(f'{centrality_type.capitalize()} Centrality')
    plt.title(f'Top 10 {centrality_type.capitalize()} Centrality of Nodes')
    plt.xticks(rotation=45, ha='right')  # Adjust rotation for better readability
    plt.tight_layout()
    plt.show()

# Plot top 5 centrality type
for centrality_type in centrality_types:
    df = results[centrality_type]
    plot_centrality(df, centrality_type)