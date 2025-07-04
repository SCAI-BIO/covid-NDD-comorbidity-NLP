# Centrality measure (various types) and bar charts for each source
from neo4j import GraphDatabase
import pandas as pd
import matplotlib.pyplot as plt

# Define connection parameters
uri = "bolt://localhost:7687"
user = "neo4j"
password = "12345678"

# Create a Neo4j driver instance
driver = GraphDatabase.driver(uri, auth=(user, password))

def execute_query(query, parameters=None):
    with driver.session() as session:
        result = session.run(query, parameters=parameters)
        return [record for record in result]

# Function to check if a graph exists
def graph_exists(graph_name):
    query = f"CALL gds.graph.exists('{graph_name}') YIELD exists"
    result = execute_query(query)
    return result[0]['exists'] if result else False

# Function to drop the graph projection if it exists
def drop_graph_if_exists(graph_name):
    if graph_exists(graph_name):
        query = f"CALL gds.graph.drop('{graph_name}', false)"
        execute_query(query)

# Function to create a graph projection filtered by source using Cypher
def create_graph_projection(graph_name, source):
    # Drop the graph if it already exists
    drop_graph_if_exists(graph_name)
    
    query = f"""
    CALL gds.graph.project.cypher(
        '{graph_name}',
        'MATCH (n) RETURN id(n) AS id',
        'MATCH ()-[r]-() WHERE toLower(r.source) contains toLower("{source}") RETURN id(r) AS id, id(startNode(r)) AS source, id(endNode(r)) AS target'
    )
    """
    execute_query(query)

# Function to compute centrality
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

# Define sources
sources = ['kegg', 'sherpa', 'opentargets', 'disgenet', 'indra', 'drugbank','pubtator', 'cbm', 'scai']

# Loop through each source and compute centrality measures
for source in sources:
    graph_name = f"graph_{source}"
    
    # Create a filtered graph projection for the current source using Cypher
    create_graph_projection(graph_name, source)
    
    results = {}
    for centrality_type in centrality_types:
        # Compute centrality for the current source and type
        data = compute_centrality(graph_name, centrality_type)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Rename columns based on actual data
        df.columns = ['node_id', 'centrality_score']
        
        # Sort by centrality_score in descending order
        df.sort_values(by='centrality_score', ascending=False, inplace=True)
        
        # Select top 10 nodes
        df_top10 = df.head(10)
        
        # Store the DataFrame for the current centrality type
        results[centrality_type] = df_top10

    # Plot centrality for the current source
    def plot_centrality(df, centrality_type, source):
        # Filter out rows where 'node_id' is None
        df = df[df['node_id'].notnull()]

        # Convert 'node_id' to string type (if needed)
        df['node_id'] = df['node_id'].astype(str)

        plt.figure(figsize=(12, 8))
        plt.bar(df['node_id'], df['centrality_score'], color='skyblue')
        plt.xlabel('Node Name')
        plt.ylabel(f'{centrality_type.capitalize()} Centrality')
        plt.title(f'Top 10 {centrality_type.capitalize()} Centrality for {source.capitalize()}')
        plt.xticks(rotation=45, ha='right')  # Adjust rotation for better readability
        plt.tight_layout()
        plt.show()

    # Plot top 10 centrality measures for each type
    for centrality_type in centrality_types:
        df = results[centrality_type]
        plot_centrality(df, centrality_type, source)

# Close the driver connection
driver.close()