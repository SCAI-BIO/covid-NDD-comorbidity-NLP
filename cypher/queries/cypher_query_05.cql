# Node label frequency for different sources
from neo4j import GraphDatabase
import pandas as pd
import matplotlib.pyplot as plt

# Define connection parameters
uri = "bolt://localhost:7687"  # Update if needed
user = "neo4j"
password = "12345678"  # Update with your password

# Create a Neo4j driver instance
driver = GraphDatabase.driver(uri, auth=(user, password))

def execute_query(query, parameters=None):
    with driver.session() as session:
        result = session.run(query, parameters=parameters)
        return [record for record in result]

def get_all_labels():
    query = "CALL db.labels() YIELD label RETURN label"
    return execute_query(query)

def count_nodes_by_label_and_source(label, source):
    query = f"""
    MATCH (n:{label})-[r]-()
    WHERE toLower(r.source) contains toLower($source)
    RETURN COUNT(n) AS frequency
    """
    result = execute_query(query, {"source": source})
    return result[0]['frequency'] if result else 0

# List of sources to search for
sources = ['sherpa', 'pubtator', 'cbm', 'scai', 'opentargets', 'disgenet', 'indra', 'drugbank','kegg']

# Retrieve all labels
labels = get_all_labels()
label_names = [record['label'] for record in labels]

# Loop through each source, calculate frequencies, and plot bar charts
for source in sources:
    # Count nodes for each label filtered by source
    label_counts = {label: count_nodes_by_label_and_source(label, source) for label in label_names}
    
    # Convert to DataFrame and sort by frequency in descending order
    df = pd.DataFrame(list(label_counts.items()), columns=['NodeType', 'Frequency'])
    df = df[df['Frequency'] > 0]  # Filter out labels with zero frequency
    df = df.sort_values(by='Frequency', ascending=False)
    
    if not df.empty:
        # Plotting for the current source
        plt.figure(figsize=(10, 6))
        plt.bar(df['NodeType'], df['Frequency'], color='skyblue')
        plt.xlabel('Node Type')
        plt.ylabel('Frequency')
        plt.title(f'Node Types and Their Frequencies for {source.capitalize()}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Show the plot
        plt.show()

# Close the driver
driver.close()