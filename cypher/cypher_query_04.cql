# Node label frequency (no source given)
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

def count_nodes_by_label(label):
    query = f"MATCH (n:{label}) RETURN COUNT(n) AS frequency"
    result = execute_query(query)
    return result[0]['frequency'] if result else 0

# Retrieve all labels
labels = get_all_labels()
label_names = [record['label'] for record in labels]

# Count nodes for each label
label_counts = {label: count_nodes_by_label(label) for label in label_names}

# Convert to DataFrame and sort by frequency in descending order
df = pd.DataFrame(list(label_counts.items()), columns=['NodeType', 'Frequency'])
df = df.sort_values(by='Frequency', ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(df['NodeType'], df['Frequency'], color='skyblue')
plt.xlabel('Node Type')
plt.ylabel('Frequency')
plt.title('Node Types and Their Frequencies')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Show the plot
plt.show()

# Close the driver
driver.close()