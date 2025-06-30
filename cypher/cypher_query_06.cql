# Relationship label frequency (no source given)
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

def get_all_relationship_types():
    query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
    return execute_query(query)

def count_relationships_by_type(relationship_type):
    query = f"MATCH ()-[r:{relationship_type}]->() RETURN COUNT(r) AS frequency"
    result = execute_query(query)
    return result[0]['frequency'] if result else 0

# Retrieve all relationship types
relationship_types = get_all_relationship_types()
relationship_type_names = [record['relationshipType'] for record in relationship_types]

# Count relationships for each type
relationship_counts = {rtype: count_relationships_by_type(rtype) for rtype in relationship_type_names}

# Convert to DataFrame and sort by frequency in descending order
df = pd.DataFrame(list(relationship_counts.items()), columns=['RelationshipType', 'Frequency'])
df = df.sort_values(by='Frequency', ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(df['RelationshipType'], df['Frequency'], color='lightgreen')
plt.xlabel('Relationship Type')
plt.ylabel('Frequency')
plt.title('Relationship Types and Their Frequencies')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Show the plot
plt.show()

# Close the driver
driver.close()