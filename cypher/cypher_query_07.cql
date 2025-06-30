# Relationship label frequency by source
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

def count_relationships_by_type_and_source(relationship_type, source):
    query = f"""
    MATCH ()-[r:{relationship_type}]->()
    WHERE toLower(r.source) contains toLower($source)
    RETURN COUNT(r) AS frequency
    """
    result = execute_query(query, {"source": source})
    return result[0]['frequency'] if result else 0

# List of sources to search for
sources = ['sherpa', 'pubtator', 'cbm', 'scai', 'opentargets', 'disgenet', 'indra', 'drugbank','kegg']

# Retrieve all relationship types
relationship_types = get_all_relationship_types()
relationship_type_names = [record['relationshipType'] for record in relationship_types]

# Loop through each source, calculate frequencies, and plot bar charts
for source in sources:
    # Count relationships for each type filtered by source
    relationship_counts = {rtype: count_relationships_by_type_and_source(rtype, source) for rtype in relationship_type_names}
    
    # Convert to DataFrame and sort by frequency in descending order
    df = pd.DataFrame(list(relationship_counts.items()), columns=['RelationshipType', 'Frequency'])
    df = df[df['Frequency'] > 0]  # Filter out relationship types with zero frequency
    df = df.sort_values(by='Frequency', ascending=False)
    
    if not df.empty:
        # Plotting for the current source
        plt.figure(figsize=(10, 6))
        plt.bar(df['RelationshipType'], df['Frequency'], color='green')
        plt.xlabel('Relationship Type')
        plt.ylabel('Frequency')
        plt.title(f'Relationship Types and Their Frequencies for {source.capitalize()}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Show the plot
        plt.show()

# Close the driver
driver.close()