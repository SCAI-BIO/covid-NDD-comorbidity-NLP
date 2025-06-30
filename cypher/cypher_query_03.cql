# Community detection
### First RUN THIS IN NEO4J browser:CALL gds.graph.drop('myGraph')

###################

from neo4j import GraphDatabase
import pandas as pd
# Run this in broser CALL gds.graph.drop('myGraph')

# Define connection parameters
uri = "bolt://localhost:7687"  # Update if needed
user = "neo4j"
password = "12345678"

# Create a Neo4j driver instance
driver = GraphDatabase.driver(uri, auth=(user, password))

def execute_query(query, parameters=None):
    with driver.session() as session:
        result = session.run(query, parameters=parameters)
        return [record for record in result]

def close_driver():
    driver.close()

# Retrieve all node labels
node_labels_query = "CALL db.labels() YIELD label RETURN label"
node_labels = execute_query(node_labels_query)
node_labels_list = [record['label'] for record in node_labels]

# Retrieve all relationship types
relationship_types_query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
relationship_types = execute_query(relationship_types_query)
relationship_types_list = [record['relationshipType'] for record in relationship_types]

# Convert lists to Cypher format
node_labels_str = ', '.join([f"'{label}'" for label in node_labels_list])
relationship_types_str = ', '.join([f"'{rtype}'" for rtype in relationship_types_list])

# Create a graph projection for GDS algorithms
create_graph_query = f"""
CALL gds.graph.project('myGraph', [{node_labels_str}], [{relationship_types_str}])
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
"""
projection_result = execute_query(create_graph_query)
print("Graph projection details:")
print(pd.DataFrame(projection_result))

# Run Louvain community detection
louvain_query = """
CALL gds.louvain.stream('myGraph')
YIELD nodeId, communityId
RETURN nodeId, communityId
"""
louvain_results = execute_query(louvain_query)
louvain_df = pd.DataFrame(louvain_results, columns=['nodeId', 'communityId'])



# Fetch node names or other properties
def get_node_properties(node_ids):
    node_properties_query = """
    UNWIND $nodeIds AS nodeId
    MATCH (n) WHERE id(n) = nodeId
    RETURN id(n) AS nodeId, n.name AS nodeName
    """
    return execute_query(node_properties_query, parameters={"nodeIds": node_ids})
column_names = ["nodeId", "nodeName"]
# Get node properties for the detected nodes
node_ids = louvain_df['nodeId'].unique()  # Get unique node IDs
node_properties = get_node_properties(node_ids)
node_properties_df = pd.DataFrame(node_properties, columns = column_names)

# print("\nNode properties:")
# print(node_properties_df.head(10))  # Print first 10 rows
# print("Columns in node properties:", node_properties_df.columns)
# print("Node properties DataFrame info:")
# node_properties_df.info()

print(node_properties_df)
# Ensure nodeId is of the same type in both DataFrames
louvain_df['nodeId'] = louvain_df['nodeId'].astype(str)
node_properties_df['nodeId'] = node_properties_df['nodeId'].astype(str)

# Perform the merge with error handling
try:
    full_df = louvain_df.merge(node_properties_df, on='nodeId', how='left')
    print("Merge successful. Full DataFrame:")
    print(full_df.head(10))
    print(full_df.info())
except KeyError as e:
    print(f"KeyError encountered: {e}")
    print("louvain_df columns:", louvain_df.columns)
    print("node_properties_df columns:", node_properties_df.columns)
except Exception as e:
    print(f"An error occurred: {e}")