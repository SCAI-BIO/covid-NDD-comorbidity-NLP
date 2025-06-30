# visualize abode(either UNIQUE or COMMON paths)
from neo4j import GraphDatabase

# Connect to Neo4j
uri = "bolt://localhost:7687"  # Adjust if necessary
username = "neo4j"
password = "12345678"  # Adjust if necessary
driver = GraphDatabase.driver(uri, auth=(username, password))

def store_common_paths_in_neo4j(common_paths):
    with driver.session() as session:
        for path in common_paths:
            nodes, relationships = path  # Unpack nodes and relationships
            previous_node = None

            for idx, node in enumerate(nodes):
                # Merge nodes
                query = """
                MERGE (n:Node {name: $node_name})
                RETURN id(n) AS node_id
                """
                result = session.run(query, node_name=node)
                node_id = result.single()["node_id"]

                if previous_node is not None:
                    # Create the relationship with the previous node
                    relationship_type = relationships[idx - 1]
                    query = f"""
                    MATCH (a:Node), (b:Node)
                    WHERE id(a) = $prev_node_id AND id(b) = $curr_node_id
                    MERGE (a)-[r:{relationship_type}]->(b)
                    """
                    session.run(query, prev_node_id=previous_node, curr_node_id=node_id)
                
                previous_node = node_id  # Update previous_node to current node_id

# Example common paths (nodes and relationships)
# common_paths = [
#     [['alzheimer', 'gene1', 'parkinson'], ['rel1', 'rel2']],
#     [['neurodegeneration', 'gene2', 'covid'], ['rel3', 'rel4']]
# ]

store_common_paths_in_neo4j(unique_paths) #or feed common paths

# Close the driver connection
driver.close()