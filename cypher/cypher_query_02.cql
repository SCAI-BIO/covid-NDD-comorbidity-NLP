# Query to count nodes
node_count_query = "MATCH (n) RETURN count(n) AS numNodes"
num_nodes = execute_query(node_count_query)
print(f"Number of Nodes: {num_nodes[0]['numNodes']}")

# Query to count edges
edge_count_query = "MATCH ()-[r]->() RETURN count(r) AS numEdges"
num_edges = execute_query(edge_count_query)
print(f"Number of Edges: {num_edges[0]['numEdges']}")