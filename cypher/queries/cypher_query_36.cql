#autoamte code above: get paths of covid ndd comorbidity for a graph
from neo4j import GraphDatabase
import json
# Connect to your Neo4j database
uri = "bolt://localhost:7687"  # Change this to your Neo4j URI
username = "neo4j"  # Change this to your username
password = "12345678"  # Change this to your password

driver = GraphDatabase.driver(uri, auth=(username, password))

def get_paths_as_list():
    with driver.session() as session:
        # Cypher query to find paths
        cypher_query = """
        WITH ['alzheimer', 'parkinson', 'neurodegenerative', 'covid', 'neurodegeneration'] AS keywords
        MATCH (d)
        WHERE any(keyword IN keywords WHERE toLower(d.name) CONTAINS keyword)
        WITH collect(d) AS diseases
        
        UNWIND diseases AS d1
        UNWIND diseases AS d2
        WITH d1, d2 WHERE id(d1) < id(d2)
        
        MATCH path = shortestPath((d1)-[*..3]-(d2))
        WITH DISTINCT [node IN nodes(path) | id(node)] AS pathNodes, 
                      [rel IN relationships(path) | type(rel)] AS pathRelationships
        
        UNWIND pathNodes AS nodeId
        MATCH (n)
        WHERE id(n) = nodeId
        WITH pathNodes, pathRelationships, n, 
             CASE WHEN 'name' IN keys(n) THEN n.name ELSE n.symbol END AS nodeName
        
        WITH DISTINCT pathNodes, pathRelationships, collect(nodeName) AS nodeNames
        
        RETURN nodeNames AS pathNodes, pathRelationships
        """
        
        # Execute the query and fetch the results
        results = session.run(cypher_query)
        
        # Process the results into a list of lists
        path_data = []
        for record in results:
            path_nodes = record["pathNodes"]  # List of node names
            path_relationships = record["pathRelationships"]  # List of relationship types
            path_data.append([path_nodes, path_relationships])
        
        return path_data
        
def save_results_to_file(data, filename):
    with open(filename, 'w') as file:
        # Write the JSON data to the file
        json.dump(data, file, indent=4)
# Fetch the paths
neo4j_results = get_paths_as_list()
# Save the results to a file
save_results_to_file(neo4j_results, 'neo4j_results.json')
# Fetch the paths and print them
print(len(neo4j_results))

# Close the driver connection
driver.close()