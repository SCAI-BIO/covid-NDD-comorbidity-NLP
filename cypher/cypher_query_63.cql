## Import Pubtator results ###
import pandas as pd
from neo4j import GraphDatabase

# Replace these with your actual connection details
uri = "bolt://localhost:7687"
user = "neo4j"
password = "12345678"

driver = GraphDatabase.driver(uri, auth=(user, password))

def validate_type(node_type):
    """Validates the node type, returning 'undefined' if the type is an integer or None."""
    if isinstance(node_type, int) or node_type is None:
        return "undefined"
    return node_type

def upload_triples_neo4j(triple):
    """Uploads a triple to Neo4j."""
    print(f"Uploading triple: {triple['node1_text']} -[{triple['relation_type']}]-> {triple['node2_text']}")
    with driver.session() as session:
        try:
            subj = triple['node1_text']
            subj_type = validate_type(triple['role1_type'])
            obj = triple['node2_text']
            obj_type = validate_type(triple['role2_type'])
            rel_type = triple['relation_type'].upper()
            pmid = int(triple['pmid'])
            score = float(triple['score'])
            evidence = triple['evidence']

            # Combine PubTator with additional source information (if available)
            source = "PubTator + " + triple.get('source', 'Unknown')

            # Create nodes and relationship with additional properties including source
            cypher_query = f"""
            MERGE (a:{subj_type} {{name: $subj}})
            MERGE (b:{obj_type} {{name: $obj}})
            MERGE (a)-[r:{rel_type} {{pmid: $pmid, score: $score, evidence: $evidence, source: $source}}]->(b)
            """

            session.run(cypher_query, subj=subj, obj=obj, pmid=pmid, score=score, evidence=evidence, source=source)
        except Exception as e:
            print("Cannot import this row to Neo4j:", e)

# Close the driver connection
driver.close()

if __name__ == "__main__":
    # Load the Excel file into a DataFrame (replace with your actual file path)
    excel_file_path = "data/PubTator/pmc_triples_100.xlsx"  # Replace with the actual path

    df = pd.read_excel(excel_file_path)

    # Process each row in the DataFrame and upload to Neo4j
    for index, row in df.iterrows():
        upload_triples_neo4j(row)