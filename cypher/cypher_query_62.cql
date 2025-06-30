###import DrugBank###
import pandas as pd
import os
from neo4j import GraphDatabase

# Function to detect the Neo4j node type (not used in this case but kept for flexibility)
def detect_neo4j_node_type(my_dict):
    node_type = "Entity"
    if "MESH" in my_dict:
        node_type = "disease"
    if "CHEBI" in my_dict:
        node_type = "chemical"
    if "HGNC" in my_dict:
        node_type = "gene"
    return node_type

# Neo4j connection details
uri = "bolt://localhost:7687"
user = "neo4j"
password = "12345678"
driver = GraphDatabase.driver(uri, auth=(user, password))

# Function to upload triples to Neo4j with a source tag
def upload_triples_neo4j(triple, disease_name):
    print(f"Uploading triples for disease: {disease_name}")
    with driver.session() as session:
        try:
            # Extract values from the triple
            subj = disease_name  # The disease name is the subject
            subj_type = "Disease"  # Type of the subject node is Disease
            obj = triple['Drug Name']  # The drug is the object
            obj_type = "Drug"  # Type of the object node is Drug
            rel_type = "ASSOCIATION"  # Type of the relationship
            drug_id = triple["Primary ID"]  # Drug ID from DrugBank
            pmid = int(triple["PubMed ID"])  # PubMed ID
            
            # Define the source tag for DrugBank
            source = "DrugBank + " + triple.get('Source', 'Unknown')

            # Create nodes and relationships, including the source and pmid
            cypher_query = f"""
            MERGE (a:{subj_type} {{name: $subj}})
            MERGE (b:{obj_type} {{name: $obj, id: $drug_id}})
            MERGE (a)-[r:{rel_type} {{pmid: $pmid, source: $source}}]->(b)
            """

            session.run(cypher_query, subj=subj, obj=obj, drug_id=drug_id, pmid=pmid, source=source)

        except Exception as e:
            print(f"Cannot import row for disease {disease_name}: {e}")

# Close the driver
driver.close()

if __name__ == "__main__":
    directory = "data/DRUGBANK/csv_output"
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            disease_name = filename.split(".csv")[0]  # Extract disease name from filename
            # Load the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            # Process each row in the DataFrame
            for index, row in df.iterrows():
                upload_triples_neo4j(row, disease_name)