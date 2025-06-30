###INDRA import to Neo4j
import pandas as pd
import ast
import os
from neo4j import GraphDatabase

# Function to detect the type of node (disease, chemical, or gene)
# Function to detect the type of node (disease, chemical, gene, pathway, or text)
def detect_neo4j_node_type(my_dict):
    node_type = "Entity"
    
    # Check for specific namespaces first
    if "MESH" in my_dict or "DOID" in my_dict:
        node_type = "Disease/BiologicalProcess"
    elif "CHEBI" in my_dict:
        node_type = "Chemical"
    elif "HGNC" in my_dict:
        node_type = "Gene/Protein"
    elif "KEGG" in my_dict:
        node_type = "Pathway"
    elif "GO" in my_dict:
        node_type = "Gene Ontology Term"
    elif "IPR" in my_dict:
        node_type = "Protein/Enzyme"  # Adding IPR to handle protein/enzyme identifiers
    elif "ECCODE" in my_dict:
        node_type = "Enzyme Code"  # Specific case for enzyme codes
        
    # If none of the identifiers match, check for TEXT (but only if no other identifier is found)
    if "TEXT" in my_dict and node_type == "Entity":
        node_type = "GeneralEntitiy"
        
    return node_type


# Neo4j connection details
uri = "bolt://localhost:7687"
user = "neo4j"
password = "12345678"
driver = GraphDatabase.driver(uri, auth=(user, password))

# Function to upload triples to Neo4j with a source tag
def upload_triples_neo4j(triple, disease_name):
    print("Uploading triples in Neo4j")
    with driver.session() as session:
        try:
            subj = triple['subj']
            obj = triple['obj']
            rel_type = triple['type']
            subj_type = detect_neo4j_node_type(triple["subj_namespace"])
            obj_type = detect_neo4j_node_type(triple["obj_namespace"])

            # Process PMIDs handling single integers, lists, and NaN
            pmids = []
            pmids_string_list = triple['pmids']

            if pd.notna(pmids_string_list):
                if isinstance(pmids_string_list, (int, float)):
                    pmids = [int(pmids_string_list)]
                elif isinstance(pmids_string_list, str):
                    if pmids_string_list.startswith("[") and pmids_string_list.endswith("]"):
                        pmids = ast.literal_eval(pmids_string_list)
                        pmids = [int(float(pmid)) for pmid in pmids if isinstance(pmid, (int, float))]
                    else:
                        pmids = [int(float(pmids_string_list))]

            evid = triple["evid_sentence"] if triple["evid_sentence"] else "NoEvidence"
            source = "INDRA + " + triple.get('source', 'Unknown')

            cypher_query = f"""
            MERGE (a:{subj_type} {{name: $subj}})
            MERGE (b:{obj_type} {{name: $obj}})
            MERGE (a)-[r:{rel_type} {{pmids: $pmids, disease_name: $disease_name, evid_sentence: $evid_sentence, source: $source}}]->(b)
            """
            session.run(cypher_query, subj=subj, obj=obj, pmids=pmids, disease_name=disease_name, evid_sentence=evid, source=source)

        except Exception as e:
            print("Cannot import this row to Neo4j", e)

# Close the driver
driver.close()

if __name__ == "__main__":
    directory = "data/INDRA/data"
    api_key = "f2be320e-22f7-471a-b457-326a3ebb5a84"
    
    for filename in os.listdir(directory):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(directory, filename)
            disease_name = filename.split(".xlsx")[0]
            # Load the Excel file into a DataFrame
            df = pd.read_excel(file_path)
            
            # Data cleaning: Drop rows where 'subj', 'obj', 'pmids' are missing, and filter 'belief' > 0.85
            df = df.dropna(subset=['subj', 'obj', 'pmids', 'score (belief)'])
            df = df[df['pmids'].astype(bool)]  # Further filter out any rows where pmids is empty
            df = df[df['score (belief)'] > 0.85]       # Filter rows where belief is > 0.85

            # Process each row in the cleaned DataFrame
            for index, row in df.iterrows():
                upload_triples_neo4j(row, disease_name)