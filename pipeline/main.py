"""
Main Pipeline Script for COVID-NDD Co-Morbidity KG Analysis
This script performs:
1. File integrity hashing
2. Knowledge graph triple import into Neo4j
3. Graph querying and phenotype/pathway analysis
"""

import hashlib

def sha256_hash(file_path):
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def verify_input_hashes():
    files = [
        "drugbank_v5.1.12.csv",
        "opentargets_v24.06.tsv",
        "disgenet_v24.2.json",
        "indra_statements.json",
        "primekg_2023.csv"
    ]
    print("Step 1: Hashing files for reproducibility")
    for f in files:
        try:
            h = sha256_hash(f)
            print(f"{f}: {h}")
        except FileNotFoundError:
            print(f"File not found: {f}")

# ----------------------------
# Step 2: Import Triples into Neo4j
# ----------------------------
def import_knowledge_graphs():
    print("\nStep 2: Importing knowledge graphs into Neo4j")
    # Imported from: import-neo4j-all-dbs.py
#!/usr/bin/env python
# coding: utf-8

# In[11]:


import requests
import json
import time
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pandas as pd
from py2neo import Graph, Node, Relationship
import math
import os


# In[ ]:


###DISGENET######
# Insert graph to Neo4j
import os
import pandas as pd
import math
from neo4j import GraphDatabase

# Neo4j connection details
uri = "bolt://localhost:7687"
user = "neo4j"
password = "12345678"
graph = GraphDatabase.driver(uri, auth=(user, password))

def merge_node_disease(label, disease_name, disease_id, source):
    merge_query = f"""
    MERGE (n:{label} {{name: $name, id: $id}})
    ON CREATE SET n.source = $source
    RETURN n
    """
    with graph.session() as session:
        return session.run(merge_query, name=disease_name, id=disease_id, source=source).single()

def merge_node_gene(label, gene_symbol, gene_ens_id, source):
    merge_query = f"""
    MERGE (n:{label} {{symbol: $symbol, id: $id}})
    ON CREATE SET n.source = $source
    RETURN n
    """
    with graph.session() as session:
        return session.run(merge_query, symbol=gene_symbol, id=gene_ens_id, source=source).single()

def merge_relationship(disease_id, gene_id, association_type, pmid, source, score):
    properties = []
    params = {'disease_id': disease_id, 'gene_id': gene_id}

    if pmid is not None:
        properties.append("pmid: $pmid")
        params['pmid'] = pmid
    if source is not None:
        properties.append("source: $source")
        params['source'] = source
    if score is not None:
        properties.append("score: $score")
        params['score'] = score

    properties_str = ", ".join(properties)

    relationship_query = f"""
    MATCH (d:Disease {{id: $disease_id}})
    MATCH (g:Gene {{id: $gene_id}})
    MERGE (d)-[r:{association_type} {{{properties_str}}}]->(g)
    RETURN r
    """
    with graph.session() as session:
        return session.run(relationship_query, **params).single()

# Read CSV file
directory = "data/DISGENET/api-call-results"
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        # Load the CSV file into a dataframe
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)

        # Iterate through the DataFrame rows
        for idx, row in df.iterrows():
            try:
                # Extract values and handle NaNs
                gene_symbol = row.get("gene_symbol", "").strip()
                gene_id = row.get("gene_id", "")
                disease_name = row.get("disease_name", "").strip()
                disease_id = row.get("disease_id", "")
                
                # Concatenate DisGeNET with the source from the row
                source_row = row.get("source", "").strip()
                source = f"DisGeNET + {source_row}"

                score = row.get("score", None)
                pmid = row.get("pmid", None)

                # Convert pmid to int if it's not NaN
                if isinstance(pmid, float) and math.isnan(pmid):
                    pmid = None
                elif isinstance(pmid, float):
                    pmid = None
                else:
                    try:
                        pmid = int(pmid) if pmid is not None else None
                    except (ValueError, TypeError):
                        pmid = None

                # Check if gene and disease identifiers are not empty
                if not gene_symbol or not gene_id or not disease_name or not disease_id:
                    print(f"Skipping row {idx} due to missing data")
                    continue

                # Merge or create nodes with the source tag
                merge_node_disease("Disease", disease_name, disease_id, source)
                merge_node_gene("Gene", gene_symbol, gene_id, source)

                # Create relationship between disease and gene with the source tag
                merge_relationship(disease_id, gene_id, "ASSOCIATION", pmid, source, score)
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                print(filename)


# In[ ]:


# DISEASE DB using one full data: NOT USED IN PAPER
import os
import pandas as pd
import json
from neo4j import GraphDatabase
from concurrent.futures import ThreadPoolExecutor, as_completed

# Neo4j connection details
uri = "bolt://localhost:7687"
user = "neo4j"
password = "12345678"

# Load the checkpoint file if it exists
checkpoint_file = "checkpoint.json"
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as file:
        checkpoint = json.load(file)
else:
    checkpoint = {}

def process_batch(session, batch):
    query = """
    UNWIND $batch AS row
    MERGE (d:Disease {id: row.disease_id, name: row.disease_name})
    MERGE (g:Gene {id: row.gene_id, symbol: row.gene_symbol})
    MERGE (d)-[r:ASSOCIATION {source: row.source}]->(g)
    SET r.score = row.score
    """
    print(f"Processing batch of {len(batch)} records")
    session.run(query, batch=batch)

def process_file(file_path):
    batch_size = 1000
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session() as session:
            # Read the first row to determine the number of columns
            first_row = pd.read_csv(file_path, sep="\t", header=None, nrows=1)

            # Check if the file has a "source" column or only 5 columns
            if first_row.shape[1] == 5:
                # No "source" column, manually add it
                column_names = ["gene_id", "gene_symbol", "disease_id", "disease_name", "score"]
            elif first_row.shape[1] == 6:
                # The "source" column is present
                column_names = ["gene_id", "gene_symbol", "disease_id", "disease_name", "score", "source"]
            else:
                raise ValueError(f"Unexpected number of columns: {first_row.shape[1]} in {file_path}")

            # Read CSV file in chunks
            for chunk in pd.read_csv(file_path, sep="\t", header=None, chunksize=batch_size):
                chunk.columns = column_names

                # If the source column doesn't exist, add a default source
                if "source" not in chunk.columns:
                    chunk["source"] = "DiseaseDB"  # Default source if missing

                # If source exists, combine it with "DiseaseDB"
                chunk['source'] = "DiseaseDB + " + chunk['source'].fillna("").astype(str)
                
                # Convert the chunk to a batch and process it
                batch = chunk.to_dict('records')
                process_batch(session, batch)

def main():
    directory = "data/Disease-db"
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {executor.submit(process_file, os.path.join(directory, filename)): filename 
                          for filename in os.listdir(directory) if filename.endswith(".tsv")}
        
        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                future.result()
            except Exception as exc:
                print(f'{filename} generated an exception: {exc}')
            else:
                print(f'{filename} processing completed')

if __name__ == "__main__":
    main()


# In[ ]:


# OpenTargets import to Neo4j
import os
import sys
import pandas as pd
import re
from neo4j import GraphDatabase

# Adjust the sys.path to include the parent directory if needed
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import custom module after adjusting sys.path
import opentarget_disease_filter
#import filtered_diseases

# Neo4j connection details
uri = "bolt://localhost:7687"
user = "neo4j"
password = "12345678"
graph = GraphDatabase.driver(uri, auth=(user, password))

def merge_node_disease(label, disease_name, disease_id, source):
    merge_query = f"""
    MERGE (n:{label} {{name: $name, id: $id}})
    ON CREATE SET n.source = $source
    RETURN n
    """
    with graph.session() as session:
        return session.run(merge_query, name=disease_name, id=disease_id, source=source).single()

def merge_node_target(label, gene_symbol, source):
    merge_query = f"""
    MERGE (n:{label} {{symbol: $symbol}})
    ON CREATE SET n.source = $source
    RETURN n
    """
    with graph.session() as session:
        return session.run(merge_query, symbol=gene_symbol, source=source).single()

def merge_relationship(disease_name, gene_symbol, association_type, score, source):
    properties = ["source: $source"]
    params = {'disease_name': disease_name, 'gene_symbol': gene_symbol, 'source': source}
    
    if score is not None:
        properties.append("score: $score")
        params['score'] = score

    properties_str = ", ".join(properties)
    
    relationship_query = f"""
    MATCH (d:Disease {{name: $disease_name}})
    MATCH (g:Gene {{symbol: $gene_symbol}})
    MERGE (d)-[r:{association_type} {{{properties_str}}}]->(g)
    RETURN r
    """
    with graph.session() as session:
        return session.run(relationship_query, **params).single()

directory = "data/OpenTargets/data"
disease_names = opentarget_disease_filter.disease_names_filter

for filename in os.listdir(directory):
    if filename.endswith(".tsv"):
        file_path = os.path.join(directory, filename)
        # Load the TSV file into a dataframe
        df = pd.read_csv(file_path, sep="\t", header=0)
        
        for idx, row in df.iterrows():
            try:
                # Extract values and handle NaNs
                gene_symbol = row.get("symbol", "")
                score = row.get("globalScore", None)
                
                # Extract disease_id from the filename and map it to disease_name
                match = re.search(r'OT-(.*?)-associated', filename)
                if match:
                    disease_id = match.group(1)
                disease_name = disease_names.get(disease_id, None)
                
                if disease_name:
                    print("imported", disease_name)
                    # Combine OpenTargets with the source column (if available)
                    source = "OpenTargets + " + row.get("source", "").strip()

                    # Merge disease node, gene node, and the relationship with source
                    merge_node_disease("Disease", disease_name, disease_id, source)
                    merge_node_target("Gene", gene_symbol, source)
                    merge_relationship(disease_name, gene_symbol, "ASSOCIATION", score, source)
            except Exception as e:
                print(f"Error processing row {idx} in file {filename}: {e}")

print("Data import completed.")


# In[2]:


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




# In[ ]:


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


# In[ ]:


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


# In[ ]:


# Upload iTEXTMine to Neo4j: NOT USED IN PAPER
import pandas as pd
import os
from neo4j import GraphDatabase

class Neo4jUploader:
    
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()

    # Upload kinase-substrate data
    def upload_kinase_substrate(self, dataframe, source):
        with self.driver.session() as session:
            for index, row in dataframe.iterrows():
                session.write_transaction(self._create_kinase_substrate_relationship, row['kinase'], row['relation'], row['substrate'], row['evidence'], source)

    # Upload gene-miRNA data
    def upload_gene_mirna(self, dataframe, source):
        with self.driver.session() as session:
            for index, row in dataframe.iterrows():
                session.write_transaction(self._create_gene_mirna_relationship, row['gene'], row['relation'], row['mirna'], row['evidence'], source)

    # Upload gene-disease-drug response data
    def upload_gene_disease_drug_response(self, dataframe, source):
        with self.driver.session() as session:
            for index, row in dataframe.iterrows():
                session.write_transaction(self._create_gene_disease_drug_relationship, row['gene'], row['disease'], row['drug_response'], row['evidence'], source)

    # Create kinase-substrate relationship
    @staticmethod
    def _create_kinase_substrate_relationship(tx, kinase, relation, substrate, evidence, source):
        query = """
        MERGE (k:Kinase {name: $kinase, source: $source})
        MERGE (s:Substrate {name: $substrate, source: $source})
        MERGE (k)-[r:RELATION {type: $relation, evidence: $evidence, source: $source}]->(s)
        """
        tx.run(query, kinase=kinase, relation=relation, substrate=substrate, evidence=evidence, source=source)

    # Create gene-miRNA relationship
    @staticmethod
    def _create_gene_mirna_relationship(tx, gene, relation, mirna, evidence, source):
        query = """
        MERGE (g:Gene {name: $gene, source: $source})
        MERGE (m:miRNA {name: $mirna, source: $source})
        MERGE (g)-[r:RELATION {type: $relation, evidence: $evidence, source: $source}]->(m)
        """
        tx.run(query, gene=gene, relation=relation, mirna=mirna, evidence=evidence, source=source)

    # Create gene-disease-drug response relationship
    @staticmethod
    def _create_gene_disease_drug_relationship(tx, gene, disease, drug_response, evidence, source):
        query = """
        MERGE (g:Gene {name: $gene, source: $source})
        MERGE (d:Disease {name: $disease, source: $source})
        MERGE (dr:DrugResponse {type: $drug_response, source: $source})
        MERGE (g)-[:DISEASE {evidence: $evidence, source: $source}]->(d)
        MERGE (d)-[:RESPONSE {evidence: $evidence, source: $source}]->(dr)
        """
        tx.run(query, gene=gene, disease=disease, drug_response=drug_response, evidence=evidence, source=source)

# Function to extract source from file path and prepend 'iTextMine + '
def extract_source_from_path(filepath):
    """Extract the source name from the file name or directory, prepending 'iTextMine + '."""
    base_filename = os.path.basename(filepath)
    # Prepend 'iTextMine + ' to the extracted source
    source_name = "iTextMine + " + os.path.splitext(base_filename)[0]  # Remove file extension
    return source_name

# Main function to run the uploads
if __name__ == "__main__":
    
    # Connection parameters for Neo4j
    uri = "bolt://localhost:7687"  # Update with your Neo4j instance URI
    user = "neo4j"  # Neo4j username
    password = "12345678"  # Neo4j password

    # Instantiate the uploader class
    uploader = Neo4jUploader(uri, user, password)
    
    try:
        # Define the CSV file paths (update these paths with your actual file locations)
        kinase_substrate_csv = "data/iTextMine/data/Kinase-Substrate_Triples_with_Evidence.csv"
        gene_mirna_csv = "data/iTextMine/data/Gene-miRNA_Triples_with_Evidence.csv"
        gene_disease_drug_csv = "data/iTextMine/data/Gene-Disease-DrugResponse_Triples_with_Evidence.csv"
        
        # Automatically extract the source from the file name or path with 'iTextMine + '
        source_kinase_substrate = extract_source_from_path(kinase_substrate_csv)
        source_gene_mirna = extract_source_from_path(gene_mirna_csv)
        source_gene_disease_drug = extract_source_from_path(gene_disease_drug_csv)

        # Load the CSVs into pandas DataFrames
        kinase_substrate_df = pd.read_csv(kinase_substrate_csv)
        gene_mirna_df = pd.read_csv(gene_mirna_csv)
        gene_disease_drug_df = pd.read_csv(gene_disease_drug_csv)

        # Upload the data to Neo4j with extracted source information
        print(f"Uploading kinase-substrate triples from {source_kinase_substrate}...")
        uploader.upload_kinase_substrate(kinase_substrate_df, source_kinase_substrate)
        
        print(f"Uploading gene-miRNA triples from {source_gene_mirna}...")
        uploader.upload_gene_mirna(gene_mirna_df, source_gene_mirna)
        
        print(f"Uploading gene-disease-drug response triples from {source_gene_disease_drug}...")
        uploader.upload_gene_disease_drug_response(gene_disease_drug_df, source_gene_disease_drug)
        
        print("Data uploaded successfully!")
    
    finally:
        # Close the Neo4j connection
        uploader.close()


# In[3]:


#Sherpa upload
import os
from bel_json_importer.n4j_meta import Neo4jClient
from bel_json_importer.n4j_bel import Neo4jBel
paths = []
for path, _, files in os.walk("data/Sherpa"): #substitute it with "data" to laod covid and NDD and sherpa triples only
    for file in files:
        print(file)
        if file.endswith(".json"):
            print(path)
            paths.append(os.path.join(path, file))
neo = Neo4jClient(
    uri="bolt://localhost:7687", database="neo4j", user="neo4j", password="12345678"
)
#Add all three graphs covid ad pd and comorbidity
n4jbel = Neo4jBel(client=neo)
for path in paths:
    n4jbel.import_json(input_path=path, update_from_protein2gene=False) #Maria added True

print("Done")

#remember to add this fr convininece:

'match(n)-[r]->(m) where "sherpa" in r.annotationDatasource set r.source = "sherpa"'


# In[ ]:


#CBM uplaod
import os
from bel_json_importer.n4j_meta import Neo4jClient
from bel_json_importer.n4j_bel import Neo4jBel
paths = []
for path, _, files in os.walk("data/CBM/data"): #substitute it with "data" to laod covid and NDD and sherpa triples only
    for file in files:
        print(file)
        if file.endswith(".json"):
            print(path)
            paths.append(os.path.join(path, file))
neo = Neo4jClient(
    uri="bolt://localhost:7687", database="neo4j", user="neo4j", password="12345678"
)
#Add all three graphs covid ad pd and comorbidity
n4jbel = Neo4jBel(client=neo)
for path in paths:
    n4jbel.import_json(input_path=path, update_from_protein2gene=False) #Maria added True

print("Done")


# In[ ]:


#UPLOAD SCAI AD PD NDD COVID graph
import os
from bel_json_importer.n4j_meta import Neo4jClient
from bel_json_importer.n4j_bel import Neo4jBel
paths = []
for path, _, files in os.walk("data/SCAI-graphs"): #substitute it with "data" to laod covid and NDD and sherpa triples only
    for file in files:
        print(file)
        if file.endswith(".json"):
            print(path)
            paths.append(os.path.join(path, file))
neo = Neo4jClient(
    uri="bolt://localhost:7687", database="neo4j", user="neo4j", password="12345678"
)
#Add all three graphs covid ad pd and comorbidity
n4jbel = Neo4jBel(client=neo)
for path in paths:
    n4jbel.import_json(input_path=path, update_from_protein2gene=False) #Maria added True

print("Done")


# In[ ]:


#IMPORT KEEGG COMPELTE DATA
import pandas as pd
import re
import os
from neo4j import GraphDatabase

# Define the directory containing the CSV files
csv_dir = file_path = r'C:\Users\nbabaiha\Documents\GitHub\COMMUTE\commute\compare-curated-sources\kegg-api-responses-complete\updated_csv_files'


# Function to clean up information
def clean_name(name):
    return re.sub(r'\s*\[.*?\]|\(.*?\)', '', name).strip()

# Connect to Neo4j
def connect_to_neo4j(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver

# Function to create nodes and relationships in Neo4j
def upload_nodes_and_relationships(driver, records):
    with driver.session() as session:
        for record in records:
            name = record.get('NAME', '')

            # Ensure each field is a string before splitting, using empty list if None
            genes = str(record.get('GENE_SYMBOL', '')).split('; ') if isinstance(record.get('GENE_SYMBOL', ''), str) else []
            drugs = str(record.get('DRUG', '')).split('; ') if isinstance(record.get('DRUG', ''), str) else []
            pathways = str(record.get('PATHWAY', '')).split('; ') if isinstance(record.get('PATHWAY', ''), str) else []
            networks = str(record.get('NETWORK', '')).split('; ') if isinstance(record.get('NETWORK', ''), str) else []

            # Log the processed data to confirm it is being read correctly
            print(f"Processing Disease Node: {name}")
            print(f"  Genes: {genes}")
            print(f"  Drugs: {drugs}")
            print(f"  Pathways: {pathways}")
            print(f"  Networks: {networks}")

            # Create NAME node (Disease)
            session.run("""
            MERGE (n:Disease {name: $name})
            """, name=name)
            
            # Create and relate each gene to the disease
            for gene in genes:
                if gene:
                    cleaned_gene = clean_name(gene)
                    print(f"  Creating Gene Node: {cleaned_gene}")
                    session.run("""
                    MERGE (g:Gene {name: $cleaned_gene})
                    MERGE (n:Disease {name: $name})
                    MERGE (n)-[r:ASSOCIATION]->(g)
                    ON CREATE SET r.source = 'KEGG'
                    """, cleaned_gene=cleaned_gene, name=name)

            # Create and relate each drug to the disease
            for drug in drugs:
                if drug:
                    cleaned_drug = clean_name(drug)
                    print(f"  Creating Drug Node: {cleaned_drug}")
                    session.run("""
                    MERGE (d:Drug {name: $cleaned_drug})
                    MERGE (n:Disease {name: $name})
                    MERGE (n)-[r:TREATED_BY]->(d)
                    ON CREATE SET r.source = 'KEGG'
                    """, cleaned_drug=cleaned_drug, name=name)

            # Create and relate each pathway to the disease
            for pathway in pathways:
                if pathway:
                    cleaned_pathway = clean_name(pathway)
                    print(f"  Creating Pathway Node: {cleaned_pathway}")
                    session.run("""
                    MERGE (p:Pathway {name: $cleaned_pathway})
                    MERGE (n:Disease {name: $name})
                    MERGE (n)-[r:INVOLVED_IN]->(p)
                    ON CREATE SET r.source = 'KEGG'
                    """, cleaned_pathway=cleaned_pathway, name=name)

            # Create and relate each network to the disease
            for network in networks:
                if network:
                    cleaned_network = clean_name(network)
                    print(f"  Creating Network Node: {cleaned_network}")
                    session.run("""
                    MERGE (nw:Network {name: $cleaned_network})
                    MERGE (n:Disease {name: $name})
                    MERGE (n)-[r:PART_OF_NETWORK]->(nw)
                    ON CREATE SET r.source = 'KEGG'
                    """, cleaned_network=cleaned_network, name=name)

        print("Upload completed successfully!")

# Your Neo4j credentials and connection details
neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "12345678"

# Connect to the Neo4j instance
driver = connect_to_neo4j(neo4j_uri, neo4j_user, neo4j_password)

# Process each CSV file in the directory
for filename in os.listdir(csv_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(csv_dir, filename)
        data = pd.read_csv(file_path)

        if 'NAME' not in data.columns:
            raise ValueError(f"The file {filename} must contain a 'NAME' column.")

        # Create a list of dictionaries for each row in the file
        records = data.to_dict('records')

        # Upload nodes and relationships for each record
        print(f"Uploading records from {filename} to Neo4j...")
        upload_nodes_and_relationships(driver, records)

# Close the Neo4j driver connection
driver.close()





# # Queries

# In[ ]:


### get triples by source
"""

MATCH (a)-[r]->(b)
WHERE r.source STARTS WITH 'iTextMine'
RETURN a.name AS Subject, type(r) AS Predicate, b.name AS Object, r.source AS Source, r.evidence AS Evidence

"""


##CBM or Sherpa triples

"""

MATCH (a)-[r]->(b)
WHERE r.filePath CONTAINS 'CBM'
RETURN a.name AS Subject, type(r) AS Predicate, b.name AS Object, r.filePath AS FilePath, r.evidence AS Evidence


"""

#get triples by frequency
"""
MATCH (a)-[r]->(b) where b.name is not NULL 
RETURN a.name AS Subject, type(r) AS Predicate, b.name AS Object, COUNT(*) AS Frequency
ORDER BY Frequency DESC
"""

# common nodes between KEGG and Sherpa
"""MATCH (n1)-[r1]->(m1), (n2)-[r2]->(m2)
WHERE apoc.text.distance(n1.name, n2.name) < 7
  AND r1.source = 'KEGG'
  AND "sherpa" in r2.annotationDatasource 
RETURN DISTINCT n1.name AS Common_Node_KEGG, n2.name AS Common_Node_Sherpa"""


# In[ ]:






# ----------------------------
# Step 3: Analyze Graphs in Neo4j
# ----------------------------
def analyze_graphs():
    print("\nStep 3: Running graph analysis and queries in Neo4j")
    # Imported from: analyze-neo4j.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
from neo4j import GraphDatabase
import pandas as pd

# Define connection parameters
uri = "bolt://localhost:7687"  # Update if needed
user = "neo4j"
password = "12345678"

# Create a Neo4j driver instance
driver = GraphDatabase.driver(uri, auth=(user, password))

# Function to execute Cypher queries
def execute_query(query):
    with driver.session() as session:
        result = session.run(query)
        return [record for record in result]

# Close the driver connection
def close_driver():
    driver.close()


# In[ ]:


# Query to count nodes
node_count_query = "MATCH (n) RETURN count(n) AS numNodes"
num_nodes = execute_query(node_count_query)
print(f"Number of Nodes: {num_nodes[0]['numNodes']}")

# Query to count edges
edge_count_query = "MATCH ()-[r]->() RETURN count(r) AS numEdges"
num_edges = execute_query(edge_count_query)
print(f"Number of Edges: {num_edges[0]['numEdges']}")


# In[ ]:


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


# # Community detection using Neo4j Browser
# To visualize the communities as a graph instead of a table, you can modify your Cypher query to update the nodes with their community IDs and then return a visual graph representation. Here's how you can do it:
# 
# 1. First, run the code above, then run the Louvain algorithm and update the nodes with their community IDs:
# 
# ```cypher
# CALL gds.louvain.stream('myGraph')
# YIELD nodeId, communityId
# WITH gds.util.asNode(nodeId) AS node, communityId
# SET node.community = communityId
# ```
# 
# 2. Then, create a query to visualize the graph with nodes colored by community:
# 
# ```cypher
# MATCH (n)
# WITH n, n.community AS community
# RETURN n, community
# ```
# 
# When you run this query in the Neo4j Browser, it should display a graph visualization where nodes are grouped and colored by their community.
# 
# To enhance the visualization, you can add some styling to the Neo4j Browser. After running the query, you can add the following style commands in the Neo4j Browser:
# 
# ```
# :style
# node {
#   diameter: 50px;
#   color: #A5ABB6;
#   border-color: #9AA1AC;
#   border-width: 2px;
#   text-color-internal: #FFFFFF;
#   font-size: 10px;
# }
# node.community {
#   caption: '{community}';
# }
# ```
# 
# This will style the nodes to show their community IDs and give them a consistent size.
# 
# Would you like me to explain any part of this process or provide more information on customizing the visualization?

# In[ ]:


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


# In[3]:


# Node label frequency for different sources
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

def count_nodes_by_label_and_source(label, source):
    query = f"""
    MATCH (n:{label})-[r]-()
    WHERE toLower(r.source) contains toLower($source)
    RETURN COUNT(n) AS frequency
    """
    result = execute_query(query, {"source": source})
    return result[0]['frequency'] if result else 0

# List of sources to search for
sources = ['sherpa', 'pubtator', 'cbm', 'scai', 'opentargets', 'disgenet', 'indra', 'drugbank','kegg']

# Retrieve all labels
labels = get_all_labels()
label_names = [record['label'] for record in labels]

# Loop through each source, calculate frequencies, and plot bar charts
for source in sources:
    # Count nodes for each label filtered by source
    label_counts = {label: count_nodes_by_label_and_source(label, source) for label in label_names}
    
    # Convert to DataFrame and sort by frequency in descending order
    df = pd.DataFrame(list(label_counts.items()), columns=['NodeType', 'Frequency'])
    df = df[df['Frequency'] > 0]  # Filter out labels with zero frequency
    df = df.sort_values(by='Frequency', ascending=False)
    
    if not df.empty:
        # Plotting for the current source
        plt.figure(figsize=(10, 6))
        plt.bar(df['NodeType'], df['Frequency'], color='skyblue')
        plt.xlabel('Node Type')
        plt.ylabel('Frequency')
        plt.title(f'Node Types and Their Frequencies for {source.capitalize()}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Show the plot
        plt.show()

# Close the driver
driver.close()


# In[ ]:


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


# In[3]:


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


# In[ ]:


#node and realtionship tzpe distributions for primekg:
"""
MATCH (n)-[r]->(m)
WHERE any(term IN ["covid", "alzheimer", "neurodegeneration", "parkinson"] 
           WHERE n.name =~ ('(?i).*' + term + '.*') 
              OR m.name =~ ('(?i).*' + term + '.*'))
WITH DISTINCT n AS node, type(r) AS relationship_type
WITH labels(node) AS node_labels, relationship_type, count(node) AS node_count
RETURN node_labels, relationship_type, node_count
ORDER BY node_count DESC
""" 


# degree centrality for primekg

""""
MATCH (n)-[r]->(m)
WHERE any(term IN ["covid", "alzheimer", "neurodegeneration", "parkinson"] 
           WHERE n.name =~ ('(?i).*' + term + '.*') 
              OR m.name =~ ('(?i).*' + term + '.*'))

// Calculate out-degree and in-degree in a single pass
WITH n AS node, count(r) AS out_degree, collect(m) AS targets
UNWIND targets AS target
WITH node, out_degree, target
OPTIONAL MATCH (target)<-[r2]-()
WITH node, out_degree, count(r2) AS in_degree

// Calculate total degree and normalized degree centrality
WITH node, out_degree, in_degree, (out_degree + in_degree) AS total_degree
RETURN 
    node.name AS Node,
    labels(node) AS NodeLabels,
    total_degree,
    out_degree,
    in_degree,
    (total_degree * 1.0 / (1000 - 1)) AS normalized_degree_centrality  // Substitute 1000 with the actual total node count
ORDER BY normalized_degree_centrality DESC

"""


# In[ ]:


# Centrality measure (various types) and bar charts (no data source given)
from neo4j import GraphDatabase
import pandas as pd
import matplotlib.pyplot as plt

# Define connection parameters
uri = "bolt://localhost:7687"
user = "neo4j"
password = "12345678"

# Create a Neo4j driver instance
driver = GraphDatabase.driver(uri, auth=(user, password))

def execute_query(query):
    with driver.session() as session:
        result = session.run(query)
        return [record for record in result]

def compute_centrality(graph_name, centrality_type):
    query = f"""
    CALL gds.{centrality_type}.stream('{graph_name}')
    YIELD nodeId, score
    RETURN gds.util.asNode(nodeId).name AS nodeName, score AS centrality
    ORDER BY centrality DESC;
    """
    return execute_query(query)

# Define centrality types
centrality_types = ['degree', 'closeness', 'betweenness', 'eigenvector']

# Compute and store centrality measures
results = {}
for centrality_type in centrality_types:
    data = compute_centrality('myGraph', centrality_type)
    df = pd.DataFrame(data)
    
    # Print the columns to check their names
    print(f"Original columns for {centrality_type}: {df.columns.tolist()}")
    
    # Rename columns
    df.rename(columns={0: 'node_id', 1: 'centrality_score'}, inplace=True)
    
    # Sort by centrality_score in descending order
    df.sort_values(by='centrality_score', ascending=False, inplace=True)
    
    # Select top 5
    df_top10 = df.head(10)
    
    # Store the DataFrame
    results[centrality_type] = df_top10

# Close the driver
driver.close()

# Define a function to plot centrality
def plot_centrality(df, centrality_type):
    plt.figure(figsize=(12, 8))
    plt.bar(df['node_id'], df['centrality_score'], color='skyblue')
    plt.xlabel('Node Name')
    plt.ylabel(f'{centrality_type.capitalize()} Centrality')
    plt.title(f'Top 10 {centrality_type.capitalize()} Centrality of Nodes')
    plt.xticks(rotation=45, ha='right')  # Adjust rotation for better readability
    plt.tight_layout()
    plt.show()

# Plot top 5 centrality type
for centrality_type in centrality_types:
    df = results[centrality_type]
    plot_centrality(df, centrality_type)


# In[1]:


# Centrality measure (various types) and bar charts for each source
from neo4j import GraphDatabase
import pandas as pd
import matplotlib.pyplot as plt

# Define connection parameters
uri = "bolt://localhost:7687"
user = "neo4j"
password = "12345678"

# Create a Neo4j driver instance
driver = GraphDatabase.driver(uri, auth=(user, password))

def execute_query(query, parameters=None):
    with driver.session() as session:
        result = session.run(query, parameters=parameters)
        return [record for record in result]

# Function to check if a graph exists
def graph_exists(graph_name):
    query = f"CALL gds.graph.exists('{graph_name}') YIELD exists"
    result = execute_query(query)
    return result[0]['exists'] if result else False

# Function to drop the graph projection if it exists
def drop_graph_if_exists(graph_name):
    if graph_exists(graph_name):
        query = f"CALL gds.graph.drop('{graph_name}', false)"
        execute_query(query)

# Function to create a graph projection filtered by source using Cypher
def create_graph_projection(graph_name, source):
    # Drop the graph if it already exists
    drop_graph_if_exists(graph_name)
    
    query = f"""
    CALL gds.graph.project.cypher(
        '{graph_name}',
        'MATCH (n) RETURN id(n) AS id',
        'MATCH ()-[r]-() WHERE toLower(r.source) contains toLower("{source}") RETURN id(r) AS id, id(startNode(r)) AS source, id(endNode(r)) AS target'
    )
    """
    execute_query(query)

# Function to compute centrality
def compute_centrality(graph_name, centrality_type):
    query = f"""
    CALL gds.{centrality_type}.stream('{graph_name}')
    YIELD nodeId, score
    RETURN gds.util.asNode(nodeId).name AS nodeName, score AS centrality
    ORDER BY centrality DESC;
    """
    return execute_query(query)

# Define centrality types
centrality_types = ['degree', 'closeness', 'betweenness', 'eigenvector']

# Define sources
sources = ['kegg', 'sherpa', 'opentargets', 'disgenet', 'indra', 'drugbank','pubtator', 'cbm', 'scai']

# Loop through each source and compute centrality measures
for source in sources:
    graph_name = f"graph_{source}"
    
    # Create a filtered graph projection for the current source using Cypher
    create_graph_projection(graph_name, source)
    
    results = {}
    for centrality_type in centrality_types:
        # Compute centrality for the current source and type
        data = compute_centrality(graph_name, centrality_type)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Rename columns based on actual data
        df.columns = ['node_id', 'centrality_score']
        
        # Sort by centrality_score in descending order
        df.sort_values(by='centrality_score', ascending=False, inplace=True)
        
        # Select top 10 nodes
        df_top10 = df.head(10)
        
        # Store the DataFrame for the current centrality type
        results[centrality_type] = df_top10

    # Plot centrality for the current source
    def plot_centrality(df, centrality_type, source):
        # Filter out rows where 'node_id' is None
        df = df[df['node_id'].notnull()]

        # Convert 'node_id' to string type (if needed)
        df['node_id'] = df['node_id'].astype(str)

        plt.figure(figsize=(12, 8))
        plt.bar(df['node_id'], df['centrality_score'], color='skyblue')
        plt.xlabel('Node Name')
        plt.ylabel(f'{centrality_type.capitalize()} Centrality')
        plt.title(f'Top 10 {centrality_type.capitalize()} Centrality for {source.capitalize()}')
        plt.xticks(rotation=45, ha='right')  # Adjust rotation for better readability
        plt.tight_layout()
        plt.show()

    # Plot top 10 centrality measures for each type
    for centrality_type in centrality_types:
        df = results[centrality_type]
        plot_centrality(df, centrality_type, source)

# Close the driver connection
driver.close()


# In[ ]:


#general statistics
"""  
#unique nodes Sherpa


MATCH (n)-[r]->()
WHERE toLower(r.source) = toLower('sherpa')
RETURN COUNT(DISTINCT n) AS uniqueNodeCount;


#unique triples

first run:

:param source => 'sherpa'

then:

MATCH (n1)-[r]->(n2)
WHERE toLower(r.source) = toLower($source)
RETURN COUNT(DISTINCT id(n1) + id(r) + id(n2)) AS uniqueTripleCount;

"""


# In[1]:


from neo4j import GraphDatabase

# Define connection parameters
uri = "bolt://localhost:7687"  # Update with your Neo4j Bolt URL
user = "neo4j"  # Replace with your Neo4j username
password = "12345678"  # Replace with your Neo4j password

# Create a Neo4j driver instance
driver = GraphDatabase.driver(uri, auth=(user, password))

# Queries with directed relationships (for sources containing 'sherpa')
query_directed_unique_nodes = """
MATCH (n)-[r]->()
WHERE toLower(r.source) CONTAINS toLower($source)
RETURN COUNT(DISTINCT n) AS uniqueNodeCount
"""

query_directed_unique_triples = """
MATCH (n1)-[r]->(n2)
WHERE toLower(r.source) CONTAINS toLower($source)
RETURN COUNT(DISTINCT id(n1) + id(r) + id(n2)) AS uniqueTripleCount
"""

# Queries with undirected relationships (for all other sources)
query_undirected_unique_nodes = """
MATCH (n)-[r]-()
WHERE toLower(r.source) contains toLower($source)
RETURN COUNT(DISTINCT n) AS uniqueNodeCount
"""

query_undirected_unique_triples = """
MATCH (n1)-[r]-(n2)
WHERE toLower(r.source) contains toLower($source)
RETURN COUNT(DISTINCT id(n1) + id(r) + id(n2)) AS uniqueTripleCount
"""

# List of sources to query for
sources = ['indra','sherpa', 'opentargets', 'disgenet', 'indra', 'drugbank']

# Function to execute the queries for each source
def get_stats_for_source(source):
    with driver.session() as session:
        if "sherpa" in source.lower() or "indra" in source.lower() :
            # Use directed relationships for sources containing 'sherpa'
            result_nodes = session.run(query_directed_unique_nodes, source=source)
            unique_node_count = result_nodes.single()["uniqueNodeCount"]

            result_triples = session.run(query_directed_unique_triples, source=source)
            unique_triple_count = result_triples.single()["uniqueTripleCount"]
        else:
            # Use undirected relationships for other sources
            result_nodes = session.run(query_undirected_unique_nodes, source=source)
            unique_node_count = result_nodes.single()["uniqueNodeCount"]

            result_triples = session.run(query_undirected_unique_triples, source=source)
            unique_triple_count = result_triples.single()["uniqueTripleCount"]

        # Return the results
        return unique_node_count, unique_triple_count

# Loop over each source and print the results
for source in sources:
    unique_nodes, unique_triples = get_stats_for_source(source)
    print(f"Source: {source.capitalize()}")
    print(f"  Unique Nodes: {unique_nodes}")
    print(f"  Unique Triples: {unique_triples}")
    print()

# Close the Neo4j driver connection
driver.close()


# In[3]:


#above save data in csv
import pandas as pd
from neo4j import GraphDatabase

# Define connection parameters
uri = "bolt://localhost:7687"  # Update with your Neo4j Bolt URL
user = "neo4j"  # Replace with your Neo4j username
password = "12345678"  # Replace with your Neo4j password

# Create a Neo4j driver instance
driver = GraphDatabase.driver(uri, auth=(user, password))

# Queries with directed relationships (for sources containing 'sherpa')
query_directed_unique_nodes = """
MATCH (n)-[r]->()
WHERE toLower(r.source) CONTAINS toLower($source)
RETURN COUNT(DISTINCT n) AS uniqueNodeCount
"""

query_directed_unique_triples = """
MATCH (n1)-[r]->(n2)
WHERE toLower(r.source) CONTAINS toLower($source)
RETURN COUNT(DISTINCT id(n1) + id(r) + id(n2)) AS uniqueTripleCount
"""

# Queries with undirected relationships (for all other sources)
query_undirected_unique_nodes = """
MATCH (n)-[r]-()
WHERE toLower(r.source) CONTAINS toLower($source)
RETURN COUNT(DISTINCT n) AS uniqueNodeCount
"""

query_undirected_unique_triples = """
MATCH (n1)-[r]-(n2)
WHERE toLower(r.source) CONTAINS toLower($source)
RETURN COUNT(DISTINCT id(n1) + id(r) + id(n2)) AS uniqueTripleCount
"""

# List of sources to query for
sources = ['sherpa', 'opentargets', 'disgenet', 'indra', 'drugbank', 'kegg', 'cbm', 'scai','pubtator']
print("Note!!!!!For CBM and SCAI graphs and SHERPA, first choose where r.filePath is cbm or scai or sherpa, set r.source = ...")
# Function to execute the queries for each source
def get_stats_for_source(source):
    with driver.session() as session:
        if "sherpa" in source.lower():
            # Use directed relationships for sources containing 'sherpa'
            result_nodes = session.run(query_directed_unique_nodes, source=source)
            unique_node_count = result_nodes.single()["uniqueNodeCount"]

            result_triples = session.run(query_directed_unique_triples, source=source)
            unique_triple_count = result_triples.single()["uniqueTripleCount"]
        else:
            # Use undirected relationships for other sources
            result_nodes = session.run(query_undirected_unique_nodes, source=source)
            unique_node_count = result_nodes.single()["uniqueNodeCount"]

            result_triples = session.run(query_undirected_unique_triples, source=source)
            unique_triple_count = result_triples.single()["uniqueTripleCount"]

        # Return the results
        return unique_node_count, unique_triple_count

# Initialize an empty list to store the results
results = []

# Loop over each source and collect the results in the list
for source in sources:
    unique_nodes, unique_triples = get_stats_for_source(source)
    results.append({"Source": source, "Unique Nodes": unique_nodes, "Unique Triples": unique_triples})

# Create a pandas DataFrame from the results
df = pd.DataFrame(results)

# Display the DataFrame
print(df)

# Optionally, save the DataFrame to a CSV file
df.to_csv(r"C:\Users\nbabaiha\Documents\GitHub\COMMUTE\commute\neo4j-import-analysis\neo4j-analysis-results\source_statistics_recheck.csv", index=False)

# Close the Neo4j driver connection
driver.close()


# In[ ]:


# Use path files extracted from Neo4js, this Function to extract unique nodes from the 'Nodes' column
import pandas as pd

# Load the two CSV files into pandas DataFrames
file_path_all = r"C:\Users\nbabaiha\Documents\GitHub\COMMUTE\commute\neo4j-import-analysis\neo4j-db-analysis-textmining-databases/all-dbs-comorbidity-paths-3-hops.csv"
file_path_sherpa = r"C:\Users\nbabaiha\Documents\GitHub\COMMUTE\commute\neo4j-import-analysis\neo4j-db-analysis-textmining-databases/sherpa-comorbidity-paths-3-hops.csv"

df_all = pd.read_csv(file_path_all)
df_sherpa = pd.read_csv(file_path_sherpa)

# Function to extract unique nodes from the 'Nodes' column
def extract_unique_nodes(df):
    # Flatten the list of nodes and extract unique values
    nodes_series = df['Nodes'].apply(lambda x: eval(x))  # Convert string representation of list to list
    unique_nodes = set([node for sublist in nodes_series for node in sublist])
    return unique_nodes

# Extract unique nodes from both datasets
unique_nodes_all = extract_unique_nodes(df_all)
unique_nodes_sherpa = extract_unique_nodes(df_sherpa)

# Convert nodes to lowercase for case-insensitive comparison
unique_nodes_all_lower = {node.lower() for node in unique_nodes_all}
unique_nodes_sherpa_lower = {node.lower() for node in unique_nodes_sherpa}

# Function to perform fuzzy matching with a similarity threshold
def fuzzy_match_nodes(set1, set2, threshold=85):
    matched_nodes = set()
    unmatched_set1 = set()
    unmatched_set2 = set(set2)  # Track which nodes in set2 are unmatched

    for node1 in set1:
        # Find the best match from set2 for node1 using fuzzy matching
        best_match, score = process.extractOne(node1, set2, scorer=fuzz.ratio)
        
        # If the score exceeds the threshold, consider it a match
        if score >= threshold:
            matched_nodes.add((node1, best_match, score))
            unmatched_set2.discard(best_match)  # Remove matched node from unmatched set2
        else:
            unmatched_set1.add(node1)

    return matched_nodes, unmatched_set1, unmatched_set2

# Perform fuzzy matching
fuzzy_matched_nodes, fuzzy_unmatched_all, fuzzy_unmatched_sherpa = fuzzy_match_nodes(unique_nodes_all_lower, unique_nodes_sherpa_lower)

# Function to extract paths as a tuple of (Nodes, Relationships)
def extract_paths(df):
    paths_series = df.apply(lambda row: (tuple(eval(row['Nodes'])), tuple(eval(row['Relationships']))), axis=1)
    return set(paths_series)

# Extract paths from both datasets
paths_all = extract_paths(df_all)
paths_sherpa = extract_paths(df_sherpa)

# Find common paths and unique paths in both datasets
common_paths = paths_all.intersection(paths_sherpa)
unique_paths_all_only = paths_all - paths_sherpa
unique_paths_sherpa_only = paths_sherpa - paths_all

# Updated print function to print nodes in a structured format
def print_nodes(title, node_list):
    print(f"\n{title} ({len(node_list)}):")
    for idx, node in enumerate(sorted(node_list), 1):
        print(f"  {idx}. {node}")

# Display structured results
print(f"Fuzzy Matched Nodes ({len(fuzzy_matched_nodes)}):")
for idx, match in enumerate(fuzzy_matched_nodes, 1):
    print(f"  {idx}. {match[0]} (from 'all-dbs-comorbidity-paths-3-hops.csv') matches {match[1]} (from 'sherpa-comorbidity-paths-3-hops.csv') with {match[2]}% similarity")

# Print unique nodes from both datasets
print_nodes("Unique Nodes in 'all-dbs-comorbidity-paths-3-hops.csv' (Unmatched)", fuzzy_unmatched_all)
print_nodes("Unique Nodes in 'sherpa-comorbidity-paths-3-hops.csv' (Unmatched)", fuzzy_unmatched_sherpa)

# Display the path results
print(f"\nCommon Paths: {len(common_paths)}")
print(f"Unique Paths in 'all-dbs-comorbidity-paths-3-hops.csv': {len(unique_paths_all_only)}")
print(f"Unique Paths in 'sherpa-comorbidity-paths-3-hops.csv': {len(unique_paths_sherpa_only)}")


# In[ ]:


#visualize inforamtion above 
import pandas as pd
from fuzzywuzzy import fuzz, process
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# Load the two CSV files into pandas DataFrames
file_path_all = r"C:\Users\nbabaiha\Documents\GitHub\COMMUTE\commute\neo4j-import-analysis\neo4j-db-analysis-textmining-databases/all-dbs-comorbidity-paths-3-hops.csv"
file_path_sherpa = r"C:\Users\nbabaiha\Documents\GitHub\COMMUTE\commute\neo4j-import-analysis\neo4j-db-analysis-textmining-databases/sherpa-comorbidity-paths-3-hops.csv"

df_all = pd.read_csv(file_path_all)
df_sherpa = pd.read_csv(file_path_sherpa)

# Function to extract unique nodes from the 'Nodes' column
def extract_unique_nodes(df):
    # Flatten the list of nodes and extract unique values
    nodes_series = df['Nodes'].apply(lambda x: eval(x))  # Convert string representation of list to list
    unique_nodes = set([node for sublist in nodes_series for node in sublist])
    return unique_nodes

# Extract unique nodes from both datasets
unique_nodes_all = extract_unique_nodes(df_all)
unique_nodes_sherpa = extract_unique_nodes(df_sherpa)

# Convert nodes to lowercase for case-insensitive comparison
unique_nodes_all_lower = {node.lower() for node in unique_nodes_all}
unique_nodes_sherpa_lower = {node.lower() for node in unique_nodes_sherpa}

# Function to perform fuzzy matching with a similarity threshold
def fuzzy_match_nodes(set1, set2, threshold=85):
    matched_nodes = set()
    unmatched_set1 = set()
    unmatched_set2 = set(set2)  # Track which nodes in set2 are unmatched

    for node1 in set1:
        # Find the best match from set2 for node1 using fuzzy matching
        best_match, score = process.extractOne(node1, set2, scorer=fuzz.ratio)
        
        # If the score exceeds the threshold, consider it a match
        if score >= threshold:
            matched_nodes.add((node1, best_match, score))
            unmatched_set2.discard(best_match)  # Remove matched node from unmatched set2
        else:
            unmatched_set1.add(node1)

    return matched_nodes, unmatched_set1, unmatched_set2

# Perform fuzzy matching
fuzzy_matched_nodes, fuzzy_unmatched_all, fuzzy_unmatched_sherpa = fuzzy_match_nodes(unique_nodes_all_lower, unique_nodes_sherpa_lower)

# Function to extract paths as a tuple of (Nodes, Relationships)
def extract_paths(df):
    paths_series = df.apply(lambda row: (tuple(eval(row['Nodes'])), tuple(eval(row['Relationships']))), axis=1)
    return set(paths_series)

# Extract paths from both datasets
paths_all = extract_paths(df_all)
paths_sherpa = extract_paths(df_sherpa)

# Find common paths and unique paths in both datasets
common_paths = paths_all.intersection(paths_sherpa)
unique_paths_all_only = paths_all - paths_sherpa
unique_paths_sherpa_only = paths_sherpa - paths_all

# Updated print function to print nodes in a structured format
def print_nodes(title, node_list):
    print(f"\n{title} ({len(node_list)}):")
    for idx, node in enumerate(sorted(node_list), 1):
        print(f"  {idx}. {node}")

# Display structured results
print(f"Fuzzy Matched Nodes ({len(fuzzy_matched_nodes)}):")
for idx, match in enumerate(fuzzy_matched_nodes, 1):
    print(f"  {idx}. {match[0]} (from 'all-dbs-comorbidity-paths-3-hops.csv') matches {match[1]} (from 'sherpa-comorbidity-paths-3-hops.csv') with {match[2]}% similarity")

# Print unique nodes from both datasets
print_nodes("Unique Nodes in 'all-dbs-comorbidity-paths-3-hops.csv' (Unmatched)", fuzzy_unmatched_all)
print_nodes("Unique Nodes in 'sherpa-comorbidity-paths-3-hops.csv' (Unmatched)", fuzzy_unmatched_sherpa)

# Display the path results
print(f"\nCommon Paths: {len(common_paths)}")
print(f"Unique Paths in 'all-dbs-comorbidity-paths-3-hops.csv': {len(unique_paths_all_only)}")
print(f"Unique Paths in 'sherpa-comorbidity-paths-3-hops.csv': {len(unique_paths_sherpa_only)}")

# Data for visualization
common_nodes_count = len(fuzzy_matched_nodes)
unique_nodes_all_count = len(fuzzy_unmatched_all)
unique_nodes_sherpa_count = len(fuzzy_unmatched_sherpa)

common_paths_count = len(common_paths)
unique_paths_all_count = len(unique_paths_all_only)
unique_paths_sherpa_count = len(unique_paths_sherpa_only)

# Bar chart to compare the number of common and unique nodes
plt.figure(figsize=(10, 5))
categories = ['Common Nodes', 'Unique Nodes (All)', 'Unique Nodes (Sherpa)']
values = [common_nodes_count, unique_nodes_all_count, unique_nodes_sherpa_count]
plt.bar(categories, values, color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Comparison of Nodes Between All-dbs and Sherpa')
plt.ylabel('Node Count')
plt.show()

# Bar chart to compare the number of common and unique paths
plt.figure(figsize=(10, 5))
categories = ['Common Paths', 'Unique Paths (All)', 'Unique Paths (Sherpa)']
values = [common_paths_count, unique_paths_all_count, unique_paths_sherpa_count]
plt.bar(categories, values, color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Comparison of Paths Between All-dbs and Sherpa')
plt.ylabel('Path Count')
plt.show()

# Venn diagram for nodes
plt.figure(figsize=(6, 6))
venn2(subsets=(unique_nodes_all_count, unique_nodes_sherpa_count, common_nodes_count), 
      set_labels=('All-dbs Nodes', 'Sherpa Nodes'))
plt.title('Venn Diagram of Nodes Comparison')
plt.show()

# Venn diagram for paths
plt.figure(figsize=(6, 6))
venn2(subsets=(unique_paths_all_count, unique_paths_sherpa_count, common_paths_count), 
      set_labels=('All-dbs Paths', 'Sherpa Paths'))
plt.title('Venn Diagram of Paths Comparison')
plt.show()


# ## Use pathway excel files (3 or 5 hops) and Compute intersection, common/unique paths and nodes among all sources

# In[2]:


import pandas as pd
import os
import re
from fuzzywuzzy import fuzz  # Alternatively, you can use rapidfuzz

# List of common suffixes/terms to remove
suffixes_to_remove = ["disease", "disorder", "syndrome", "infection", "virus", "condition", r"\d+"]  # Includes numbers (e.g., "19")

# Function to normalize nodes (lowercase, remove punctuation, strip leading/trailing spaces, remove suffixes but preserve spaces)
def normalize_node(node):
    # Convert to lowercase
    node = node.lower()
    
    # Remove possessive apostrophes
    node = re.sub(r"'s", "", node)
    
    # Remove punctuation except spaces
    node = re.sub(r'[^\w\s]', '', node)
    
    # Remove common suffixes
    for suffix in suffixes_to_remove:
        node = re.sub(rf"\b{suffix}\b", "", node)  # Remove the suffix
    
    # Remove extra spaces from leading/trailing
    node = node.strip()
    
    return node

# Function to perform fuzzy matching and identify similar nodes
def fuzzy_unique_nodes(node_list, threshold=80):
    unique_nodes = []
    
    for node in node_list:
        matched = False
        for unique_node in unique_nodes:
            if fuzz.ratio(node, unique_node) >= threshold:
                matched = True
                break
        if not matched:
            unique_nodes.append(node)
    
    return unique_nodes

# Load all the provided CSV files
file_paths = [
    r'C:\Users\nbabaiha\Documents\GitHub\COMMUTE\commute\neo4j-import-analysis\neo4j-analysis-results/opentargets+disgenet+drugbank+indra-comorbidity-paths-3-hops.csv',
    r'C:\Users\nbabaiha\Documents\GitHub\COMMUTE\commute\neo4j-import-analysis\neo4j-analysis-results/cbm-comorbidity-paths-3-hops.csv',
    r'C:\Users\nbabaiha\Documents\GitHub\COMMUTE\commute\neo4j-import-analysis\neo4j-analysis-results/pubtator-comorbidity-paths-3-hops.csv',
    r'C:\Users\nbabaiha\Documents\GitHub\COMMUTE\commute\neo4j-import-analysis\neo4j-analysis-results/scaiDMaps-comorbidity-paths-3-hops.csv',
    r'C:\Users\nbabaiha\Documents\GitHub\COMMUTE\commute\neo4j-import-analysis\neo4j-analysis-results/sherpa-comorbidity-paths-3-hops.csv',
    r"C:\Users\nbabaiha\Documents\GitHub\COMMUTE\commute\neo4j-import-analysis\neo4j-analysis-results\primekg-comorbidity-paths-3-hops.csv"
]

# Load the dataframes into a dictionary for processing
dfs = {file_path: pd.read_csv(file_path) for file_path in file_paths}

# Function to normalize and extract nodes and paths
def normalize_and_extract_nodes_paths(df, node_col='Nodes'):
    all_nodes = []
    all_paths = []
    
    # Extract individual nodes and paths from the series of nodes in every path
    for path in df[node_col]:
        nodes = eval(path)  # Convert string to list
        normalized_nodes = [normalize_node(node) for node in nodes]  # Normalize each node
        all_nodes.extend(normalized_nodes)  # Add normalized nodes to the list
        normalized_path = tuple(normalized_nodes)  # Store path as a tuple (immutable) for uniqueness
        all_paths.append(normalized_path)
    
    # Apply fuzzy matching to the nodes list to group similar nodes
    unique_fuzzy_nodes = fuzzy_unique_nodes(all_nodes)
    
    return unique_fuzzy_nodes, all_paths

# Step 1: Collect all normalized and fuzzy-matched nodes and paths from each source
all_nodes_paths = {}
for file_path in dfs:
    nodes, paths = normalize_and_extract_nodes_paths(dfs[file_path])
    all_nodes_paths[file_path] = {
        "Nodes": nodes,
        "Paths": paths,
        "Total Paths": len(paths)  # Count total number of paths for this source
    }

# Step 2: Identify unique nodes and paths for each source (compared to all other sources)
unique_summary = {}
all_sources_nodes = {file_path: set(all_nodes_paths[file_path]["Nodes"]) for file_path in all_nodes_paths}
all_sources_paths = {file_path: set(all_nodes_paths[file_path]["Paths"]) for file_path in all_nodes_paths}

for file_path in dfs:
    # Unique nodes for this source: Nodes in this source but not in any other source
    other_sources_nodes = set().union(*[all_sources_nodes[other] for other in all_sources_nodes if other != file_path])
    unique_nodes = all_sources_nodes[file_path] - other_sources_nodes

    # Unique paths for this source: Paths in this source but not in any other source
    other_sources_paths = set().union(*[all_sources_paths[other] for other in all_sources_paths if other != file_path])
    unique_paths = all_sources_paths[file_path] - other_sources_paths

    unique_summary[file_path] = {
        "Unique Nodes": list(unique_nodes),
        "Unique Nodes Count": len(unique_nodes),
        "Unique Paths": list(unique_paths),
        "Unique Paths Count": len(unique_paths),
        "Total Paths": all_nodes_paths[file_path]["Total Paths"]  # Total number of paths
    }

# Step 3: Identify common nodes and paths (present in all sources)
common_nodes = set.intersection(*all_sources_nodes.values())
common_paths = set.intersection(*all_sources_paths.values())

# Create a summary DataFrame for unique node, path counts, and total paths
summary_data = {
    file_path: [
        unique_summary[file_path]["Unique Nodes Count"],
        unique_summary[file_path]["Unique Paths Count"],
        unique_summary[file_path]["Total Paths"]
    ]
    for file_path in unique_summary
}
summary_df = pd.DataFrame(summary_data, index=["Unique Nodes Count", "Unique Paths Count", "Total Paths"]).T

# Create a combined DataFrame for all unique nodes from each source
unique_nodes_combined = pd.DataFrame()
for file_path in unique_summary:
    base_name = os.path.basename(file_path).split('.')[0][:20]  # Limit to 20 characters
    unique_nodes_combined[base_name] = pd.Series(unique_summary[file_path]["Unique Nodes"])

# Save the summary, unique nodes, unique paths, and common elements to an Excel file
output_file_path_summary = 'all_paths_3hops_summary.xlsx'
with pd.ExcelWriter(output_file_path_summary) as writer:
    # Save the summary data
    summary_df.to_excel(writer, sheet_name="Summary")
    
    # Save unique nodes and paths for each source into separate sheets
    for file_path in unique_summary:
        base_name = os.path.basename(file_path).split('.')[0][:20]  # Limit to 20 characters
        
        # Save unique nodes
        nodes_df = pd.DataFrame(unique_summary[file_path]["Unique Nodes"], columns=["Unique Nodes"])
        nodes_df.to_excel(writer, sheet_name=base_name + "_un_nodes")
        
        # Save unique paths
        paths_df = pd.DataFrame([' -> '.join(path) for path in unique_summary[file_path]["Unique Paths"]], columns=["Unique Paths"])
        paths_df.to_excel(writer, sheet_name=base_name + "_un_paths")
    
    # Save the common nodes across all sources
    common_nodes_df = pd.DataFrame(list(common_nodes), columns=["Common Nodes"])
    common_nodes_df.to_excel(writer, sheet_name="Common Nodes")
    
    # Save the common paths across all sources
    common_paths_df = pd.DataFrame([' -> '.join(path) for path in common_paths], columns=["Common Paths"])
    common_paths_df.to_excel(writer, sheet_name="Common Paths")
    
    # Save the combined unique nodes from all sources in one sheet
    unique_nodes_combined.to_excel(writer, sheet_name="All Unique Nodes")

# Print the summary to the console
print(summary_df)

# Output the file path for the saved Excel file
print(f"Summary, unique nodes, paths, and common elements saved to: {output_file_path_summary}")


# ## Use the excel table : like the one in paper and create barcharts from it

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt

# Updated order of file names as requested
data_ordered = {
    'File': [
        "DisGeNET-OpenTargets-DrugBank-INDRA",
        "CBM",
        "SCAI-DMaps",
        "PrimeKG",
        "Pubtator3",
        "Sherpa"
    ],
    'Unique Nodes': [44, 27, 17, 58, 1, 3],
    'Unique Paths': [96, 39, 14, 64, 1, 6],
    'Total Paths': [104, 39, 14, 64, 1, 7]
}

# Create ordered DataFrame
df_ordered = pd.DataFrame(data_ordered)

# Plotting the ordered data
fig, ax = plt.subplots(figsize=(10, 6))

# Bar width
bar_width = 0.25
index = range(len(df_ordered))

# Plot each bar for Unique Nodes, Unique Paths, and Total Paths
ax.bar(index, df_ordered['Unique Nodes'], bar_width, label='Unique Nodes', color='blue')
ax.bar([i + bar_width for i in index], df_ordered['Unique Paths'], bar_width, label='Unique Paths', color='green')
ax.bar([i + 2 * bar_width for i in index], df_ordered['Total Paths'], bar_width, label='Total Paths', color='red')

# Add labels, title, and ticks
ax.set_xlabel('Sources', fontsize=12)  # Change from Files to Sources
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Comparison of Unique Nodes, Unique Paths, and Total Paths', fontsize=14)
ax.set_xticks([i + bar_width for i in index])
ax.set_xticklabels(df_ordered['File'], rotation=45, ha='right', fontsize=10)

# Add legend
ax.legend()

# Show the updated plot
plt.tight_layout()
plt.show()


# # CYPHER QUERIES

# ### SCAI graphs schema to get covid vs NDD graphs (AD and PD)

# In[ ]:


"""
WITH ['alzheimer', 'parkinson'] AS targetKeywords, ['covid'] AS covidKeyword
MATCH (d1), (d2)
WHERE 
  any(keyword IN targetKeywords WHERE toLower(d1.name) CONTAINS toLower(keyword)) 
  AND any(keyword IN covidKeyword WHERE toLower(d2.name) CONTAINS toLower(keyword)) 
  AND toLower(d1.source) CONTAINS "scai"
  AND toLower(d2.source) CONTAINS "scai"

// Step 3: Compute shortest path
MATCH path = shortestPath((d1)-[*..3]-(d2)) // Find shortest path within 3 hops
RETURN DISTINCT path, length(path) AS pathLength

"""


# In[ ]:


## get as alist of nds and rels the paths of SCAI graphs of comorbidity
"""
WITH ['alzheimer', 'parkinson'] AS targetKeywords, ['covid'] AS covidKeyword
MATCH (d1), (d2)
WHERE 
  any(keyword IN targetKeywords WHERE toLower(d1.name) CONTAINS toLower(keyword)) 
  AND any(keyword IN covidKeyword WHERE toLower(d2.name) CONTAINS toLower(keyword)) 
  AND toLower(d1.source) CONTAINS "scai"
  AND toLower(d2.source) CONTAINS "scai"

// Step 3: Compute shortest path
MATCH path = shortestPath((d1)-[*..3]-(d2)) // Find shortest path within 3 hops

// Step 4: Extract node IDs and relationship types from the path
WITH DISTINCT [node IN nodes(path) | id(node)] AS pathNodes, 
               [rel IN relationships(path) | type(rel)] AS pathRelationships

// Step 5: Retrieve node details based on node IDs
UNWIND pathNodes AS nodeId
MATCH (n)
WHERE id(n) = nodeId
WITH pathNodes, pathRelationships, n, 
     CASE WHEN 'name' IN keys(n) THEN n.name ELSE n.symbol END AS nodeName

// Step 6: Collect node names back into a list
WITH DISTINCT pathNodes, pathRelationships, collect(nodeName) AS nodeNames

// Step 7: Return the lists of node names and relationship types
RETURN nodeNames AS pathNodes, pathRelationships
"""


# In[ ]:


#modify code above to get also graph spurce(path)
"""
WITH ['alzheimer', 'parkinson'] AS targetKeywords, ['covid'] AS covidKeyword
MATCH (d1), (d2)
WHERE 
  any(keyword IN targetKeywords WHERE toLower(d1.name) CONTAINS toLower(keyword)) 
  AND any(keyword IN covidKeyword WHERE toLower(d2.name) CONTAINS toLower(keyword)) 
  AND toLower(d1.source) CONTAINS "scai"
  AND toLower(d2.source) CONTAINS "scai"

// Step 3: Compute shortest path
MATCH path = shortestPath((d1)-[*..3]-(d2)) // Find shortest path within 3 hops

// Step 4: Extract node IDs, relationship types, and modified file paths from the path
WITH DISTINCT [node IN nodes(path) | id(node)] AS pathNodes, 
               [rel IN relationships(path) | 
                   {type: type(rel), 
                    filePath: substring(split(rel.filePath, "data/SCAI-graphs/")[1], 0, size(split(rel.filePath, ".bel.json")[0]) - size("data/SCAI-graphs/"))}
               ] AS pathRelationships

// Step 5: Retrieve node details based on node IDs
UNWIND pathNodes AS nodeId
MATCH (n)
WHERE id(n) = nodeId
WITH pathNodes, pathRelationships, n, 
     CASE WHEN 'name' IN keys(n) THEN n.name ELSE n.symbol END AS nodeName

// Step 6: Collect node names back into a list
WITH DISTINCT pathNodes, pathRelationships, collect(nodeName) AS nodeNames

// Step 7: Return the lists of node names and relationship details
RETURN nodeNames AS pathNodes, pathRelationships


"""


# # INTERSECTION of all graphs of all sources, (SHOULD CHECK PRIMEKG separately!!!!!)

# In[ ]:


"""  For exact match of tuples and same names
MATCH (n1)-[r1]->(m1), (n2)-[r2]->(m2)
WHERE n1.name = n2.name
  AND m1.name = m2.name
  AND r1.source <> r2.source
RETURN 
  n1.name AS Common_Node,
  type(r1) AS Relationship_Type_1, 
  type(r2) AS Relationship_Type_2,
  r1.source AS Source_1, 
  r2.source AS Source_2, 
  m1.name AS Related_Node
LIMIT 100

"""

#For fuzzy match:
"""" 
MATCH (n1)-[r1]->(m1), (n2)-[r2]->(m2)
WHERE apoc.text.levenshteinDistance(n1.name, n2.name) < 3
  AND apoc.text.levenshteinDistance(m1.name, m2.name) < 3
  AND r1.source <> r2.source
RETURN 
  n1.name AS Common_Node_1,
  n2.name AS Common_Node_2,
  type(r1) AS Relationship_Type_1, 
  type(r2) AS Relationship_Type_2,
  r1.source AS Source_1, 
  r2.source AS Source_2, 
  m1.name AS Related_Node_1,
  m2.name AS Related_Node_2
LIMIT 100

"""


# # Common Schema: get all paths between all disease with hop max 3 as a graph usign SHortest path analysis

# ## get graph of database integration as graph (not used for paper)!
# 

# In[ ]:


""""
WITH ['alzheimer', 'parkinson', 'neurodegenerative', 'covid', 'neurodegeneration'] AS keywords

// Step 1: Find disease-related nodes
MATCH (d)
WHERE any(keyword IN keywords WHERE toLower(d.name) CONTAINS keyword)
WITH collect(d) AS disease_nodes

// Step 2: Generate unique pairs of disease nodes
UNWIND disease_nodes AS d1
UNWIND disease_nodes AS d2
WITH d1, d2
WHERE id(d1) < id(d2)  // Ensure unique pairs and avoid self-matches

// Step 3: Compute shortest path with filtered relationships
MATCH path = shortestPath((d1)-[r*..3]-(d2)) // Find shortest path within 3 hops
WHERE all(rel IN relationships(path) WHERE 
    any(source_keyword IN ['opentargets', 'disgenet', 'drugbank', 'indra'] 
        WHERE toLower(rel.source) CONTAINS source_keyword)) // Check if source contains any of the target keywords

// Step 4: Return paths for visualization
RETURN path
LIMIT 200  // Adjust this limit based on your visualization needs and graph size
"""


# 
# ## get graph of database integration as list of paths and ndoeds
# 

# In[ ]:


""""
WITH ['alzheimer', 'parkinson', 'neurodegenerative', 'covid', 'neurodegeneration'] AS keywords

// Step 1: Group similar nodes
MATCH (d)
WHERE any(keyword IN keywords WHERE toLower(d.name) CONTAINS keyword)
WITH d.name AS name, collect(d) AS similar_nodes, keywords
WITH apoc.text.levenshteinSimilarity(name, reduce(s = '', k IN keywords | s + k)) AS similarity, name, similar_nodes
WITH similarity, name, similar_nodes
ORDER BY similarity DESC
WITH collect({similarity: similarity, name: name, nodes: similar_nodes}) AS grouped_nodes

// Step 2: Generate unique pairs of grouped nodes
UNWIND range(0, size(grouped_nodes)-2) AS i
UNWIND range(i+1, size(grouped_nodes)-1) AS j
WITH grouped_nodes[i] AS group1, grouped_nodes[j] AS group2

// Step 3: Compute shortest path with filtered relationships
UNWIND group1.nodes AS n1
UNWIND group2.nodes AS n2
MATCH path = shortestPath((n1)-[r*..3]-(n2)) // Find shortest path within 3 hops
WHERE all(rel IN relationships(path) WHERE 
    any(source_keyword IN ['opentargets', 'disgenet', 'drugbank', 'indra'] 
        WHERE toLower(rel.source) CONTAINS source_keyword)) // Check if source contains any of the target keywords

// Step 4: Extract and format path information
WITH 
    group1.name AS disease1,
    group2.name AS disease2,
    [node IN nodes(path) | node.name] AS nodeNames,
    [rel IN relationships(path) | type(rel)] AS relTypes,
    [rel IN relationships(path) | rel.source] AS relSources,
    length(path) AS pathLength

// Step 5: Format data for CSV
RETURN 
    disease1 AS Disease1,
    disease2 AS Disease2,
    pathLength AS PathLength,
    apoc.text.join(nodeNames, ' -> ') AS PathNodes,
    apoc.text.join(relTypes, ' -> ') AS RelationshipTypes,
    apoc.text.join(relSources, ' -> ') AS RelationshipSources
ORDER BY PathLength"""


# 
# ## use strict groupping (ignore 's and take all covid nodes as one) (not used for paper exactly!)
# 
# 

# In[ ]:


"""WITH ['alzheimer', 'parkinson', 'neurodegenerative', 'covid', 'neurodegeneration'] AS keywords,
     ['disease', 'familial', 'susceptibility', 'to', 'with', 'spastic', 'paraparesis'] AS commonWords // Define common words to be ignored

// Step 1: Match nodes and clean up names by removing punctuation and common words
MATCH (d)
WHERE any(keyword IN keywords WHERE toLower(d.name) CONTAINS keyword)
WITH d, 
     apoc.text.regreplace(toLower(d.name), "[^a-zA-Z0-9\\s]", "") AS cleanName, // Remove punctuation and convert to lowercase
     commonWords

// Step 2: Remove common words like 'disease' and trim spaces
WITH d, reduce(name = trim(cleanName), word IN commonWords | apoc.text.replace(name, word, '')) AS finalName

// Step 3: Keep only the base name before the first number (if any) or extra characters
WITH d, apoc.text.regexGroups(finalName, "(alzheimer|parkinson|neurodegenerative|covid|neurodegeneration)") AS baseName

// Step 4: Group nodes by the cleaned and normalized base name
WITH baseName[0] AS GroupName, collect(d) AS similar_nodes

// Step 5: Return distinct groups and their nodes
RETURN 
    GroupName, 
    [node IN similar_nodes | node.name] AS NodeNames
ORDER BY GroupName"""


# ## get paths as graphs for strict groupping (ignore 's and take all covid nodes as one)

# In[ ]:


""""  
WITH ['alzheimer', 'parkinson', 'neurodegenerative', 'covid', 'neurodegeneration'] AS keywords,
     ['disease', 'susceptibility', 'familial', 'with', 'to', 'and', 'type', 'number'] AS commonWords // Common words to ignore

// Step 1: Match nodes and clean up names by removing punctuation and common words
MATCH (d)
WHERE any(keyword IN keywords WHERE toLower(d.name) CONTAINS keyword)
WITH d, 
     apoc.text.regreplace(toLower(d.name), "[^a-zA-Z0-9\\s]", "") AS cleanName, // Remove punctuation and convert to lowercase
     commonWords

// Step 2: Remove common words like 'disease' and numbers
WITH d, trim(reduce(name = cleanName, word IN commonWords | apoc.text.replace(name, word, ''))) AS finalName // Remove common words

// Step 3: Group nodes by the cleaned and normalized name
WITH finalName, collect(d) AS similar_nodes

// Step 4: Generate unique pairs of grouped nodes
WITH collect({name: finalName, nodes: similar_nodes}) AS grouped_nodes
UNWIND range(0, size(grouped_nodes)-2) AS i
UNWIND range(i+1, size(grouped_nodes)-1) AS j
WITH grouped_nodes[i] AS group1, grouped_nodes[j] AS group2

// Step 5: Compute shortest path with filtered relationships
UNWIND group1.nodes AS n1
UNWIND group2.nodes AS n2
MATCH path = shortestPath((n1)-[r*..3]-(n2)) // Find shortest path within 3 hops
WHERE all(rel IN relationships(path) WHERE 
    any(source_keyword IN ['opentargets', 'disgenet', 'drugbank', 'indra'] 
        WHERE toLower(rel.source) CONTAINS source_keyword)) // Check if source contains any of the target keywords

// Step 6: Return the full path for visualization
RETURN path


"""


# 
# 
# ## get paths as list of nodeds and edges for use strict groupping (ignore 's and take all covid nodes as one) (used for paper)
# 
# 

# In[ ]:


"""  
WITH ['alzheimer', 'parkinson', 'neurodegenerative', 'covid', 'neurodegeneration'] AS keywords,
     ['disease', 'susceptibility', 'familial', 'with', 'to', 'and', 'type', 'number'] AS commonWords // Common words to ignore

// Step 1: Match nodes and clean up names by removing punctuation and common words
MATCH (d)
WHERE any(keyword IN keywords WHERE toLower(d.name) CONTAINS keyword)
WITH d, 
     apoc.text.regreplace(toLower(d.name), "[^a-zA-Z0-9\\s]", "") AS cleanName, // Remove punctuation and convert to lowercase
     commonWords

// Step 2: Remove common words like 'disease' and numbers
WITH d, trim(reduce(name = cleanName, word IN commonWords | apoc.text.replace(name, word, ''))) AS finalName // Remove common words

// Step 3: Group nodes by the cleaned and normalized name
WITH finalName, collect(d) AS similar_nodes

// Step 4: Generate unique pairs of grouped nodes
WITH collect({name: finalName, nodes: similar_nodes}) AS grouped_nodes
UNWIND range(0, size(grouped_nodes)-2) AS i
UNWIND range(i+1, size(grouped_nodes)-1) AS j
WITH grouped_nodes[i] AS group1, grouped_nodes[j] AS group2

// Step 5: Compute shortest path with filtered relationships
UNWIND group1.nodes AS n1
UNWIND group2.nodes AS n2
MATCH path = shortestPath((n1)-[r*..3]-(n2)) // Find shortest path within 3 hops
WHERE all(rel IN relationships(path) WHERE 
    any(source_keyword IN ['opentargets', 'disgenet', 'drugbank', 'indra'] 
        WHERE toLower(rel.source) CONTAINS source_keyword)) // Check if source contains any of the target keywords

// Step 6: Extract and format path information, using COALESCE for node fallback
WITH 
    group1.name AS disease1,
    group2.name AS disease2,
    [node IN nodes(path) | COALESCE(node.name, node.symbol, toString(id(node)))] AS nodeNames, // Fallback: name -> symbol -> id
    [rel IN relationships(path) | type(rel)] AS relTypes,
    [rel IN relationships(path) | rel.source] AS relSources,
    length(path) AS pathLength

// Step 7: Format data for CSV
RETURN 
    disease1 AS Disease1,
    disease2 AS Disease2,
    pathLength AS PathLength,
    apoc.text.join(nodeNames, ' -> ') AS PathNodes,
    apoc.text.join(relTypes, ' -> ') AS RelationshipTypes,
    apoc.text.join(relSources, ' -> ') AS RelationshipSources
ORDER BY PathLength
"""


# ## above but getting and visualizing paths where covid is one of the nodes
# 

# In[ ]:


"""  
WITH ['alzheimer', 'parkinson', 'neurodegenerative', 'covid', 'neurodegeneration'] AS keywords,
     ['disease', 'susceptibility', 'familial', 'with', 'to', 'and', 'type', 'number'] AS commonWords // Common words to ignore

// Step 1: Match nodes and clean up names by removing punctuation and common words
MATCH (d)
WHERE any(keyword IN keywords WHERE toLower(d.name) CONTAINS keyword)
WITH d, 
     apoc.text.regreplace(toLower(d.name), "[^a-zA-Z0-9\\s]", "") AS cleanName, // Remove punctuation and convert to lowercase
     commonWords

// Step 2: Remove common words like 'disease' and numbers
WITH d, trim(reduce(name = cleanName, word IN commonWords | apoc.text.replace(name, word, ''))) AS finalName // Remove common words

// Step 3: Group nodes by the cleaned and normalized name
WITH finalName, collect(d) AS similar_nodes

// Step 4: Generate unique pairs of grouped nodes
WITH collect({name: finalName, nodes: similar_nodes}) AS grouped_nodes
UNWIND range(0, size(grouped_nodes)-2) AS i
UNWIND range(i+1, size(grouped_nodes)-1) AS j
WITH grouped_nodes[i] AS group1, grouped_nodes[j] AS group2

// Step 5: Compute shortest path with filtered relationships
UNWIND group1.nodes AS n1
UNWIND group2.nodes AS n2
MATCH path = shortestPath((n1)-[r*..3]-(n2)) // Find shortest path within 3 hops
WHERE all(rel IN relationships(path) WHERE 
    any(source_keyword IN ['opentargets', 'disgenet', 'drugbank', 'indra'] 
        WHERE toLower(rel.source) CONTAINS source_keyword)) // Check if source contains any of the target keywords

// Step 6: Filter for paths where one node is related to COVID
WITH path, nodes(path) AS pathNodes
WHERE any(node IN pathNodes WHERE toLower(node.name) CONTAINS 'covid') // Ensure one node contains 'covid'

// Step 7: Return the full path for visualization
RETURN path

"""


# ## above but as paths as a list of ndoeds and eedges

# In[ ]:


"""  
WITH ['alzheimer', 'parkinson', 'neurodegenerative', 'covid', 'neurodegeneration'] AS keywords,
     ['disease', 'susceptibility', 'familial', 'with', 'to', 'and', 'type', 'number'] AS commonWords // Common words to ignore

// Step 1: Match nodes and clean up names by removing punctuation and common words
MATCH (d)
WHERE any(keyword IN keywords WHERE toLower(d.name) CONTAINS keyword)
WITH d, 
     apoc.text.regreplace(toLower(d.name), "[^a-zA-Z0-9\\s]", "") AS cleanName, // Remove punctuation and convert to lowercase
     commonWords

// Step 2: Remove common words like 'disease' and numbers
WITH d, trim(reduce(name = cleanName, word IN commonWords | apoc.text.replace(name, word, ''))) AS finalName // Remove common words

// Step 3: Group nodes by the cleaned and normalized name
WITH finalName, collect(d) AS similar_nodes

// Step 4: Generate unique pairs of grouped nodes
WITH collect({name: finalName, nodes: similar_nodes}) AS grouped_nodes
UNWIND range(0, size(grouped_nodes)-2) AS i
UNWIND range(i+1, size(grouped_nodes)-1) AS j
WITH grouped_nodes[i] AS group1, grouped_nodes[j] AS group2

// Step 5: Compute shortest path with filtered relationships
UNWIND group1.nodes AS n1
UNWIND group2.nodes AS n2
MATCH path = shortestPath((n1)-[r*..3]-(n2)) // Find shortest path within 3 hops
WHERE all(rel IN relationships(path) WHERE 
    any(source_keyword IN ['opentargets', 'disgenet', 'drugbank', 'indra'] 
        WHERE toLower(rel.source) CONTAINS source_keyword)) // Check if source contains any of the target keywords

// Step 6: Filter for paths where one node is related to COVID
WITH path, nodes(path) AS pathNodes, relationships(path) AS pathRels
WHERE any(node IN pathNodes WHERE toLower(node.name) CONTAINS 'covid') // Ensure one node contains 'covid'

// Step 7: Extract node and relationship information for the paths
WITH 
    [node IN pathNodes | COALESCE(node.name, node.symbol, toString(id(node)))] AS nodeNames, // Fallback: name -> symbol -> id
    [rel IN pathRels | type(rel)] AS relTypes,
    [rel IN pathRels | rel.source] AS relSources,
    length(path) AS pathLength

// Step 8: Return the lists of nodes, relationships, path length, and sources
RETURN 
    nodeNames AS Nodes, 
    relTypes AS Relationships, 
    pathLength AS PathLength,
    relSources AS RelationshipSources
ORDER BY pathLength ASC // Sort by path length in ascending order
//LIMIT 200

"""


# ## graph statistics: total nodes, unique nodes, total tripels, unique tripels and density

# In[ ]:


# graph density general query

""" 
// Calculate total number of nodes
MATCH (n)
WITH count(n) as nodeCount

// Calculate total number of relationships
MATCH ()-[r]->()
WITH nodeCount, count(r) as edgeCount

// Calculate maximum possible edges for a directed graph
// For undirected graph, divide by 2
WITH nodeCount, edgeCount, nodeCount * (nodeCount - 1) as maxPossibleEdges

// Calculate density
// Density = actual edges / possible edges
WITH 
    nodeCount as nodes,
    edgeCount as edges,
    maxPossibleEdges as possibleEdges,
    toFloat(edgeCount) / (nodeCount * (nodeCount - 1)) as density

RETURN {
    numberOfNodes: nodes,
    numberOfEdges: edges,
    maxPossibleEdges: possibleEdges,
    graphDensity: density
} as graphMetrics
"""


# In[1]:


"""  
WITH ['Sherpa', 'PubTator', 'CBM', 'scai', 'opentargets', 'disgenet', 'indra', 'drugbank', 'kegg'] AS sources
UNWIND sources AS source
MATCH (n)-[r]-(m)
WHERE toLower(r.source) CONTAINS toLower(source)
WITH source, 
     count(DISTINCT id(n) + id(r) + id(m)) AS unique_triples, 
     count(r) AS total_triples, 
     count(DISTINCT n) AS unique_nodes, 
     count(DISTINCT m) AS unique_target_nodes
WITH source, unique_nodes + unique_target_nodes AS total_nodes, unique_nodes, unique_triples, total_triples
WITH source, total_nodes, unique_nodes, unique_triples, total_triples,
     CASE
         WHEN total_nodes > 1 THEN unique_triples * 1.0 / (total_nodes * (total_nodes - 1) / 2)
         ELSE 0
     END AS density
RETURN source, total_nodes, unique_nodes, unique_triples, total_triples, density
ORDER BY source

"""

#for primekg
"""  
MATCH (n)-[r]-(m)
WITH 
     count(DISTINCT id(n) + id(r) + id(m)) AS unique_triples, 
     count(r) AS total_triples, 
     count(DISTINCT n) AS unique_nodes, 
     count(DISTINCT m) AS unique_target_nodes
WITH unique_nodes + unique_target_nodes AS total_nodes, unique_nodes, unique_triples, total_triples
WITH total_nodes, unique_nodes, unique_triples, total_triples,
     CASE
         WHEN total_nodes > 1 THEN unique_triples * 1.0 / (total_nodes * (total_nodes - 1) / 2)
         ELSE 0
     END AS density
RETURN total_nodes, unique_nodes, unique_triples, total_triples, density
ORDER BY total_nodes DESC

"""


# # COVID NDD triples for SCAI (later added to comorbidity database)

# In[ ]:


"""  
WITH ['alzheimer', 'parkinson', 'neurodegenerative', 'covid', 'sars', "cov", 'neurodegen', 'inflamamtion'] AS keywords
MATCH (d)
WHERE any(keyword IN keywords WHERE toLower(d.name) CONTAINS toLower(keyword))
WITH d
MATCH (d1)-[r*..1]-(d2)
WHERE id(d1) < id(d2)
AND d1.name IS NOT NULL 
AND d2.name IS NOT NULL
AND all(rel IN r WHERE toLower(rel.source) CONTAINS 'scai')
AND all(rel IN r WHERE rel.pmid IS NOT NULL)
AND all(rel IN r WHERE rel.evidence IS NOT NULL)
AND any(rel IN r WHERE 
    toLower(rel.evidence) CONTAINS 'covid' OR 
    toLower(rel.evidence) CONTAINS 'viral infection' OR
    toLower(rel.evidence) CONTAINS 'sars cov' OR
    toLower(rel.evidence) CONTAINS 'neuro')  // Filter for evidence containing relevant terms
UNWIND r AS rel
RETURN DISTINCT
   d1.name AS Subject,
   d2.name AS Object,
   type(rel) AS Relation,
   rel.pmid AS PMID,
   rel.evidence AS Evidence,
   rel.source AS Source

"""


# In[2]:


#correct general statistics above for undirected graphs
""" 
WITH ['Sherpa', 'PubTator', 'CBM', 'scai', 'opentargets', 'disgenet', 'indra', 'drugbank', 'kegg'] AS sources
UNWIND sources AS source
MATCH (n)-[r]-(m)
WHERE toLower(r.source) CONTAINS toLower(source) AND id(n) < id(m)
WITH source, 
     count(DISTINCT id(n) + id(r) + id(m)) AS unique_triples, 
     count(r) AS total_triples, 
     collect(DISTINCT n) + collect(DISTINCT m) AS unique_nodes,
     collect(n) + collect(m) AS total_nodes
WITH source, 
     unique_triples, 
     total_triples,
     size(apoc.coll.toSet(unique_nodes)) AS unique_node_count,
     size(total_nodes) AS total_node_count
WITH source, unique_node_count, total_node_count, unique_triples, total_triples,
     CASE
         WHEN unique_node_count > 1 THEN unique_triples * 2.0 / (unique_node_count * (unique_node_count - 1))
         ELSE 0
     END AS density
RETURN source, unique_node_count, total_node_count, unique_triples, total_triples, density
ORDER BY source

""" 

#this is for directed graphs (SHERPA, CBM, SCAI, INDRA, KEGG (pathology))
""" 
WITH ['Sherpa', 'PubTator', 'CBM', 'scai', 'opentargets', 'disgenet', 'indra', 'drugbank', 'kegg'] AS sources
UNWIND sources AS source
MATCH (n)-[r]->(m)
WHERE toLower(r.source) CONTAINS toLower(source)
WITH source, 
     count(DISTINCT id(n) + id(r) + id(m)) AS unique_triples, 
     count(r) AS total_triples, 
     collect(DISTINCT n) + collect(DISTINCT m) AS unique_nodes,
     collect(n) + collect(m) AS total_nodes
WITH source, 
     unique_triples, 
     total_triples,
     size(apoc.coll.toSet(unique_nodes)) AS unique_node_count,
     size(total_nodes) AS total_node_count
WITH source, unique_node_count, total_node_count, unique_triples, total_triples,
     CASE
         WHEN unique_node_count > 1 THEN unique_triples * 1.0 / (unique_node_count * (unique_node_count - 1))
         ELSE 0
     END AS density
RETURN source, unique_node_count, total_node_count, unique_triples, total_triples, density
ORDER BY source

""" 

#For primekg (old incorrect)
"""  


MATCH (n)-[r]->(m)
WITH 
     count(DISTINCT id(n) + id(r) + id(m)) AS unique_triples, 
     count(r) AS total_triples, 
     collect(DISTINCT n) + collect(DISTINCT m) AS unique_nodes,
     collect(n) + collect(m) AS total_nodes
WITH 
     unique_triples, 
     total_triples,
     size(apoc.coll.toSet(unique_nodes)) AS unique_node_count,
     size(total_nodes) AS total_node_count
WITH unique_node_count, total_node_count, unique_triples, total_triples,
     CASE
         WHEN unique_node_count > 1 THEN unique_triples * 1.0 / (unique_node_count * (unique_node_count - 1))
         ELSE 0
     END AS density
RETURN unique_node_count, total_node_count, unique_triples, total_triples, density
ORDER BY unique_node_count DESC


"""

#For primekg corrected

"""  
MATCH (n)-[r]->(m)
WHERE any(term IN ["covid", "alzheimer", "neurodegeneration", "parkinson"] 
           WHERE n.name =~ ('(?i).*' + term + '.*') 
              OR m.name =~ ('(?i).*' + term + '.*'))
WITH 
     count(DISTINCT id(n) + id(r) + id(m)) AS unique_triples, 
     count(r) AS total_triples, 
     collect(DISTINCT n) + collect(DISTINCT m) AS unique_nodes,
     collect(n) + collect(m) AS total_nodes
WITH 
     unique_triples, 
     total_triples,
     size(apoc.coll.toSet(unique_nodes)) AS unique_node_count,
     size(total_nodes) AS total_node_count
WITH unique_node_count, total_node_count, unique_triples, total_triples,
     CASE
         WHEN unique_node_count > 1 THEN unique_triples * 1.0 / (unique_node_count * (unique_node_count - 1))
         ELSE 0
     END AS density
RETURN unique_node_count, total_node_count, unique_triples, total_triples, density
ORDER BY unique_node_count DESC
"""


# In[7]:


##visualize above using an excel file of the graph statistics 
import pandas as pd
import matplotlib.pyplot as plt

# Updated data based on the new input
data_updated = {
    "source": ["CBM", "DisGeNet", "DrugBank", "INDRA", "OpenTargets", "PubTator3", "SCAI-DMaps", "Sherpa", "PrimeKG"],
    "total_nodes": [4020, 1176, 128, 2004, 31644, 164, 13676, 1516, 258750],
    "unique_nodes": [2010, 588, 64, 1002, 15822, 82, 6838, 758, 129375],
    "unique_triples": [3484, 1048, 710, 1224, 29953, 75, 12302, 1757, 4960573],
    "total_triples": [7843, 2590, 1456, 3049, 72420, 156, 30597, 5018, 16200996],
    "density": [0.000431285, 0.001516862, 0.087352362, 0.000609864, 5.98276E-05, 0.005611252, 0.000131559, 0.001529995, 0.000148185]
}

# Create DataFrame with corrected names
df_updated = pd.DataFrame(data_updated)

# Sort the DataFrame by density in decreasing order
df_sorted_updated = df_updated.sort_values(by="density", ascending=False)

# Set the figure size
plt.figure(figsize=(18, 12))

# Bar plot for total_nodes, unique_nodes, unique_triples, and total_triples sorted by density with logarithmic scale
plt.subplot(2, 2, 1)
df_sorted_updated.plot(x="source", y=["total_nodes", "unique_nodes", "unique_triples", "total_triples"], 
                       kind="bar", ax=plt.gca(), log=True)
plt.title("Total Nodes, Unique Nodes, Unique Triples, Total Triples (Sorted by Density, Log Scale)")
plt.ylabel("Log Count")
plt.xticks(rotation=45)

# Scatter plot for density vs total_nodes (sorted by density)
plt.subplot(2, 2, 2)
plt.scatter(df_sorted_updated["total_nodes"], df_sorted_updated["density"], label="Density vs Total Nodes", color="b")
plt.xlabel("Total Nodes")
plt.ylabel("Density")
plt.title("Density vs Total Nodes (Sorted by Density)")
for i, txt in enumerate(df_sorted_updated["source"]):
    plt.annotate(txt, (df_sorted_updated["total_nodes"].iloc[i], df_sorted_updated["density"].iloc[i]))

# Stacked bar chart for unique_triples and redundant triples sorted by density with logarithmic scale
plt.subplot(2, 2, 3)
df_sorted_updated["redundant_triples"] = df_sorted_updated["total_triples"] - df_sorted_updated["unique_triples"]
df_sorted_updated.plot(x="source", y=["unique_triples", "redundant_triples"], kind="bar", stacked=True, ax=plt.gca(), log=True)
plt.title("Unique vs Redundant Triples (Sorted by Density, Log Scale)")
plt.ylabel("Log Count")
plt.xticks(rotation=45)

# Heatmap for density sorted by density
plt.subplot(2, 2, 4)
plt.bar(df_sorted_updated["source"], df_sorted_updated["density"], color="hotpink")
plt.title("Density Heatmap by Source (Sorted by Density)")
plt.xticks(rotation=45)
plt.ylabel("Density")

# Adjust layout for readability
plt.tight_layout()

# Show the plots
plt.show()



# In[ ]:


# graph density for each source
"""MATCH (n)-[r]-(m)
WHERE r.source CONTAINS 'sherpa'
WITH count(DISTINCT n) + count(DISTINCT m) AS total_nodes, count(r) AS total_relationships
WITH total_nodes, total_relationships,
     CASE
         WHEN total_nodes > 1 THEN total_relationships * 1.0 / (total_nodes * (total_nodes - 1) / 2)
         ELSE 0
     END AS density
RETURN total_nodes, total_relationships, density
"""


# ## general: not defining te source

# In[ ]:


""" 
WITH ['alzheimer', 'parkinson', 'neurodegenerative', 'covid','neurodegeneration'] AS keywords
MATCH (d)
WHERE any(keyword IN keywords WHERE toLower(d.name) CONTAINS keyword)
WITH collect(d) AS diseases

// Step 2: Generate unique pairs
UNWIND diseases AS d1
UNWIND diseases AS d2
WITH d1, d2
WHERE id(d1) < id(d2)  // Ensure unique pairs and avoid self-matches

// Step 3: Compute shortest path
MATCH path = shortestPath((d1)-[*..3]-(d2)) // Find shortest path within 3 hops
RETURN DISTINCT path
"""


#above but defining where source is coming from
"""

WITH ['alzheimer', 'parkinson', 'neurodegenerative', 'covid', 'neurodegeneration'] AS keywords
MATCH (d)
WHERE any(keyword IN keywords WHERE toLower(d.name) CONTAINS keyword)
WITH collect(d) AS diseases

// Step 2: Generate unique pairs
UNWIND diseases AS d1
UNWIND diseases AS d2
WITH d1, d2
WHERE id(d1) < id(d2)  // Ensure unique pairs and avoid self-matches

// Step 3: Compute shortest path with filtered relationships
MATCH path = shortestPath((d1)-[r*..3]-(d2)) // Find shortest path within 3 hops
WHERE all(rel IN relationships(path) WHERE 
    any(source_keyword IN ['opentargets', 'disgenet', 'drugbank', 'indra'] 
        WHERE toLower(rel.source) CONTAINS source_keyword)) // Check if source contains any of the target keywords
RETURN DISTINCT path


"""


# above but get paths as list of ndoeds and rels when defining source type
"""
WITH ['alzheimer', 'parkinson', 'neurodegenerative', 'covid', 'neurodegeneration'] AS keywords
MATCH (d)
WHERE any(keyword IN keywords WHERE toLower(d.name) CONTAINS keyword)
WITH collect(d) AS diseases

// Step 2: Generate unique pairs
UNWIND diseases AS d1
UNWIND diseases AS d2
WITH d1, d2 WHERE id(d1) < id(d2)  // Ensure unique pairs and avoid self-matches

// Step 3: Compute shortest path with filtered relationships
MATCH path = shortestPath((d1)-[r*..3]-(d2)) // Find shortest path within 3 hops
WHERE all(rel IN relationships(path) WHERE 
    any(source_keyword IN ['opentargets', 'disgenet', 'drugbank', 'indra'] 
        WHERE toLower(rel.source) CONTAINS source_keyword)) // Filter by source

// Step 4: Extract path nodes and relationships
WITH DISTINCT [node IN nodes(path) | id(node)] AS pathNodes, 
                [rel IN relationships(path) | type(rel)] AS pathRelationships

// Step 5: Retrieve node details based on ID
UNWIND pathNodes AS nodeId
MATCH (n)
WHERE id(n) = nodeId
WITH pathNodes, pathRelationships, n, 
     CASE WHEN 'name' IN keys(n) THEN n.name ELSE n.symbol END AS nodeName

// Collect the node names back into a list
WITH DISTINCT pathNodes, pathRelationships, collect(nodeName) AS nodeNames

// Return the lists of node names and relationship types
RETURN nodeNames AS pathNodes, pathRelationships
"""


# ### Get paths as list of nodes and rels

# In[ ]:


"""
WITH ['alzheimer', 'parkinson', 'neurodegenerative', 'covid', 'neurodegeneration'] AS keywords
MATCH (d)
WHERE any(keyword IN keywords WHERE toLower(d.name) CONTAINS keyword)
WITH collect(d) AS diseases

// Step 2: Generate unique pairs
UNWIND diseases AS d1
UNWIND diseases AS d2
WITH d1, d2 WHERE id(d1) < id(d2)  // Ensure unique pairs and avoid self-matches

// Step 3: Compute shortest path
MATCH path = shortestPath((d1)-[*..3]-(d2)) // Find shortest path within 3 hops
WITH DISTINCT [node IN nodes(path) | id(node)] AS pathNodes, 
                [rel IN relationships(path) | type(rel)] AS pathRelationships

// Step 4: Retrieve node details based on ID
UNWIND pathNodes AS nodeId
MATCH (n)
WHERE id(n) = nodeId
WITH pathNodes, pathRelationships, n, 
     CASE WHEN 'name' IN keys(n) THEN n.name ELSE n.symbol END AS nodeName

// Collect the node names back into a list
WITH DISTINCT pathNodes, pathRelationships, collect(nodeName) AS nodeNames

// Return the lists of node names and relationship types
RETURN nodeNames AS pathNodes, pathRelationships
"""


# In[10]:


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


# In[16]:


# Read all json paths and compare them all as a common path
import json
import glob

# Function to load data from a JSON file
def load_results_from_file(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {filename}")
        return []

# Function to load and aggregate data from multiple JSON files
def load_all_json_files(file_pattern):
    all_results = []
    for filename in glob.glob(file_pattern):
        print(f"Loading data from {filename}...")
        data = load_results_from_file(filename)
        if data:
            all_results.append(data)  # Do not flatten, keep each file's data separate
    return all_results

# Function to compare paths between different graphs
def compare_paths(paths_list):
    if not paths_list:
        print("No paths to compare.")
        return set()
    
    # Convert paths to sets of tuples to ensure unique paths and to compare them
    path_sets = [set(tuple(map(tuple, path)) for path in paths) for paths in paths_list]
    
    # Find the intersection of all path sets
    common_paths = set.intersection(*path_sets)

    # Convert the set of tuples back to lists
    common_paths = [list(map(list, path)) for path in common_paths]

    return common_paths

# Load data from multiple JSON files
file_pattern = 'neo4j_results_*.json'  # Adjust the pattern to match your files
all_neo4j_results = load_all_json_files(file_pattern)

if not all_neo4j_results:
    print("No results loaded from files.")
else:
    # Use the compare_paths function on the loaded results
    common_paths = compare_paths(all_neo4j_results)
    
    print(f"Common paths across all graphs: {len(common_paths)}")


# In[4]:


 #Get the paths that are unique for one json file (give a name) comapred to all paths
import json
import glob

# Function to load data from a JSON file
def load_results_from_file(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {filename}")
        return []

# Function to load and aggregate data from multiple JSON files
def load_all_json_files(file_pattern):
    all_results = {}
    for filename in glob.glob(file_pattern):
        print(f"Loading data from {filename}...")
        data = load_results_from_file(filename)
        if data:
            all_results[filename] = data  # Store each file's data with its filename as key
    return all_results

# Function to find paths unique to a specific JSON file
def find_unique_paths(target_filename, paths_dict):
    if target_filename not in paths_dict:
        print(f"File {target_filename} not found in loaded data.")
        return []
    
    # Convert paths to sets of tuples to ensure unique paths and to compare them
    target_file_paths = set(tuple(map(tuple, path)) for path in paths_dict[target_filename])
    
    # Combine paths from all other files
    other_files_paths = set()
    for filename, paths in paths_dict.items():
        if filename != target_filename:
            other_files_paths.update(tuple(map(tuple, path)) for path in paths)
    
    # Find paths unique to the target file by subtracting the other paths
    unique_paths = target_file_paths - other_files_paths

    # Convert the set of tuples back to lists
    unique_paths = [list(map(list, path)) for path in unique_paths]

    return unique_paths

# Load data from multiple JSON files
file_pattern = 'neo4j_results_*.json'  # Adjust the pattern to match your files
all_neo4j_results = load_all_json_files(file_pattern)

if not all_neo4j_results:
    print("No results loaded from files.")
else:
    # Specify the filename for which you want to find unique paths
    target_filename = 'neo4j_results_2.json'  # Replace with the specific filename
    
    # Use the find_unique_paths function on the loaded results
    unique_paths = find_unique_paths(target_filename, all_neo4j_results)
    
    print(f"Unique paths in the file {target_filename}: {unique_paths}")


# In[6]:


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



# In[ ]:


#Draw bar chart for 10 top genes by node degree  

from neo4j import GraphDatabase
import pandas as pd
import matplotlib.pyplot as plt

# Neo4j connection details
uri = "bolt://localhost:7687"  # Update this with your Neo4j connection URI
user = "neo4j"  # Update this with your Neo4j username
password = "12345678"  # Update this with your Neo4j password

# Connect to Neo4j
driver = GraphDatabase.driver(uri, auth=(user, password))

# Cypher query to retrieve the genes and their degree
query = """
MATCH (g)-[r]->()
WHERE g.namespace = "HGNC" AND "sherpa" IN r.annotationDatasource
RETURN DISTINCT g.name AS GeneName, COUNT(r) AS Degree
ORDER BY Degree DESC
"""

# query = """MATCH (g:Gene)-[r]->()
# WHERE r.source = "KEGG"
# RETURN DISTINCT g.name AS GeneName, COUNT(r) AS Degree
# ORDER BY Degree DESC
# """

# Function to execute query and return data as DataFrame
def query_neo4j(query):
    with driver.session() as session:
        result = session.run(query)
        records = result.data()
        # Convert result to a pandas DataFrame
        df = pd.DataFrame(records)
    return df

# Execute the query and get the result as a DataFrame
df = query_neo4j(query)

# Sort the DataFrame and get the top 10 genes by degree
df_top10 = df.head(10)

# Plotting the bar chart
plt.figure(figsize=(10, 6))
plt.bar(df_top10['GeneName'], df_top10['Degree'], color='blue')

# Add labels and title
plt.xlabel('Gene Name')
plt.ylabel('Degree')
plt.title('Top 10 Genes by Degree (Source: Sherpa)')

# Rotate the x labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()

# Close Neo4j connection
driver.close()


# In[11]:


#COMMON GENE nodes between KEGG and sherpa
from neo4j import GraphDatabase
from matplotlib_venn import venn2
import matplotlib.pyplot as plt

# Neo4j connection details
uri = "bolt://localhost:7687"  # Update this with your Neo4j connection URI
user = "neo4j"  # Update this with your Neo4j username
password = "12345678"  # Update this with your Neo4j password

# Connect to Neo4j
driver = GraphDatabase.driver(uri, auth=(user, password))

# Function to execute a query and return a single result
def query_neo4j(query):
    with driver.session() as session:
        result = session.run(query)
        return result.single()[0]

# Query to get Sherpa node count (only Gene nodes)
sherpa_count_query = """
MATCH (n)
WHERE 'sherpa' IN n.source and n.namespace = "HGNC"
RETURN COUNT(DISTINCT n.name) AS Sherpa_Node_Count
"""

# Query to get KEGG node count (only Gene nodes)
kegg_count_query = """
MATCH (n:Gene)
WHERE n.source = 'KEGG'
RETURN COUNT(DISTINCT n.name) AS KEGG_Node_Count
"""

# Query to get intersection count (only Gene nodes)
intersection_query = """
MATCH (n)
WHERE 'sherpa' IN n.source and n.namespace = 'HGNC'
WITH COLLECT(DISTINCT n.name) AS sherpa_genes
MATCH (m:Gene)
WHERE m.source = 'KEGG'
WITH sherpa_genes, COLLECT(DISTINCT m.name) AS kegg_genes
RETURN SIZE(apoc.coll.intersection(sherpa_genes, kegg_genes)) AS Intersection_Count
"""

# Execute queries
sherpa_count = query_neo4j(sherpa_count_query)
kegg_count = query_neo4j(kegg_count_query)
intersection_count = query_neo4j(intersection_query)

# Create a Venn diagram
plt.figure(figsize=(8, 6))
venn2(subsets=(sherpa_count - intersection_count, kegg_count - intersection_count, intersection_count),
      set_labels=('Sherpa', 'KEGG'))

# Set the title
plt.title('Venn Diagram of Sherpa and KEGG Gene Node Counts with Intersection')

# Display the Venn diagram
plt.show()

# Close the Neo4j driver connection
driver.close()


# In[ ]:


get_ipython().system('pip install upsetplot')


# ## Assign node source tag from rel tag

# In[ ]:


"""// Assign source from relationships to connected nodes
MATCH (n)-[r]->(m)
WHERE r.source IS NOT NULL  // Only consider relationships with a source property
SET n.source = r.source,
    m.source = r.source
RETURN DISTINCT n, m"""


# ### Get all triples about all diseases

# In[1]:


"""
MATCH (disease:Disease)-[r]->(related)
WHERE toLower(disease.name) CONTAINS 'alzheimer' OR 
     toLower(disease.name)  CONTAINS 'covid' OR 
      toLower(disease.name)  CONTAINS 'parkinson' OR 
      toLower(disease.name)  CONTAINS 'neurodegeneration'
RETURN disease.name AS Disease, type(r) AS RelationshipType, related AS RelatedNode

"""

"""
#rethink above?
MATCH (disease:Disease)-[r]->(related)
WHERE (toLower(disease.name) CONTAINS 'alzheimer' OR 
       toLower(disease.name) CONTAINS 'covid' OR 
       toLower(disease.name) CONTAINS 'parkinson' OR 
       toLower(disease.name) CONTAINS 'neurodegeneration') OR
      (toLower(related.name) CONTAINS 'alzheimer' OR 
       toLower(related.name) CONTAINS 'covid' OR 
       toLower(related.name) CONTAINS 'parkinson' OR 
       toLower(related.name) CONTAINS 'neurodegeneration')
RETURN disease.name AS Disease, type(r) AS RelationshipType, related
"""


# #### DISGENET 

# In[ ]:


#cypher queris to get common rels between two graphS:
#covid and ad
cypher_disgenet = """
MATCH (d1:Disease)-[r]->(g)<-[r2]-(d2:Disease)
WHERE d1.name CONTAINS 'COVID' AND d2.name CONTAINS 'Alzheimer'
RETURN d1 AS disease1, r AS relationship1, g AS gene, r2 AS relationship2, d2 AS disease2
"""
#covid and all diseases
"""
MATCH (d1:Disease)-[r]->(g)<-[r2]-(d2:Disease)
WHERE (d1.name CONTAINS 'COVID' OR d1.name CONTAINS 'Alzheimer' OR d1.name CONTAINS 'Parkinson' OR d1.name CONTAINS 'neurodegenerative')
  AND (d2.name CONTAINS 'COVID' OR d2.name CONTAINS 'Alzheimer' OR d2.name CONTAINS 'Parkinson' OR d2.name CONTAINS 'neurodegenerative')
  AND d1 <> d2  // Ensure that d1 and d2 are different nodes
RETURN d1 AS disease1, r AS relationship1, g AS gene, r2 AS relationship2, d2 AS disease2

"""

#cypher filter common genes based on score
"""MATCH (d1:Disease)-[r]->(g)<-[r2]-(d2:Disease)
WHERE (d1.name CONTAINS 'COVID' OR d1.name CONTAINS 'Alzheimer' OR d1.name CONTAINS 'Parkinson' OR d1.name CONTAINS 'neurodegenerative')
  AND (d2.name CONTAINS 'COVID' OR d2.name CONTAINS 'Alzheimer' OR d2.name CONTAINS 'Parkinson' OR d2.name CONTAINS 'neurodegenerative')
  AND d1 <> d2  // Ensure that d1 and d2 are different nodes
  AND r.score is not null  // Ensure the score property exists
  AND r.score > 0.75 // Filter by association score
RETURN d1 AS disease1, r AS relationship1, g AS gene, r2 AS relationship2, d2 AS disease2, r.score AS score
"""

cypher = """

MATCH (d1:Disease {name: 'Covid-19'})-[r1:RELATED_TO]->(g:Gene)<-[r2:RELATED_TO]-(d2:Disease {name: "Alzheimer's disease"})
RETURN d1, r1, g, r2, d2
"""

#get common rels between four disease
cypher = """MATCH (d1:Disease {name: 'Covid-19'})-[r1:RELATED_TO]->(g:Gene)
WHERE (d1)-[:RELATED_TO]->(g)
MATCH (d2:Disease {name: "Alzheimer's disease"})-[r2:RELATED_TO]->(g)
WHERE (d2)-[:RELATED_TO]->(g)
MATCH (d3:Disease {name: 'Parkinson Disease'})-[r3:RELATED_TO]->(g)
WHERE (d3)-[:RELATED_TO]->(g)
MATCH (d4:Disease {name: 'Neurodegenerative Diseases'})-[r4:RELATED_TO]->(g)
WHERE (d4)-[:RELATED_TO]->(g)
RETURN d1,d2,d3,d4,g,
       r1, r2, r3, r4"""

"""
MATCH (d:Disease)-[r]->(g:Gene)
WHERE d.name CONTAINS 'COVID'
RETURN count(distinct(g))
"""


# ### DiseaseDb

# In[ ]:


"""MATCH (d1:Disease)-[r]->(g)<-[r2]-(d2:Disease)
WHERE (d1.name CONTAINS 'COVID' OR d1.name CONTAINS 'Alzheimer' OR d1.name CONTAINS 'Parkinson' OR d1.name CONTAINS 'Neurodegenerative')
  AND (d2.name CONTAINS 'COVID' OR d2.name CONTAINS 'Alzheimer' OR d2.name CONTAINS 'Parkinson' OR d2.name CONTAINS 'Neurodegenerative')
RETURN d1 AS disease1, r AS relationship1, g AS gene, r2 AS relationship2, d2 AS disease2"""

###to delete everything in batch
"""CALL apoc.periodic.iterate(
  'MATCH (n) RETURN n',
  'DETACH DELETE n',
  {batchSize: 1000}
)"""


# ### CBM

# ### PrimeKG

# In[ ]:


""" Only triples about COVID AD PD and NDD

WITH ["covid", "sars", "alzheimer", "parkinson", "neuro", "dementia"] AS keywords
MATCH (n)-[r]->(m)
WHERE ANY(keyword IN keywords WHERE toLower(n.name) CONTAINS keyword)
   OR ANY(keyword IN keywords WHERE toLower(m.name) CONTAINS keyword)
RETURN n.name AS Subject, type(r) AS Relation, m.name AS Object


"""


#improve above and extend keyword search (old not in paper)

""" 
WITH [
    "covid", "sars", "coronavirus", "post-covid", "long covid", "SARS-CoV-2",
    "encephalopathy", "myopathy", "brain fog", "fatigue", "hypoxia", "cytokine storm",
    "stroke", "guillain-barré", "neuroinflammation", "viral infection", "ARDS",
    "pasc", "neurodegeneration", "neurodegenerative", "alzheimer", "parkinson",
    "dementia", "cognitive decline", "memory loss", "ALS", "amyotrophic lateral sclerosis",
    "huntington", "multiple sclerosis", "lewy body dementia", "frontotemporal dementia",
    "PSP", "progressive supranuclear palsy", "CBD", "corticobasal degeneration",
    "CJD", "creutzfeldt-jakob", "motor neuron disease", "ataxia", "prion disease",
    "chorea", "myoclonus", "viral-induced neurodegeneration", "neurological decline",
    "neurotropism", "neuroinflammatory response", "axonal degeneration"
] AS keywords
MATCH (n)-[r]-(m)
WHERE ANY(keyword IN keywords WHERE toLower(n.name) CONTAINS keyword)
   OR ANY(keyword IN keywords WHERE toLower(m.name) CONTAINS keyword)
RETURN n.name AS Subject, type(r) AS Relation, m.name AS Object


"""

#get covid-NDD triples only direct interaction: final in paper
"""  

WITH [
   // Core COVID-19 terms
   "SARS-CoV-2", "COVID-19", "2019-nCoV", "coronavirus", "SARS", 
   "post-COVID", "long COVID", "PASC",
   
   // COVID-19 molecular & mechanism
   "ACE2", "TMPRSS2", "spike protein", "viral pneumonia",
   "cytokine storm", "viral infection", "ARDS",
   "respiratory infection", "hypoxia",
   
   // Neurological manifestations of COVID
   "brain fog", "fatigue", "encephalopathy",
   "guillain-barré", "stroke", "myopathy",
   "neuroinflammation", "neurological decline",
   "neuroinflammatory response", "neurotropism",
   
   // Core neurodegenerative diseases
   "Alzheimer", "Parkinson", "Huntington",
   "ALS", "amyotrophic lateral sclerosis",
   "multiple sclerosis", "prion disease",
   "dementia", "neurodegeneration", "neurodegenerative",
   
   // Specific neurodegenerative conditions
   "Lewy body dementia", "frontotemporal dementia",
   "PSP", "progressive supranuclear palsy",
   "CBD", "corticobasal degeneration",
   "CJD", "Creutzfeldt-Jakob",
   "motor neuron disease",
   
   // Symptoms & manifestations
   "cognitive decline", "memory loss", "ataxia",
   "chorea", "myoclonus", "axonal degeneration",
   
   // Molecular mechanisms
   "tauopathy", "synucleinopathy", "proteinopathy"
] AS keywords

// First match potential nodes
MATCH (n)
WHERE ANY(keyword IN keywords WHERE toLower(n.name) CONTAINS toLower(keyword))

// Pass the matched nodes
WITH COLLECT(DISTINCT n) as nodes, keywords

// Then expand to relationships with matched nodes
MATCH (n)-[r]-(m)
WHERE n IN nodes
AND ANY(keyword IN keywords WHERE toLower(m.name) CONTAINS toLower(keyword))

WITH 
   COUNT(DISTINCT n) + COUNT(DISTINCT m) as total_nodes,
   COUNT(DISTINCT r) as total_edges,
   COLLECT(DISTINCT n) + COLLECT(DISTINCT m) as all_nodes,
   COLLECT(DISTINCT r) as all_edges

RETURN 
   total_nodes as TotalNodes,
   total_edges as TotalEdges,
   CASE 
       WHEN total_nodes <= 1 THEN 0
       ELSE toFloat(total_edges) / (total_nodes * (total_nodes - 1))
   END as GraphDensity
"""
#inderct interaction included as well to get covid-NDD tripels from primekg

"""  
WITH [
    // Core COVID-19 terms - reduced to most specific
    "SARS-CoV-2", "COVID-19", "coronavirus", "SARS", 
    
    // Core neurodegenerative diseases - reduced to most specific
    "Alzheimer", "Parkinson", "Huntington",
    "ALS", "multiple sclerosis", "dementia",
    "neurodegeneration"
] AS keywords

// First find nodes matching keywords
MATCH (n)
WHERE ANY(keyword IN keywords WHERE toLower(n.name) CONTAINS toLower(keyword))

// Collect these as starting points
WITH COLLECT(DISTINCT n) as starting_nodes, keywords

// Find direct and one-hop connections
MATCH path = (n)-[r]-(intermediate)-[r2]-(m)
WHERE 
    n IN starting_nodes
    AND m IN starting_nodes
    AND n <> m  // Ensure we don't match same node

RETURN 
    COUNT(DISTINCT n) + COUNT(DISTINCT m) as TotalNodes,
    COUNT(DISTINCT r) + COUNT(DISTINCT r2) as TotalEdges,
    CASE 
        WHEN COUNT(DISTINCT n) + COUNT(DISTINCT m) <= 1 THEN 0
        ELSE toFloat(COUNT(DISTINCT r) + COUNT(DISTINCT r2)) / ((COUNT(DISTINCT n) + COUNT(DISTINCT m)) * (COUNT(DISTINCT n) + COUNT(DISTINCT m) - 1))
    END as GraphDensity
LIMIT 1000

"""


# In[5]:


# Complete code for reading the covid-ndd PRIMEKG file, calculating statistics, and generating charts

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Function to clean and extract the first part of the node label
def clean_label(text):
    if isinstance(text, str) and '[' in text:
        return text.strip('[]"').split(",")[0].strip()
    return text

# Read the updated file with correct column names
file_path = 'data/primekg/final-primekg-covid-ndd.csv'
data = pd.read_csv(file_path)

# Clean and extract labels for Subject and Object
data['Subject_Label'] = data['Subj_label'].apply(clean_label)
data['Object_Label'] = data['Obj_label'].apply(clean_label)

# Calculate node label distribution
combined_labels = pd.concat([data['Subject_Label'], data['Object_Label']])
node_label_distribution = combined_labels.value_counts()

# Calculate relationship distribution
relation_distribution = data['Relation'].value_counts()

# Calculate degree centrality
node_degrees = Counter(data['Subject'].dropna()) + Counter(data['Object'].dropna())
top_10_high_degree_nodes = node_degrees.most_common(10)
top_10_high_degree_df = pd.DataFrame(top_10_high_degree_nodes, columns=["Node", "Degree"])

# Count number of unique nodes and triples
unique_nodes = len(set(data['Subject'].dropna()).union(set(data['Object'].dropna())))
total_triples = len(data)

# Calculate graph density
graph_density = 2 * total_triples / (unique_nodes * (unique_nodes - 1))

# Display statistics
print(f"Total unique nodes: {unique_nodes}")
print(f"Total number of triples: {total_triples}")
print(f"Graph density: {graph_density:.6f}")

# Plot charts
# 1. Node Label Distribution
plt.figure(figsize=(10, 6))
node_label_distribution.plot(kind='bar', color='lightblue', alpha=0.8)
plt.title("Node Types and Their Frequencies for PRIMEKG")
plt.xlabel("Node Type")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 2. Relationship Distribution
plt.figure(figsize=(10, 6))
relation_distribution.plot(kind='bar', color='green', alpha=0.8)
plt.title("Relationship Types and Their Frequencies for PRIMEKG")
plt.xlabel("Relationship Type")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 3. Top 10 High Degree Centrality Nodes
plt.figure(figsize=(10, 6))
top_10_high_degree_df.plot(kind='bar', x='Node', y='Degree', legend=False, color='skyblue', alpha=0.8)
plt.title("Top 10 Degree Centrality for PRIMEKG")
plt.xlabel("Node")
plt.ylabel("Degree Centrality")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# In[4]:


#query disease ids from ols
import requests

# Define the base API endpoint
BASE_URL = "https://www.ebi.ac.uk/ols/api/search"

# Function to search for MONDO IDs for a given disease name
def search_mondo_ids(disease_name):
    params = {
        "q": disease_name,  # Query term (disease name)
        "ontology": "mondo",  # Specify the ontology
        "type": "class",  # Search for classes (terms)
    }
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json().get("response", {}).get("docs", [])
        ids = []
        for result in results:
            mondo_id = result.get("obo_id")
            label = result.get("label")
            if mondo_id and label:
                ids.append({"id": mondo_id, "label": label})
        return ids
    else:
        print(f"Failed to fetch data for {disease_name}: {response.status_code}")
        return []

# List of diseases to search
diseases = [
    "Neurodegenerative disease",
    "Alzheimer's disease",
    "Parkinson's disease",
    "Amyotrophic lateral sclerosis",
    "Multiple sclerosis",
    "Huntington's disease",
    "Lewy body dementia",
    "Frontotemporal dementia",
    "Creutzfeldt-Jakob disease",
    "Progressive supranuclear palsy",
    "COVID-19",
    "Post-COVID syndrome",
]

# Search for MONDO IDs for each disease
all_mondo_ids = []
for disease in diseases:
    print(f"Searching for MONDO IDs for: {disease}")
    ids = search_mondo_ids(disease)
    all_mondo_ids.extend(ids)

# Print the results
print("\nRetrieved MONDO IDs:")
for item in all_mondo_ids:
    print(f"{item['label']}: {item['id']}")

# Optional: Save the results to a file
import csv
with open("mondo_ids.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["label", "id"])
    writer.writeheader()
    writer.writerows(all_mondo_ids)
print("\nResults saved to 'mondo_ids.csv'")


# # Pathophisiologal Mechanism

# In[1]:


"""
WITH ["neuroinflammation", "oxidative stress", "mitochondrial dysfunction", "blood-brain barrier disruption", "amyloid", "tau pathology", "autonomic dysfunction", "neurotropism"] AS keywords
     
MATCH p=(n)-[*1..5]-(m) 
WHERE ANY(keyword IN keywords WHERE toLower(n.name) CONTAINS toLower(keyword) OR toLower(m.name) CONTAINS toLower(keyword))
RETURN p
LIMIT 50

"""
#efficient: mainly gives text minign triples

"""
WITH ["neuroinflammation", "inflammation", "neuroimmune response", "neuroimmunology", "brain inflammation",
      "oxidative stress", "free radicals", "reactive oxygen species", "ROS", "oxidative damage",
      "mitochondrial", "mitochondrial dysfunction", "mitochondrial damage", "mitochondrial health", "energy production",
      "blood-brain barrier", "brain barrier", "BBB", "neurovascular unit", "blood-brain permeability",
      "amyloid", "amyloid plaques", "beta-amyloid", "amyloid beta", "plaques in Alzheimer's",
      "tau", "tau protein", "tau tangles", "tauopathies", "tau neurofibrillary tangles",
      "autonomic dysfunction", "autonomic nervous system", "ANS", "autonomic regulation", "autonomic imbalance",
      "neurotropism", "neuronal targeting", "neurotropic", "neurotropic viruses", "neurotropic diseases"]AS keywords
UNWIND keywords AS keyword
WITH DISTINCT toLower(keyword) AS lower_keyword
MATCH (n)-[r*1..3]-(m)  // Depth limited to 1..3 for better performance
WHERE (toLower(n.name) CONTAINS lower_keyword OR toLower(m.name) CONTAINS lower_keyword)
WITH n, m
LIMIT 5000  // Limit the number of node pairs
MATCH p = shortestPath((n)-[*1..3]-(m))  // Use shortestPath to get only the shortest path
RETURN p


"""
# get path nodes

"""  
  
WITH ["neuroinflammation", "inflammation", "neuroimmune response", "neuroimmunology", "brain inflammation",
      "oxidative stress", "free radicals", "reactive oxygen species", "ROS", "oxidative damage",
      "mitochondrial", "mitochondrial dysfunction", "mitochondrial damage", "mitochondrial health", "energy production",
      "blood-brain barrier", "brain barrier", "BBB", "neurovascular unit", "blood-brain permeability",
      "amyloid", "amyloid plaques", "beta-amyloid", "amyloid beta", "plaques in Alzheimer's",
      "tau", "tau protein", "tau tangles", "tauopathies", "tau neurofibrillary tangles",
      "autonomic dysfunction", "autonomic nervous system", "ANS", "autonomic regulation", "autonomic imbalance",
      "neurotropism", "neuronal targeting", "neurotropic", "neurotropic viruses", "neurotropic diseases"] AS keywords
UNWIND keywords AS keyword
WITH DISTINCT toLower(keyword) AS lower_keyword
MATCH (n)-[r*1..3]-(m)  // Set the relationship depth to 1..3 for better performance
WHERE (toLower(n.name) CONTAINS lower_keyword OR toLower(m.name) CONTAINS lower_keyword)
WITH n, m
LIMIT 5000  // Limit the number of node pairs considered
MATCH p = shortestPath((n)-[*1..3]-(m))  // Use shortestPath to find the shortest path
WITH n, m, p, relationships(p) AS relTypes, length(p) AS pathLength
UNWIND relTypes AS rel  // Unwind relationships to extract the types and properties of relationships
WITH n, m, rel, pathLength,
     CASE 
        WHEN rel.source IS NOT NULL AND rel.source <> '' THEN rel.source 
        ELSE null 
     END AS validSource  // Only include source if it is not null or empty
WHERE validSource IS NOT NULL  // Filter out paths where source is null or empty
RETURN 
    n.name AS `Start Node`,  // Start Node (name of the first node in the path)
    type(rel) AS `REL_type`,  // Relationship type (for each relationship in the path)
    m.name AS `End Node`,
    rel.pmid AS PMID,  // End Node (name of the second node in the path)
    rel.evidence AS `Evidences`,  // Evidence (assuming Evidence is a property of the relationship)
    validSource AS `Source`,  // Only include Source if it is not null or empty
    pathLength  // Path Length (optional, shows the length of the path)
ORDER BY pathLength  // Sort by path length
LIMIT 500  // Limit the number of paths returned

"""
##add cbm triple as well
"""WITH [
    // Neuroinflammatory mechanisms
    "neuroinflammation", "inflammation", "neuroimmune response", "neuroimmunology", "brain inflammation",
    "cytokine storm", "inflammatory mediators", "glial activation", "microglial activation",
    
    // Oxidative stress pathway
    "oxidative stress", "free radicals", "reactive oxygen species", "ROS", "oxidative damage",
    "antioxidant response", "redox signaling", "lipid peroxidation",
    
    // Mitochondrial dysfunction
    "mitochondrial", "mitochondrial dysfunction", "mitochondrial damage", "mitochondrial health",
    "energy production", "ATP synthesis", "electron transport chain",
    
    // Blood-brain barrier disruption
    "blood-brain barrier", "brain barrier", "BBB", "neurovascular unit", "blood-brain permeability",
    "tight junction proteins", "endothelial dysfunction",
    
    // Protein aggregation
    "amyloid", "amyloid plaques", "beta-amyloid", "amyloid beta", "protein aggregation",
    "protein misfolding", "proteostasis", "protein degradation",
    
    // Tau pathology
    "tau", "tau protein", "tau tangles", "tauopathies", "tau phosphorylation",
    "microtubule dysfunction", "axonal transport",
    
    // Autonomic dysfunction
    "autonomic dysfunction", "autonomic nervous system", "ANS", "autonomic regulation",
    "sympathetic activation", "parasympathetic dysfunction",
    
    // Neurotropism
    "neurotropism", "neuronal targeting", "neurotropic", "viral neuroinvasion",
    "neural spread", "synaptic dysfunction"
] AS keywords

// Process keywords once
UNWIND keywords AS keyword
WITH DISTINCT toLower(keyword) AS lower_keyword
WITH collect(lower_keyword) AS processed_keywords

// Main query with optimizations
MATCH (n)
WHERE any(keyword IN processed_keywords WHERE toLower(n.name) CONTAINS keyword)
MATCH (m)
WHERE id(m) > id(n) AND 
      any(keyword IN processed_keywords WHERE toLower(m.name) CONTAINS keyword)
WITH n, m
CALL apoc.path.expandConfig(n, {
    minLevel: 1,
    maxLevel: 3,
    uniqueness: "NODE_PATH",
    targetNode: m,
    limit: 1
}) YIELD path

WITH n, m, path,
     [rel in relationships(path) | rel] AS rels,
     length(path) AS pathLength
UNWIND rels AS rel

WITH n, m, rel, pathLength,
     CASE 
        WHEN rel.source IS NOT NULL AND trim(rel.source) <> '' 
        THEN rel.source 
        ELSE null 
     END AS validSource,
     CASE
        WHEN rel.evidence IS NOT NULL AND trim(rel.evidence) <> ''
        THEN rel.evidence
        ELSE 'No evidence provided'
     END AS evidence
WHERE validSource IS NOT NULL

RETURN DISTINCT
    n.name AS `Start Node`,
    type(rel) AS `Relationship`,
    rel.pmid as PMID,
    m.name AS `End Node`,
    evidence AS `Evidence`,
    validSource AS `Source`,
    pathLength AS `Path Length`,
    CASE
        WHEN n.type IS NOT NULL THEN n.type
        ELSE 'Unknown'
    END AS `Start Node Type`,
    CASE
        WHEN m.type IS NOT NULL THEN m.type
        ELSE 'Unknown'
    END AS `End Node Type`
ORDER BY pathLength, `Start Node`
LIMIT 1000
"""


# # Shared Mechanisms, Genes, Symbols (overall query)

# In[ ]:


"""  
MATCH (n)-[r]->(m)
WHERE toLower(r.source) CONTAINS "sherpa" 
  AND (
    toLower(n.name) CONTAINS "covid" OR 
    toLower(n.name) CONTAINS "sars-cov-2" OR
    toLower(n.name) CONTAINS "neuroinflammation" OR
    toLower(n.name) CONTAINS "oxidative stress" OR
    toLower(n.name) CONTAINS "immune response" OR
    toLower(n.name) CONTAINS "viral protein aggregation" OR
    toLower(n.name) CONTAINS "hypoxia" OR
    toLower(n.name) CONTAINS "amyloidogenesis" OR
    toLower(n.name) CONTAINS "neurotropism" OR
    toLower(n.name) CONTAINS "mitochondrial dysfunction" OR
    toLower(n.name) CONTAINS "cognitive decline" OR
    toLower(n.name) CONTAINS "parkinsonism" OR
    toLower(n.name) CONTAINS "fatigue" OR
    toLower(n.name) CONTAINS "mood disorders" OR
    toLower(n.name) CONTAINS "anosmia" OR
    toLower(n.name) CONTAINS "ageusia" OR
    toLower(n.name) CONTAINS "tau" OR
    toLower(n.name) CONTAINS "ace2" OR
    toLower(n.name) CONTAINS "snca" OR
    toLower(n.name) CONTAINS "lrk2" OR
    toLower(n.name) CONTAINS "amyloid beta" OR
    toLower(n.name) CONTAINS "glial cell activation" OR
    toLower(n.name) CONTAINS "cognitive dysfunction" OR
    toLower(n.name) CONTAINS "neurodegeneration" OR
    toLower(n.name) CONTAINS "synaptic dysfunction" OR
    toLower(n.name) CONTAINS "neurovascular dysfunction" OR
    toLower(n.name) CONTAINS "il-6" OR
    toLower(n.name) CONTAINS "tnf-α" OR
    toLower(n.name) CONTAINS "autophagy impairment" OR
    toLower(n.name) CONTAINS "motor symptoms"
  )
RETURN n.name as Start_Node, type(r) as REL_type, m.name as End_Node, r.evidence as Evidence, r.pmid as PMID

"""


# # Shared Genes

# In[ ]:


"""
MATCH (gene:Gene)-[r1]-(d1:Disease)
MATCH (gene)-[r2]-(d2:Disease)
WHERE 
    toLower(r1.source) CONTAINS "opentarget" 
    AND toLower(r2.source) CONTAINS "opentarget"
    AND (toLower(d1.name) CONTAINS "covid" OR toLower(d1.name) CONTAINS "covid-19")
    AND (
        toLower(d2.name) CONTAINS "alzheimer" 
        OR toLower(d2.name) CONTAINS "parkinson"
        OR toLower(d2.name) CONTAINS "neuro"
    )
RETURN DISTINCT 
    gene.symbol as Gene,
    d1.name as Covid_Related,
    d2.name as Neurological_Disease
ORDER BY Gene
""""""


# # Phenotypes

# In[ ]:


#Find paths that have an occurence of phenotpyes (not fuzzy)
"""
WITH ["Memory Impairment", "memory loss", "memory deficit",
      "Psychomotor Deterioration", "motor skills", "motor impairment",
      "Cognitive Epileptic Aura", "cognitive aura", "seizure disturbance",
      "Dysdiadochokinesis", "alternating movement", "motor coordination",
      "Status Epilepticus", "seizure paralysis", "motor weakness",
      "Sensorineural Hearing Impairment", "hearing loss", "deafness",
      "Gaze Palsy", "eye movement", "gaze palsy",
      "Leukoencephalopathy", "white matter", "brain disease",
      "Daytime Somnolence", "drowsiness", "sleepiness",
      "Diabetes Insipidus", "ADH deficiency", "hormone disorder",
      "Sleep-Wake Rhythm", "sleep cycle", "circadian disturbance",
      "Motor Neuron Dysfunction", "spasticity", "hyperreflexia",
      "Memory Impairment", "past memory loss", "memory recall",
      "Sensory Seizure", "smell hallucinations", "odor perception",
      "CSF Pyridoxal", "Vitamin B6", "CSF B6 levels",
      "Demyelination", "nerve damage", "myelin sheath",
      "Gait Ataxia", "unsteady walk", "balance",
      "Cognitive Seizure", "memory issues", "focal seizure",
      "Neurofibrillary Tangles", "tau protein", "brain tangles",
      "Neuroinflammation", "brain inflammation", "CNS inflammation"
     ] AS keywords,
     ["disgenet", "opentarget", "drugbank", "indra"] AS sources
 
MATCH p=(n)-[r]-(m) 
WHERE ANY(source IN sources WHERE toLower(r.source) CONTAINS toLower(source)) 
  AND ANY(keyword IN keywords WHERE toLower(n.name) CONTAINS toLower(keyword) OR m.name CONTAINS toLower(keyword))
RETURN p
LIMIT 50

"""

#save above as list of nodeds and edges
""" 
WITH ["Memory Impairment", "memory loss", "memory deficit",
      "Psychomotor Deterioration", "motor skills", "motor impairment",
      "Cognitive Epileptic Aura", "cognitive aura", "seizure disturbance",
      "Dysdiadochokinesis", "alternating movement", "motor coordination",
      "Status Epilepticus", "seizure paralysis", "motor weakness",
      "Sensorineural Hearing Impairment", "hearing loss", "deafness",
      "Gaze Palsy", "eye movement", "gaze palsy",
      "Leukoencephalopathy", "white matter", "brain disease",
      "Daytime Somnolence", "drowsiness", "sleepiness",
      "Diabetes Insipidus", "ADH deficiency", "hormone disorder",
      "Sleep-Wake Rhythm", "sleep cycle", "circadian disturbance",
      "Motor Neuron Dysfunction", "spasticity", "hyperreflexia",
      "Memory Impairment", "past memory loss", "memory recall",
      "Sensory Seizure", "smell hallucinations", "odor perception",
      "CSF Pyridoxal", "Vitamin B6", "CSF B6 levels",
      "Demyelination", "nerve damage", "myelin sheath",
      "Gait Ataxia", "unsteady walk", "balance",
      "Cognitive Seizure", "memory issues", "focal seizure",
      "Neurofibrillary Tangles", "tau protein", "brain tangles",
      "Neuroinflammation", "brain inflammation", "CNS inflammation"
     ] AS keywords,
     ["disgenet", "opentarget", "drugbank", "indra"] AS sources

MATCH p=(n)-[r]-(m) 
WHERE ANY(source IN sources WHERE toLower(r.source) CONTAINS toLower(source)) 
  AND ANY(keyword IN keywords WHERE toLower(n.name) CONTAINS toLower(keyword) OR toLower(m.name) CONTAINS toLower(keyword))
WITH p, 
     [node IN nodes(p) | node.name] AS node_names,
     [rel IN relationships(p) | type(rel)] AS relationship_types,
     [rel IN relationships(p) | rel.source] AS relationship_sources,
     length(p) AS path_length
RETURN node_names, relationship_types, relationship_sources, path_length
LIMIT 50
"""

#above with fuzzy matching

"""  

WITH ["Memory Impairment", "memory loss", "memory deficit",
      "Psychomotor Deterioration", "motor skills", "motor impairment",
      "Cognitive Epileptic Aura", "cognitive aura", "seizure disturbance",
      "Dysdiadochokinesis", "alternating movement", "motor coordination",
      "Status Epilepticus", "seizure paralysis", "motor weakness",
      "Sensorineural Hearing Impairment", "hearing loss", "deafness",
      "Gaze Palsy", "eye movement", "gaze palsy",
      "Leukoencephalopathy", "white matter", "brain disease",
      "Daytime Somnolence", "drowsiness", "sleepiness",
      "Diabetes Insipidus", "ADH deficiency", "hormone disorder",
      "Sleep-Wake Rhythm", "sleep cycle", "circadian disturbance",
      "Motor Neuron Dysfunction", "spasticity", "hyperreflexia",
      "Memory Impairment", "past memory loss", "memory recall",
      "Sensory Seizure", "smell hallucinations", "odor perception",
      "CSF Pyridoxal", "Vitamin B6", "CSF B6 levels",
      "Demyelination", "nerve damage", "myelin sheath",
      "Gait Ataxia", "unsteady walk", "balance",
      "Cognitive Seizure", "memory issues", "focal seizure",
      "Neurofibrillary Tangles", "tau protein", "brain tangles",
      "Neuroinflammation", "brain inflammation", "CNS inflammation"
     ] AS keywords,
     ["disgenet", "opentarget", "drugbank", "indra"] AS sources
 
MATCH p=(n)-[r]-(m)
WHERE ANY(source IN sources WHERE apoc.text.levenshteinDistance(toLower(r.source), toLower(source)) <= 2) 
  AND ANY(keyword IN keywords WHERE apoc.text.levenshteinDistance(toLower(n.name), toLower(keyword)) <= 2 
      OR apoc.text.levenshteinDistance(toLower(m.name), toLower(keyword)) <= 2)
RETURN p
LIMIT 50

"""


# # Shared Genes
# 

# In[ ]:


""" 
MATCH (d1:Disease)-[r]->(g:Gene)<-[r2]-(d2:Disease)
WHERE (
toLower(d1.name) CONTAINS 'covid'
AND (
toLower(d2.name) CONTAINS 'alzheimer' OR
toLower(d2.name) CONTAINS 'parkinson' OR
toLower(d2.name) CONTAINS 'neurodegenerative'
)
)
OR (
toLower(d2.name) CONTAINS 'covid'
AND (
toLower(d1.name) CONTAINS 'alzheimer' OR
toLower(d1.name) CONTAINS 'parkinson' OR
toLower(d1.name) CONTAINS 'neurodegenerative'
)
)
AND d1 <> d2 // Ensure that d1 and d2 are different nodes
RETURN
d1.name AS disease1,
TYPE(r) AS relationship1,
r.source as source_1,
g.name AS gene,
TYPE(r2) AS relationship2,
r2.source as source2,
d2.name AS disease2
"""  



# In[ ]:


#Use cell type biological process info as well
""" 
WITH [
    "epithelial", "immune cell types", "alveolar macrophages",
    "Monocytes", "myeloid cells recruited to tissues",
    "Neuronal cells", "immune cells",
    "Microglia", "astrocytes", "oligodendrocytes",
    "Microglia and hippocampal synapses", "inflammatory responses",
    "Glial cells", "microglia and astrocytes in brain inflammation",
    "Astrocytes", "microglia in cognitive impairment pathways",
    "Pyramidal neurons", "CA1 hippocampal region", "stress",
    "Microglia activation", "synapse loss",
    "inflammatory responses", "IL6", "IL1B", "TNF",
    "interferon responses",
    "CDC25C", "PLK1",
    "APOE", "TREM2", "TLR4 activation",
    "Genetic variants in TLR4","TLR4", "brain inflammation",
    "CDC25C", "PLK1",
    "Inflammatory gene responses", "cognitive decline",
    "Transcriptional changes",
    "TREM2", "APOE",
    "TLR4 genetic variant", "neuroinflammation",
    "Cytokine storms", "megakaryocytes", "monocytes",
    "Chemokine signaling", "cellular stress",
    "Neuroinflammatory processes", "cytokine","oxidative stress",
    "APP metabolism", "lipid metabolism", "TLR4",
    "TLR4-mediated", "neuroinflammation", "synaptic pruning", "activated microglia",
    "Neuroinflammation", "immune activation", "cognitive decline",
    "Persistent neuroinflammation", "blood-brain barrier disruption",
    "hippocampal sclerosis",
    "lipid metabolism", "protein aggregation", "TLR4 activation",
    "Microglial phagocytosis of synapses"
] AS keywords,
     ["sherpa"] AS sources
 
MATCH p=(n)-[r]-(m) 
WHERE ANY(source IN sources WHERE toLower(r.source) CONTAINS toLower(source)) 
  AND ANY(keyword IN keywords WHERE toLower(n.name) CONTAINS toLower(keyword) OR toLower(m.name) CONTAINS toLower(keyword))
RETURN p
limit 50

"""

#to get list of pahs
"""
WITH p, 
     [node IN nodes(p) | node.name] AS node_names,
     [rel IN relationships(p) | type(rel)] AS relationship_types,
     [rel IN relationships(p) | rel.source] AS relationship_sources,
     length(p) AS path_length
RETURN node_names, relationship_types, relationship_sources, path_length
""" 


#Above but using fuzzy matching (USED FOR PAPER!!!!!)
"""
WITH [
    "epithelial", "immune cell types", "alveolar macrophages",
    "Monocytes", "myeloid cells recruited to tissues",
    "Neuronal cells", "immune cells",
    "Microglia", "astrocytes", "oligodendrocytes",
    "Microglia and hippocampal synapses", "inflammatory responses",
    "Glial cells", "microglia and astrocytes in brain inflammation",
    "Astrocytes", "microglia in cognitive impairment pathways",
    "Pyramidal neurons", "CA1 hippocampal region", "stress",
    "Microglia activation", "synapse loss",
    "inflammatory responses", "IL6", "IL1B", "TNF",
    "interferon responses",
    "CDC25C", "PLK1",
    "APOE", "TREM2", "TLR4 activation",
    "Genetic variants in TLR4","TLR4", "brain inflammation",
    "CDC25C", "PLK1",
    "Inflammatory gene responses", "cognitive decline",
    "Transcriptional changes",
    "TREM2", "APOE",
    "TLR4 genetic variant", "neuroinflammation",
    "Cytokine storms", "megakaryocytes", "monocytes",
    "Chemokine signaling", "cellular stress",
    "Neuroinflammatory processes", "cytokine","oxidative stress",
    "APP metabolism", "lipid metabolism", "TLR4",
    "TLR4-mediated", "neuroinflammation", "synaptic pruning", "activated microglia",
    "Neuroinflammation", "immune activation", "cognitive decline",
    "Persistent neuroinflammation", "blood-brain barrier disruption",
    "hippocampal sclerosis",
    "lipid metabolism", "protein aggregation", "TLR4 activation",
    "Microglial phagocytosis of synapses"
] AS keywords,
     ["sherpa"] AS sources
 
MATCH p=(n)-[r]-(m)
WHERE ANY(source IN sources WHERE apoc.text.levenshteinDistance(toLower(r.source), toLower(source)) <= 2)
  AND ANY(keyword IN keywords WHERE apoc.text.levenshteinDistance(toLower(n.name), toLower(keyword)) <= 2 
      OR apoc.text.levenshteinDistance(toLower(m.name), toLower(keyword)) <= 2)
RETURN p
LIMIT 50

""" 

#to get list of pahs
"""
WITH p, 
     [node IN nodes(p) | node.name] AS node_names,
     [rel IN relationships(p) | type(rel)] AS relationship_types,
     [rel IN relationships(p) | rel.source] AS relationship_sources,
     length(p) AS path_length
RETURN node_names, relationship_types, relationship_sources, path_length
""" 

#above expand paths to more than length one (not used for paper took much time, mainly for PrimeKG intended first!!)

"""  
WITH [
    "epithelial", "immune cell types", "alveolar macrophages",
    "Monocytes", "myeloid cells recruited to tissues",
    "Neuronal cells", "immune cells",
    "Microglia", "astrocytes", "oligodendrocytes",
    "Microglia and hippocampal synapses", "inflammatory responses",
    "Glial cells", "microglia and astrocytes in brain inflammation",
    "Astrocytes", "microglia in cognitive impairment pathways",
    "Pyramidal neurons", "CA1 hippocampal region", "stress",
    "Microglia activation", "synapse loss",
    "inflammatory responses", "IL6", "IL1B", "TNF",
    "interferon responses",
    "CDC25C", "PLK1",
    "APOE", "TREM2", "TLR4 activation",
    "Genetic variants in TLR4","TLR4", "brain inflammation",
    "CDC25C", "PLK1",
    "Inflammatory gene responses", "cognitive decline",
    "Transcriptional changes",
    "TREM2", "APOE",
    "TLR4 genetic variant", "neuroinflammation",
    "Cytokine storms", "megakaryocytes", "monocytes",
    "Chemokine signaling", "cellular stress",
    "Neuroinflammatory processes", "cytokine","oxidative stress",
    "APP metabolism", "lipid metabolism", "TLR4",
    "TLR4-mediated", "neuroinflammation", "synaptic pruning", "activated microglia",
    "Neuroinflammation", "immune activation", "cognitive decline",
    "Persistent neuroinflammation", "blood-brain barrier disruption",
    "hippocampal sclerosis",
    "lipid metabolism", "protein aggregation", "TLR4 activation",
    "Microglial phagocytosis of synapses"
] AS keywords,
     ["cbm"] AS sources

MATCH p=(n)-[r*1..5]-(m)
WHERE ANY(rel IN r WHERE ANY(source IN sources WHERE apoc.text.levenshteinDistance(toLower(rel.source), toLower(source)) <= 2))
  AND ANY(keyword IN keywords WHERE apoc.text.levenshteinDistance(toLower(n.name), toLower(keyword)) <= 2 
      OR apoc.text.levenshteinDistance(toLower(m.name), toLower(keyword)) <= 2)
RETURN 
    [node IN nodes(p) | node] AS nodes,
    [rel IN relationships(p) | rel] AS relationships,
    length(p) AS path_length
LIMIT 50

"""


# In[ ]:


#above but branch up till reach covid node (for PRIMEKG especially)
""" 
WITH [
    "epithelial", "immune cell types", "alveolar macrophages",
    "Monocytes", "myeloid cells recruited to tissues",
    "Neuronal cells", "immune cells",
    "Microglia", "astrocytes", "oligodendrocytes",
    "Microglia and hippocampal synapses", "inflammatory responses",
    "Glial cells", "microglia and astrocytes in brain inflammation",
    "Astrocytes", "microglia in cognitive impairment pathways",
    "Pyramidal neurons", "CA1 hippocampal region", "stress",
    "Microglia activation", "synapse loss",
    "inflammatory responses", "IL6", "IL1B", "TNF",
    "interferon responses",
    "CDC25C", "PLK1",
    "APOE", "TREM2", "TLR4 activation",
    "Genetic variants in TLR4","TLR4", "brain inflammation",
    "CDC25C", "PLK1",
    "Inflammatory gene responses", "cognitive decline",
    "Transcriptional changes",
    "TREM2", "APOE",
    "TLR4 genetic variant", "neuroinflammation",
    "Cytokine storms", "megakaryocytes", "monocytes",
    "Chemokine signaling", "cellular stress",
    "Neuroinflammatory processes", "cytokine","oxidative stress",
    "APP metabolism", "lipid metabolism", "TLR4",
    "TLR4-mediated", "neuroinflammation", "synaptic pruning", "activated microglia",
    "Neuroinflammation", "immune activation", "cognitive decline",
    "Persistent neuroinflammation", "blood-brain barrier disruption",
    "hippocampal sclerosis",
    "lipid metabolism", "protein aggregation", "TLR4 activation",
    "Microglial phagocytosis of synapses"
] AS keywords,
     ["scai"] AS sources
 
MATCH initial_path=(n)-[r]-(m)
WHERE ANY(keyword IN keywords WHERE apoc.text.levenshteinDistance(toLower(n.name), toLower(keyword)) <= 2 
      OR apoc.text.levenshteinDistance(toLower(m.name), toLower(keyword)) <= 2)
WITH DISTINCT nodes(initial_path) AS initial_nodes

// Expand and find the shortest path from each initial node to a COVID-related node
UNWIND initial_nodes AS start_node
MATCH sp=shortestPath((start_node)-[*]-(covid_node))
WHERE apoc.text.levenshteinDistance(toLower(covid_node.name), "covid") <= 2
RETURN 
    [node IN nodes(sp) | node] AS path_nodes,
    [rel IN relationships(sp) | rel] AS path_relationships,
    length(sp) AS path_length,
    covid_node.name AS covid_node_name
LIMIT 100
"""


# In[ ]:


#To asses and valdiate and check Sherpa paths in hypothesis space

"""  
match(n)-[r]-(m) where "sherpa" in r.source and toLower(n.name) contains "inflammation" and toLower(m.name) contains "ifng" return n,r,m
"""

"""

"""#To asses and valdiate and check  CBM  paths in hypothesis space

"""  
match(n)-[r]-(m) where  toLower(r.source) contains "cbm" and toLower(n.name) contains "inflammation" and toLower(m.name) contains "ifng" return n,r,m
"""


# In[ ]:


#Draw plots: paths showing comorbidity (showing node degree as well)

import matplotlib.pylab as plt
import networkx as nx
import matplotlib.cm as cm
import numpy as np

# Set up figure size and layout for longer node names
plt.rcParams["figure.figsize"] = [16, 16]
plt.rcParams["figure.autolayout"] = True

# Create an undirected graph
G = nx.Graph()

# Add original nodes and edges
new_edges = [
    ("COVID-19", "Inflammation"),
    ("Inflammation", "MOG"),
    ("MOG", "Encephalomyelitis, Acute Disseminated"),
    ("COVID-19", "Inflammation"),
    ("Inflammation", "IgG"),
    ("IgG", "Immune System Disease"),
    ("COVID-19", "Inflammation"),
    ("Inflammation", "IgG"),
    ("IgG", "Immune System Disease"),
    ("Immune System Disease", "Myelitis"),
    ("Myelitis", "Lymphopenia"),
    ("Lymphopenia", "MOG"),
    ("MOG", "Optic Neuritis"),
    ("COVID-19", "Inflammation"),
    ("Inflammation", "IFNG"),
    ("IFNG", "IL1B"),
    ("IL1B", "Taupaties"),
    ("SARS-CoV-2", "Myelitis"),
    ("Myelitis", "Peripheral Nervous System Diseases"),
    ("SARS-CoV-2", "IL6"),
    ("IL6", "Depression"),
    ("Depression", "TNF"),
    ("TNF", "Neurotransmitters"),
    ("Neurotransmitters", "Encephalitis")
]

# Add edges to the graph
G.add_edges_from(new_edges)

# Generate layout positions
pos = nx.spring_layout(G, k=1.2, seed=42)

# Compute node degrees and normalize for node size and color scaling
degrees = dict(G.degree())
node_size = [v * 300 for v in degrees.values()]  # scale node size by degree
node_color = [v for v in degrees.values()]  # node color based on degree
norm = plt.Normalize(vmin=min(node_color), vmax=max(node_color))  # normalize colors

# Draw nodes with gradient coloring and varied sizes
nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, cmap=cm.viridis, alpha=0.9)

# Draw curved edges with transparency to reduce clutter
nx.draw_networkx_edges(G, pos, connectionstyle="arc3,rad=0.4", alpha=0.5)

# Draw labels with larger font for clarity
nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", font_color="black", font_weight="bold")

# Add the title with styling
plt.title("Sherpa", fontsize=16, fontweight='bold', color='darkblue')

# Add a colorbar to show the degree of the nodes
sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
sm.set_array([])
plt.colorbar(sm, label="Node Degree")

plt.show()


# # Use chatgpt for figure creation

# In[ ]:


# prompt for mergin multiple graphs

""" merge all these figures in a professional way for paper: to specify the graphs, add label for each graph as name of file : Sherpa, SCAI-DMaps, PubTator3, PrimeKG, OpenTargets, DisGeNET, INDRA, KEGG, DrugBank, CBM
"""



# ----------------------------
# Main Entry
# ----------------------------
if __name__ == "__main__":
    verify_input_hashes()
    import_knowledge_graphs()
    analyze_graphs()
