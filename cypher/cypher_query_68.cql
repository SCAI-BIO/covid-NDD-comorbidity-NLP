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