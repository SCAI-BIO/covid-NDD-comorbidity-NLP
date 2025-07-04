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