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