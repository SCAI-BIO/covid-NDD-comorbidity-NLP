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