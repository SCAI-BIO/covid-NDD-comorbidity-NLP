
# % AURA DB upload
from neo4j import GraphDatabase
import pandas as pd
import re

# Neo4j AuraDB connection details
NEO4J_URI = "neo4j+s://1af6a087.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "JpH3143DPIU5vYOhBBvYUAao6uUN9yDqZnI_14asTx0"

# Input text file with paths and harmonized CSV
input_file = "comorbidity_paths.txt"  # Replace with your file path
harmonized_csv_path = "harmonized_hypotheses.csv"

# Establish Neo4j connection
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Load harmonized entities from the CSV
harmonized_data = pd.read_csv(harmonized_csv_path)
harmonized_dict = {
    row["Original Entity"]: (row["Harmonized Entity"], row["Ontology"])
    for _, row in harmonized_data.iterrows()
}

# Function to sanitize ontology labels
def sanitize_label(ontology):
    if ontology == "HGNC":
        return "Gene"
    elif ontology in {"Disease Ontology", "MESH", "MESHD"}:
        return "Disease"
    elif ontology.lower() == "chebi":
        return "Chemical"
    elif ontology == "GO":
        return "Biological_Process"
    else:
        return ontology.replace(" ", "_")
        
from rapidfuzz import fuzz, process  # For fast fuzzy matching

def normalize_entity(entity):
    # Convert the input entity to lowercase for case-insensitive matching
    entity_lower = entity.lower()
    
    # Create a mapping of lowercase keys for fuzzy matching
    harmonized_lower_dict = {key.lower(): key for key in harmonized_dict}

    # Perform a fuzzy match
    best_match, score, _ = process.extractOne(entity_lower, harmonized_lower_dict.keys(), scorer=fuzz.ratio)

    # If the score exceeds the threshold (e.g., 80), consider it a match
    if score > 80:
        matched_key = harmonized_lower_dict[best_match]
        harmonized_name, ontology = harmonized_dict[matched_key]
        return harmonized_name, sanitize_label(ontology), ontology
    
    # If no match is found, return default values
    return entity, "Unknown", "Unknown"


# Function to parse paths and relationships
def parse_paths_from_text(file_path):
    paths = []
    current_path = None
    with open(file_path, 'r', encoding="utf-8") as f:  # Ensure proper encoding
        for line in f:
            line = line.strip()
            # Replace incorrectly encoded characters
            line = line.replace("â†’", "→")
            if line.startswith("Path:"):  # Detect path lines
                if current_path:
                    paths.append(current_path)
                current_path = {"path": [], "source": None, "mechanism": None}
                # Split entities using → delimiter
                entities = [entity.strip() for entity in line[len("Path:"):].split("→")]
                current_path["path"] = entities
            elif line.lower().startswith("source:"):  # Detect source lines
                current_path["source"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("mechanism:"):  # Detect mechanism lines
                current_path["mechanism"] = line.split(":", 1)[1].strip()
        if current_path:  # Add the last path if exists
            paths.append(current_path)
    return paths

# Function to upload paths to Neo4j
def upload_to_neo4j(tx, paths):
    for entry in paths:
        path = entry["path"]
        source = entry["source"]

        for i in range(len(path) - 1):
            source_node, source_label, source_ontology = normalize_entity(path[i])
            target_node, target_label, target_ontology = normalize_entity(path[i + 1])

            print(f"Adding: {source_node} ({source_label}) -> {target_node} ({target_label}) [source: {source}]")  # Debug

            # Add nodes with ontology as a property
            query = f"""
            MERGE (src:{source_label} {{name: $source_name}})
            SET src.namespace = $source_namespace
            MERGE (tgt:{target_label} {{name: $target_name}})
            SET tgt.namespace = $target_namespace
            MERGE (src)-[rel:RELATED_TO {{source: $source}}]->(tgt)
            """
            tx.run(
                query,
                source_name=source_node,
                source_namespace=source_ontology,
                target_name=target_node,
                target_namespace=target_ontology,
                source=source,
            )

# Parse paths from the input text file
paths_data = parse_paths_from_text(input_file)

# Upload data to Neo4j
with driver.session() as session:
    session.write_transaction(upload_to_neo4j, paths_data)

print("Data upload completed successfully.")

# %%
