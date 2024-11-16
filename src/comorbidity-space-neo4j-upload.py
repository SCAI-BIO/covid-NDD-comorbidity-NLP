
# %% Upload hypothesis in csv format to Neo4j AuraDB
import pandas as pd
import requests
from rapidfuzz import fuzz, process
from neo4j import GraphDatabase

def sanitize_label(label):
    """Sanitize labels for Neo4j compatibility."""
    return label.replace("-", "_").replace(" ", "_").replace(".", "_")

def determine_entity_type(entity):
    """
    Pre-check the entity string for obvious entity types before making API calls.
    Returns (is_likely_gene, suggested_label) tuple.
    """
    entity = entity.strip().upper()
    
    # Common disease indicators
    disease_indicators = {"SYNDROME", "DISEASE", "DISORDER", "IMMUNODEFICIENCY", 
                         "ITIS", "OSIS", "PATHY", "CANCER", "TUMOR", "DEFICIENCY"}
    
    # Common biological process indicators
    process_indicators = {"PATHWAY", "SIGNALING", "INFLAMMATION", "RESPONSE", 
                        "REGULATION", "METABOLISM", "SYNTHESIS", "ACTIVITY"}
    
    # Check for disease indicators
    if any(indicator in entity for indicator in disease_indicators):
        return False, "Disease"
        
    # Check for biological process indicators
    if any(indicator in entity for indicator in process_indicators):
        return False, "Biological_Process"
        
    # If no obvious indicators, might be a gene
    return True, None

def query_hgnc_fuzzy(entity):
    """Query HGNC database for gene symbols."""
    entity = entity.strip().upper()
    headers = {"Accept": "application/json"}
    
    # Try exact symbol match
    url = f"https://rest.genenames.org/search/symbol/{entity}"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        results = response.json()
        if results.get("response", {}).get("numFound", 0) > 0:
            return "Gene", "HGNC"
            
    # Try alias search
    url = f"https://rest.genenames.org/search/alias_symbol/{entity}"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        results = response.json()
        if results.get("response", {}).get("numFound", 0) > 0:
            return "Gene", "HGNC"
    
    return None, None

def query_ols_fuzzy(entity):
    """Query OLS for ontology information."""
    url = f"https://www.ebi.ac.uk/ols/api/search?q={entity}"
    response = requests.get(url)
    
    if response.status_code == 200:
        results = response.json()
        docs = results.get("response", {}).get("docs", [])
        
        if docs:
            for doc in docs:
                if "ontology_prefix" in doc:
                    ontology = doc["ontology_prefix"]
                    
                    # Map ontology prefixes to entity types
                    if ontology in {"DOID", "MESH", "MESHD"}:
                        return "Disease", ontology
                    elif ontology == "CHEBI":
                        return "Chemical", ontology
                    elif ontology == "GO":
                        return "Biological_Process", ontology
    
    return None, None

def determine_entity_label(entity):
    """
    Improved function to determine entity label using a multi-step approach.
    """
    # Step 1: Check for obvious indicators in the entity name
    is_likely_gene, suggested_label = determine_entity_type(entity)
    
    if suggested_label:
        return sanitize_label(suggested_label), "General"
    
    # Step 2: If it might be a gene, check HGNC
    if is_likely_gene:
        label, namespace = query_hgnc_fuzzy(entity)
        if label:
            return sanitize_label(label), sanitize_label(namespace)
    
    # Step 3: Check OLS for other entity types
    label, namespace = query_ols_fuzzy(entity)
    if label:
        return sanitize_label(label), sanitize_label(namespace)
    
    # Step 4: Fallback to general entity type
    return "General_Entity", "General"

class Neo4jUploader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()

    def upload_triples(self, triples):
        with self.driver.session() as session:
            for idx, row in triples.iterrows():
                print(f"Processing row {idx + 1}...")
                try:
                    session.write_transaction(self.create_relationship, row)
                except Exception as e:
                    print(f"Error uploading row {idx + 1}: {e}")
                    raise e

    @staticmethod
    def create_relationship(tx, row):
        start_entity = row['Start Node'].strip().upper()
        end_entity = row['End Node'].strip().upper()
        rel_type = row['REL_type']
        pmid = row['PMID']
        evidence = row['Evidences']
        source = row['Source']

        # Determine labels and namespaces
        start_label, start_namespace = determine_entity_label(start_entity)
        end_label, end_namespace = determine_entity_label(end_entity)

        query = """
        MERGE (start:%s {name: $start_entity, namespace: $start_namespace})
        MERGE (end:%s {name: $end_entity, namespace: $end_namespace})
        MERGE (start)-[rel:%s {
            pmid: $pmid,
            evidence: $evidence,
            source: $source
        }]->(end)
        """ % (start_label, end_label, rel_type)
        
        print(f"Entity: {start_entity} -> Label: {start_label}")
        print(f"Entity: {end_entity} -> Label: {end_label}")
        
        tx.run(
            query,
            start_entity=start_entity,
            end_entity=end_entity,
            start_namespace=start_namespace,
            end_namespace=end_namespace,
            pmid=pmid,
            evidence=evidence,
            source=source,
        )

# Usage example:
def main():
    # Load CSV file with triples
    file_path = 'hypothesis_pmid_evidences.csv'
    data = pd.read_csv(file_path)
    data.fillna("Unknown_Entity", inplace=True)

    # Prepare the data for upload
    triples = data[["Source", "Start Node", "End Node", "REL_type", "PMID", "Evidences"]]

    # Upload data to Neo4j
    neo4j_uploader = Neo4jUploader(
        "neo4j+s://1af6a087.databases.neo4j.io",
        "neo4j",
        "JpH3143DPIU5vYOhBBvYUAao6uUN9yDqZnI_14asTx0"
    )
    
    try:
        neo4j_uploader.upload_triples(triples)
        print("Data upload to Neo4j completed successfully.")
    finally:
        neo4j_uploader.close()

if __name__ == "__main__":
    main()
# %%
