
# %% Upload curated pathways to database
from operator import delitem
import pandas as pd
import requests
from rapidfuzz import fuzz, process
from neo4j import GraphDatabase
import os
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
    
    gene_indicators = {"APOE4", "BRCA1", "TP53"}
    print("entitiy", entity)
    if entity.lower().startswith("apoe") or entity in gene_indicators:
        #print("gene")
        return True, "Gene"

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
def query_disease_ontology_fuzzy(entity, threshold=0.7):
    """
    Query Disease Ontology (DO) for approximate string matching.
    
    Parameters:
        entity (str): The entity to search for.
        threshold (float): The similarity threshold for approximate matches (0 to 1).
        
    Returns:
        tuple: The type "Disease" and "DO" if a match is found; otherwise, (None, None).
    """
    url = f"https://www.ebi.ac.uk/ols/api/ontologies/doid/terms?q={entity.lower()}"
    response = requests.get(url)
    
    if response.status_code == 200:
        results = response.json()
        terms = results.get("_embedded", {}).get("terms", [])
        
        if terms:
            for term in terms:
                label = term.get("label", "").lower()
                if entity.lower() in label or \
                   get_close_matches(entity.lower(), [label], n=1, cutoff=threshold):
                    return "Disease", "DO"
    
    return None, None

def query_ols_fuzzy(entity, threshold=0.7):
    """
    Query OLS for ontology information with case-insensitive and approximate string matching,
    including a dedicated check for the Disease Ontology.
    
    Parameters:
        entity (str): The entity to search for.
        threshold (float): The similarity threshold for approximate matches (0 to 1).
        
    Returns:
        tuple: The type of entity and its ontology prefix if found; otherwise, (None, None).
    """
    # First, check specifically in Disease Ontology
    label, namespace = query_disease_ontology_fuzzy(entity, threshold)
    if label:
        return label, namespace

    # Generic OLS query for all ontologies
    url = f"https://www.ebi.ac.uk/ols/api/search?q={entity.lower()}"
    response = requests.get(url)
    
    if response.status_code == 200:
        results = response.json()
        docs = results.get("response", {}).get("docs", [])
        
        if docs:
            for doc in docs:
                if "ontology_prefix" in doc:
                    ontology = doc["ontology_prefix"]
                    label = doc.get("label", "").lower()
                    
                    # Approximate string matching
                    if entity.lower() in label or \
                       get_close_matches(entity.lower(), [label], n=1, cutoff=threshold):
                        
                        # Map ontology prefixes to entity types
                        if ontology in {"DOID", "MESH", "MESHD"}:
                            return "Disease", ontology
                        # if ontology in {"DOID", "MESH", "MESHD"}:
                        #     return "Disease", ontology
                        elif ontology == "CHEBI":
                            return "Chemical", ontology
                        elif ontology == "GO":
                            return "Biological_Process", ontology
    
    return None, None

import requests
from difflib import get_close_matches
def query_mesh_fuzzy(entity, threshold=0.7):
    """
    Query MESH for fuzzy term matching.
    Returns the term name if found, None otherwise.
    """
    url = f"https://www.ebi.ac.uk/ols/api/ontologies/mesh/terms?q={entity.lower()}"
    response = requests.get(url)
    
    if response.status_code == 200:
        results = response.json()
        terms = results.get("_embedded", {}).get("terms", [])
        
        if terms:
            best_match = None
            highest_ratio = 0
            
            for term in terms:
                label = term.get("label", "").lower()
                ratio = fuzz.ratio(entity.lower(), label)
                
                if ratio > highest_ratio and ratio >= threshold * 100:
                    highest_ratio = ratio
                    best_match = term.get("label")
            
            return best_match
    
    return None

def determine_entity_label(entity):
    """
    Improved function to determine entity label using a multi-step approach.
    Exhaustively searches through multiple ontologies before making a determination.
    """
    # Step 1: Check for obvious indicators in the entity name
    is_likely_gene, suggested_label = determine_entity_type(entity)
    
    # Step 2: Handle specific entity types based on suggested label
    if suggested_label:
        if suggested_label == "Disease":
            # Search specifically in Disease Ontology
            label, namespace = query_disease_ontology_fuzzy(entity)
            if label:
                return sanitize_label(label), sanitize_label(namespace)
        
        elif suggested_label == "Biological_Process":
            # Search in MESH for biological processes
            url = f"https://www.ebi.ac.uk/ols/api/ontologies/mesh/terms?q={entity.lower()}"
            response = requests.get(url)
            
            if response.status_code == 200:
                results = response.json()
                terms = results.get("_embedded", {}).get("terms", [])
                
                if terms:
                    for term in terms:
                        label = term.get("label", "").lower()
                        if entity.lower() in label or \
                           get_close_matches(entity.lower(), [label], n=1, cutoff=0.7):
                            return "Biological_Process", "MESH"
    
    # Step 3: If it might be a gene, check HGNC
    if is_likely_gene:
        label, namespace = query_hgnc_fuzzy(entity)
        print(label)
        if label:
            return sanitize_label(label), sanitize_label(namespace)
        """
    Improved function to determine entity label, first checking MESH terms.
    """
    # Step 1: Try to find exact MESH term
    mesh_term = query_mesh_fuzzy(entity)
    if mesh_term:
        # If found in MESH, determine appropriate category
        if any(indicator in mesh_term.upper() for indicator in 
               ["DISEASE", "SYNDROME", "DISORDER", "DEFICIENCY"]):
            return "Disease", "MESH"
        elif any(indicator in mesh_term.upper() for indicator in 
                ["PATHWAY", "PROCESS", "ACTIVITY", "FUNCTION"]):
            return "Biological_Process", "MESH"
    
    # Step 2: Continue with existing checks
    is_likely_gene, suggested_label = determine_entity_type(entity)
    
    if suggested_label:
        if suggested_label == "Disease":
            label, namespace = query_disease_ontology_fuzzy(entity)
            if label:
                return sanitize_label(label), sanitize_label(namespace)
        
        elif suggested_label == "Biological_Process":
            label, namespace = query_gene_ontology_fuzzy(entity)
            if label:
                return sanitize_label(label), sanitize_label(namespace)
    
    if is_likely_gene:
        #print("test",entity)
        label, namespace = query_hgnc_fuzzy(entity)
        #print("test test",label)
        if label:
            return sanitize_label(label), sanitize_label(namespace)

    # Step 4: Comprehensive ontology search
    # First, try Disease Ontology if not already checked
    if not suggested_label or suggested_label != "Disease":
        label, namespace = query_disease_ontology_fuzzy(entity)
        if label:
            return sanitize_label(label), sanitize_label(namespace)
    
    # Try MESH for biological processes if not already checked
    if not suggested_label or suggested_label != "Biological_Process":
        url = f"https://www.ebi.ac.uk/ols/api/ontologies/mesh/terms?q={entity.lower()}"
        response = requests.get(url)
        
        if response.status_code == 200:
            results = response.json()
            terms = results.get("_embedded", {}).get("terms", [])
            
            if terms:
                for term in terms:
                    label = term.get("label", "").lower()
                    if entity.lower() in label or \
                       get_close_matches(entity.lower(), [label], n=1, cutoff=0.7):
                        return "Biological_Process", "MESH"
    
    # Comprehensive OLS search across multiple ontologies
    ontology_prefixes = {
        "GO": "Biological_Process",
        "CHEBI": "Chemical",
        "HP": "Phenotype",
        "MONDO": "Disease",
        "UBERON": "Anatomy",
        "CL": "Cell",
        "PR": "Protein",
        "PW": "Pathway",
        "MP": "Phenotype",
        "NCIT": "Disease",  # National Cancer Institute Thesaurus
        "EFO": "Experimental_Factor",  # Experimental Factor Ontology
        "SYMP": "Symptom"  # Symptom Ontology
    }
    
    url = f"https://www.ebi.ac.uk/ols/api/search?q={entity.lower()}"
    response = requests.get(url)
    
    if response.status_code == 200:
        results = response.json()
        docs = results.get("response", {}).get("docs", [])
        
        if docs:
            # First pass: try to find exact or close matches
            for doc in docs:
                ontology = doc.get("ontology_prefix")
                label = doc.get("label", "").lower()
                
                if ontology in ontology_prefixes and (
                    entity.lower() in label or 
                    get_close_matches(entity.lower(), [label], n=1, cutoff=0.7)
                ):
                    return ontology_prefixes[ontology], ontology
            
            # Second pass: try partial matches with lower threshold
            for doc in docs:
                ontology = doc.get("ontology_prefix")
                label = doc.get("label", "").lower()
                
                if ontology in ontology_prefixes and (
                    get_close_matches(entity.lower(), [label], n=1, cutoff=0.6)
                ):
                    return ontology_prefixes[ontology], ontology
    
    # If no matches found in any ontology, use the most relevant ontology based on text similarity
    url = f"https://www.ebi.ac.uk/ols/api/search?q={entity.lower()}&rows=50"
    response = requests.get(url)
    
    if response.status_code == 200:
        results = response.json()
        docs = results.get("response", {}).get("docs", [])
        
        if docs:
            # Get the most frequently occurring ontology in the results
            ontology_counts = {}
            for doc in docs:
                ontology = doc.get("ontology_prefix")
                if ontology in ontology_prefixes:
                    ontology_counts[ontology] = ontology_counts.get(ontology, 0) + 1
            
            if ontology_counts:
                most_likely_ontology = max(ontology_counts.items(), key=lambda x: x[1])[0]
                return ontology_prefixes[most_likely_ontology], most_likely_ontology
    
    # If still no matches, return based on text analysis
    text_indicators = {
        r".*ase$|.*itis$|.*osis$": ("Disease", "MONDO"),
        r".*ation$|.*ing$|.*sis$": ("Biological_Process", "GO"),
        r".*in$|.*or$": ("Protein", "PR"),
        r".*way$": ("Pathway", "PW"),
        r".*cell.*|.*cyte.*": ("Cell", "CL")
    }
    
    for pattern, (label, ontology) in text_indicators.items():
        if re.search(pattern, entity.lower()):
            return label, ontology
            
    # If absolutely nothing matches, default to most generic detected category
    if "process" in entity.lower() or "activity" in entity.lower():
        return "Biological_Process", "GO"
    elif "disease" in entity.lower() or "syndrome" in entity.lower():
        return "Disease", "MONDO"
    else:
        return "Biological_Process", "GO"  # Most general biological category

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
        rel_type = sanitize_label(row['REL_type'])
        pmid = row['PMID']
        evidence = row['Evidences']
        source = row['Source']

        # Determine labels and namespaces
        start_label, start_namespace = determine_entity_label(start_entity)
        end_label, end_namespace = determine_entity_label(end_entity)

        # Validate labels
        if not start_label or not end_label:
            raise ValueError(f"Could not determine label for entities: {start_entity} -> {end_entity}")

        # Sanitize labels
        start_label = sanitize_label(start_label)
        end_label = sanitize_label(end_label)

        # Modified query to only create the relationship if it doesn't exist
        query = """
        MATCH (start:%s {name: $start_entity})
        MATCH (end:%s {name: $end_entity})
        MERGE (start)-[rel:%s {pmid: $pmid}]->(end)
        ON CREATE SET 
            rel.evidence = $evidence,
            rel.source = $source,
            rel.created_at = timestamp()
        """ % (start_label, end_label, rel_type)

        try:
            # First, ensure nodes exist
            create_nodes_query = """
            MERGE (start:%s {name: $start_entity})
            ON CREATE SET 
                start.namespace = $start_namespace,
                start.created_at = timestamp()
            ON MATCH SET 
                start.last_seen = timestamp()
                
            MERGE (end:%s {name: $end_entity})
            ON CREATE SET 
                end.namespace = $end_namespace,
                end.created_at = timestamp()
            ON MATCH SET 
                end.last_seen = timestamp()
            """ % (start_label, end_label)

            tx.run(
                create_nodes_query,
                start_entity=start_entity,
                end_entity=end_entity,
                start_namespace=start_namespace or "UNKNOWN",
                end_namespace=end_namespace or "UNKNOWN"
            )

            # Then create relationship
            tx.run(
                query,
                start_entity=start_entity,
                end_entity=end_entity,
                pmid=str(pmid),
                evidence=str(evidence),
                source=str(source)
            )
            print(f"Successfully processed: {start_entity} ({start_label}) -> {end_entity} ({end_label})")
        except Exception as e:
            print(f"Error processing: {start_entity} -> {end_entity}")
            print(f"Labels: {start_label} -> {end_label}")
            print(f"Error: {str(e)}")
            raise e
        
def query_gene_ontology_fuzzy(entity, threshold=0.7):
    """
    Query Gene Ontology (GO) for approximate string matching.
    """
    url = f"https://www.ebi.ac.uk/ols/api/ontologies/go/terms?q={entity.lower()}"
    response = requests.get(url)

    if response.status_code == 200:
        results = response.json()
        terms = results.get("_embedded", {}).get("terms", [])

        if terms:
            for term in terms:
                label = term.get("label", "").lower()
                if entity.lower() in label or \
                        get_close_matches(entity.lower(), [label], n=1, cutoff=threshold):
                    return "Biological_Process", "GO"

    return None, None


def query_ols_fuzzy(entity, threshold=0.7):
    """
    Query OLS for ontology information with case-insensitive and approximate string matching,
    including checks for Disease Ontology and Gene Ontology.
    
    Parameters:
        entity (str): The entity to search for.
        threshold (float): The similarity threshold for approximate matches (0 to 1).
        
    Returns:
        tuple: The type of entity and its ontology prefix if found; otherwise, (None, None).
    """
    # First, check specifically in Disease Ontology
    label, namespace = query_disease_ontology_fuzzy(entity, threshold)
    if label:
        return label, namespace

    # Then, check in Gene Ontology
    label, namespace = query_gene_ontology_fuzzy(entity, threshold)
    if label:
        return label, namespace

    # Generic OLS query for all ontologies
    url = f"https://www.ebi.ac.uk/ols/api/search?q={entity.lower()}"
    response = requests.get(url)
    
    if response.status_code == 200:
        results = response.json()
        docs = results.get("response", {}).get("docs", [])
        
        if docs:
            for doc in docs:
                if "ontology_prefix" in doc:
                    ontology = doc["ontology_prefix"]
                    label = doc.get("label", "").lower()
                    
                    # Approximate string matching
                    if entity.lower() in label or \
                       get_close_matches(entity.lower(), [label], n=1, cutoff=threshold):
                        
                        # Map ontology prefixes to entity types
                        if ontology in {"DOID", "MESH", "MESHD"}:
                            #print("test...", ontology)
                            return "Disease", ontology
                        elif ontology == "CHEBI":
                            return "Chemical", ontology
                        elif ontology == "GO":
                            return "Biological_Process", ontology
    
    return None, None

# Usage example:
def main():
    # Load CSV file with triples
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, "hypothesis_pmid_evidences.csv")
    data = pd.read_csv(file_path, encoding="latin1")
    data.fillna("Unknown_Entity", inplace=True)

    # Prepare the data for upload
    triples = data[["Source", "Start Node", "End Node", "REL_type", "PMID", "Evidences"]]

    #Upload data to Neo4jAura DB REMOTELY
    neo4j_uploader = Neo4jUploader(
        "neo4j+s://09f8d4e9.databases.neo4j.io",
        "neo4j",
        "pass"
    )
    
    try:
        neo4j_uploader.upload_triples(triples)
        print("Data upload to Neo4j completed successfully.")
    finally:
        neo4j_uploader.close()

if __name__ == "__main__":
    main()
#%% Upload selected comorbidty triples from all sources to database
import pandas as pd
import requests
from fuzzywuzzy import process, fuzz
from neo4j import GraphDatabase
import re

def sanitize_label(label):
    """Sanitize labels for Neo4j compatibility."""
    if not isinstance(label, str):
        return "Unknown"
    return label.replace("-", "_").replace(" ", "_").replace(".", "_").upper()

def determine_label_from_namespace(namespace):
    """
    Map namespaces to their respective node labels with expanded ontology coverage.
    Includes broader coverage of biomedical ontologies and their variations.
    """
    namespace_mapping = {
        # Disease-related ontologies
        "MONDO": "Disease",
        "DOID": "Disease",
        "DO": "Disease",
        "MESH": "Disease",
        "OMIM": "Disease",
        "NCIT": "Disease",
        "ICD10": "Disease",
        "ICD9": "Disease",
        "UMLS": "Disease",
        
        # Biological process ontologies
        "GO": "Biological_Process",
        "GO_BP": "Biological_Process",
        "GO_MF": "Molecular_Function",
        "GO_CC": "Cellular_Component",
        
        # Chemical and drug ontologies
        "CHEBI": "Chemical",
        "DRUGBANK": "Drug",
        "PUBCHEM": "Chemical",
        "CHEMBL": "Chemical",
        
        # Phenotype ontologies
        "HP": "Phenotype",
        "MP": "Phenotype",
        "HPO": "Phenotype",
        
        # Anatomy ontologies
        "UBERON": "Anatomy",
        "FMA": "Anatomy",
        "CARO": "Anatomy",
        
        # Cell ontologies
        "CL": "Cell",
        "CELL": "Cell",
        
        # Protein and gene ontologies
        "PR": "Protein",
        "UNIPROT": "Protein",
        "HGNC": "Gene",
        "ENSEMBL": "Gene",
        "NCBI_GENE": "Gene",
        
        # Pathway ontologies
        "PW": "Pathway",
        "KEGG": "Pathway",
        "REACTOME": "Pathway",
        
        # Experimental and phenotype ontologies
        "EFO": "Experimental_Factor",
        "OBI": "Experimental_Factor",
        
        # Symptom ontologies
        "SYMP": "Symptom",
        "SYMPTOM": "Symptom",
        
        # Other common ontologies
        "ECO": "Evidence",
        "IAO": "Information_Artifact",
        "PATO": "Quality",
        "RO": "Relation",
    }
    
    if not isinstance(namespace, str):
        return "Unknown"
    
    clean_namespace = namespace.upper().strip()
    
    if ":" in clean_namespace:
        parts = clean_namespace.split(":")
        clean_namespace = f"{parts[0]}_{parts[1]}" if len(parts) > 1 else parts[0]
    
    return namespace_mapping.get(clean_namespace, "Unknown")

def normalize_entity(entity):
    """Enhanced normalization for entity strings, especially for disease names."""
    if not isinstance(entity, str):
        return "unknown"
    
    # Convert underscores and hyphens to spaces
    entity = entity.replace("_", " ").replace("-", " ")
    
    # Remove special characters but keep apostrophes for diseases like Alzheimer's
    entity = re.sub(r'[^a-zA-Z0-9\s\']', '', entity)
    
    # Handle specific disease patterns
    entity = re.sub(r"s disease$", "'s disease", entity, flags=re.IGNORECASE)
    entity = re.sub(r"s syndrome$", "'s syndrome", entity, flags=re.IGNORECASE)
    
    # Remove multiple spaces and strip
    entity = " ".join(entity.split())
    
    return entity.lower().strip()

def query_ontology(entity, ontology):
    """Enhanced ontology query with better error handling and pagination."""
    normalized_entity = normalize_entity(entity)
    base_url = "https://www.ebi.ac.uk/ols/api/ontologies"
    
    # Add specific handling for disease ontologies
    if ontology.lower() == "do":
        search_urls = [
            f"{base_url}/{ontology}/terms?q={normalized_entity}",
            f"{base_url}/{ontology}/terms?q={normalized_entity}%20disease",
            f"{base_url}/{ontology}/terms?q={normalized_entity.replace(' disease', '')}"
        ]
    else:
        search_urls = [f"{base_url}/{ontology}/terms?q={normalized_entity}"]
    
    all_terms = []
    for url in search_urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                results = response.json()
                terms = results.get("_embedded", {}).get("terms", [])
                all_terms.extend(terms)
        except requests.exceptions.RequestException as e:
            print(f"Error querying {url}: {e}")
            continue
    
    return all_terms

def fuzzy_match_entity(entity):
    """Enhanced fuzzy matching with better disease recognition."""
    entity = normalize_entity(entity)
    
    # Prioritize disease ontologies first
    disease_ontologies = ["doid", "mondo", "mesh", "efo", "hp"]
    other_ontologies = ["go", "ncit", "uberon", "cl"]
    
    # Check if the entity might be a disease
    disease_keywords = ["disease", "syndrome", "disorder", "condition", "deficiency"]
    likely_disease = any(keyword in entity.lower() for keyword in disease_keywords)
    
    # Choose ontology order based on entity characteristics
    ontologies = disease_ontologies + other_ontologies if likely_disease else other_ontologies + disease_ontologies
    
    best_overall_match = None
    highest_overall_score = 0
    matched_ontology = "Unknown"
    
    for ontology in ontologies:
        terms = query_ontology(entity, ontology)
        
        for term in terms:
            label = term.get("label", "")
            synonyms = term.get("synonyms", [])
            
            # Check main label
            if label:
                score = fuzz.ratio(normalize_entity(entity), normalize_entity(label))
                if score > highest_overall_score:
                    highest_overall_score = score
                    best_overall_match = label
                    matched_ontology = ontology.upper()
            
            # Check synonyms
            for synonym in synonyms:
                score = fuzz.ratio(normalize_entity(entity), normalize_entity(synonym))
                if score > highest_overall_score:
                    highest_overall_score = score
                    best_overall_match = label  # Use main label even if synonym matches
                    matched_ontology = ontology.upper()
    
    # Lower threshold for disease terms to catch more matches
    threshold = 55 if likely_disease else 60
    
    if highest_overall_score >= threshold:
        return best_overall_match, matched_ontology
    
    return None, "Unknown"


def check_chebi(entity):
    """Query ChEBI for chemical compounds."""
    url = f"https://www.ebi.ac.uk/webservices/chebi/2.0/test/getCompleteEntity?chebiId={entity}"
    try:
        response = requests.get(
            f"https://www.ebi.ac.uk/ols/api/search?q={entity}&ontology=chebi",
            timeout=10
        )
        if response.status_code == 200:
            results = response.json()
            docs = results.get("response", {}).get("docs", [])
            for doc in docs:
                if fuzz.ratio(entity.lower(), doc.get("label", "").lower()) > 80:
                    return True
    except:
        pass
    return False

def normalize_term(term):
    """Basic term normalization."""
    term = term.lower()
    term = re.sub(r'[_\-]', ' ', term)  # Replace underscores and hyphens with spaces
    term = re.sub(r'[^\w\s]', '', term)  # Remove other special characters
    return ' '.join(term.split())  # Normalize spaces

def get_variations(term):
    """Generate term variations using pattern matching."""
    variations = set()
    normalized = normalize_term(term)
    variations.add(normalized)
    
    # Tokenize
    words = normalized.split()
    
    # Handle each word's variations
    new_word_sets = []
    for word in words:
        word_vars = {word}
        
        # Basic word variations
        if word.endswith('ing'):
            base = word[:-3]
            word_vars.update([base, base + 'ed', base + 's'])
        elif word.endswith('ed'):
            base = word[:-2]
            word_vars.update([base, base + 'ing', base + 's'])
        elif word.endswith('ies'):
            word_vars.add(word[:-3] + 'y')
        elif word.endswith('es'):
            word_vars.update([word[:-2], word[:-1]])
        elif word.endswith('s') and not any(word.endswith(x) for x in ['ss', 'us', 'is']):
            word_vars.add(word[:-1])
        elif word.endswith('y'):
            word_vars.add(word[:-1] + 'ies')
        elif word.endswith('is'):
            word_vars.add(word[:-2] + 'es')
        else:
            word_vars.update([word + 's', word + 'es'])
            
        new_word_sets.append(word_vars)
    
    # Generate combinations
    from itertools import product
    for combo in product(*new_word_sets):
        variations.add(' '.join(combo))
        
    # Add variations with different word orders for 2-3 word terms
    if 1 < len(words) <= 3:
        from itertools import permutations
        for perm in permutations(words):
            base_term = ' '.join(perm)
            variations.add(base_term)
    
    return variations

def score_match(search_term, mesh_term):
    """Enhanced scoring function for term matching."""
    search_term = search_term.lower()
    mesh_term = mesh_term.lower()
    
    # Calculate different types of match scores
    token_set = fuzz.token_set_ratio(search_term, mesh_term)
    token_sort = fuzz.token_sort_ratio(search_term, mesh_term)
    partial = fuzz.partial_ratio(search_term, mesh_term)
    
    # Get word sets for exact word matching
    search_words = set(search_term.split())
    mesh_words = set(mesh_term.split())
    
    # Calculate exact word matches
    exact_matches = len(search_words.intersection(mesh_words))
    word_match_ratio = exact_matches / max(len(search_words), len(mesh_words))
    
    # Weighted scoring
    base_score = max(token_set, token_sort, partial)
    word_match_boost = word_match_ratio * 20  # Up to 20 point boost for exact word matches
    
    final_score = min(100, base_score + word_match_boost)
    return final_score

def query_mesh_terms(entity):
    """Query MeSH with flexible fuzzy matching."""
    def normalize_term(term):
        """Simple normalization: lowercase and remove separators."""
        return re.sub(r'[_\-]', ' ', term.lower()).strip()

    try:
        # Normalize the search term
        search_term = normalize_term(entity)
        
        # Single query to MESH with the normalized term
        response = requests.get(
            f"https://www.ebi.ac.uk/ols/api/search?q={search_term}&ontology=mesh&rows=50",
            timeout=10
        )
        
        if response.status_code == 200:
            results = response.json()
            docs = results.get("response", {}).get("docs", [])
            
            best_score = 0
            best_result = None
            best_match = None
            
            for doc in docs:
                label = doc.get("label", "").lower()
                synonyms = doc.get("synonym", [])
                if isinstance(synonyms, str):
                    synonyms = [synonyms]
                
                terms_to_check = [label] + [s.lower() for s in synonyms if isinstance(s, str)]
                
                for term in terms_to_check:
                    # Use multiple fuzzy matching algorithms
                    token_set = fuzz.token_set_ratio(search_term, term)
                    partial = fuzz.partial_ratio(search_term, term)
                    token_sort = fuzz.token_sort_ratio(search_term, term)
                    
                    # Take the highest score
                    score = max(token_set, partial, token_sort)
                    
                    if score > best_score:
                        best_score = score
                        best_match = term
                        tree_numbers = doc.get("obo_id", [])
                        if isinstance(tree_numbers, str):
                            tree_numbers = [tree_numbers]
                        
                        # Map tree numbers to entity types
                        for tree_num in tree_numbers:
                            if tree_num.startswith(('MESH:C', 'MESH:D')):
                                best_result = ("Disease", "MESH")
                            elif tree_num.startswith('MESH:F'):
                                best_result = ("Biological_Process", "MESH")
                            elif tree_num.startswith('MESH:G'):
                                best_result = ("Phenotype", "MESH")
                            elif tree_num.startswith('MESH:E'):
                                best_result = ("Clinical_Feature", "MESH")
                            elif tree_num.startswith('MESH:B'):
                                best_result = ("Biological_Process", "MESH")
                        
                        if not best_result:
                            best_result = ("Disease", "MESH")

            if best_score >= 60:
                print(f"Matched '{entity}' to '{best_match}' (score: {best_score})")
                return best_result
                        
    except Exception as e:
        print(f"Error in MESH query for {entity}: {e}")
    
    return None, None
 
def determine_entity_type(entity):
    """Enhanced entity type detection with better MESH integration."""
    entity = str(entity).strip().upper()
    
    # Add common medical terms that should be in MESH
    medical_terms = {
        "SMOKING", "COGNITIVE_DYSFUNCTION", "BRAIN_INJURY", "BRAIN_INJURIES",
        "MEMORY", "LEARNING", "BEHAVIOR", "COGNITION", "MENTAL_HEALTH",
        "PHYSICAL_ACTIVITY", "EXERCISE", "DIET", "NUTRITION"
    }
    
    # Check MESH first for medical terms
    if entity in medical_terms or "_" in entity:
        mesh_result = query_mesh_terms(entity)
        if mesh_result[0]:
            return mesh_result
    
    # Rest of your existing determine_entity_type function...

    entity = str(entity).strip().upper()
    
    # Common non-gene terms that might be mistaken for genes
    non_gene_terms = {
        # Diseases/Conditions
        "STROKE", "PAIN", "STRESS", "CANCER", "FEVER", "COLD", 
        "TREMOR", "SHOCK", "AIDS", "RAGE",
        
        # Chemicals/Neurotransmitters
        "DOPAMINE", "SEROTONIN", "GABA", "GLUTAMATE", "ACETYLCHOLINE",
        "NOREPINEPHRINE", "HISTAMINE", "GLYCINE", "ASPARTATE",
        
        # Common biological terms
        "INSULIN", "GLUCOSE", "PROTEIN", "PEPTIDE", "HORMONE",
        "CYTOKINE", "ENZYME", "RECEPTOR"
    }
    
    if entity in non_gene_terms:
        # Check CHEBI first for chemical compounds
        if check_chebi(entity):
            return "Chemical", "CHEBI"
        return None, None  # Let it fall through to comprehensive ontology search
    
    # Check for chemical/drug patterns
    chemical_patterns = [
        r'.*AMINE$',
        r'.*INE$',
        r'.*OL$',
        r'.*IC_ACID$',
        r'.*STEROID$'
    ]
    
    if any(re.search(pattern, entity) for pattern in chemical_patterns):
        if check_chebi(entity):
            return "Chemical", "CHEBI"
    
    # Gene patterns (more restrictive)
    gene_patterns = [
        r'^[A-Z][A-Z0-9]{1,7}[0-9]$',  # Must end with a number
        r'^[A-Z]{2,4}\d{1,2}[A-Z]?$',   # Letters followed by numbers, optional letter
    ]
    
    # Known short gene names
    known_genes = {"IL6", "IL2", "IL1", "TNF", "IL4", "CD4", "CD8", "JAK", "FOS"}
    
    if entity in known_genes:
        return "Gene", "HGNC"
    
    # Check gene patterns with stricter validation
    if any(re.match(pattern, entity) for pattern in gene_patterns):
        # Query HGNC to confirm it's really a gene
        headers = {"Accept": "application/json"}
        response = requests.get(f"https://rest.genenames.org/search/symbol/{entity}", 
                              headers=headers)
        if response.status_code == 200:
            results = response.json()
            if results.get("response", {}).get("numFound", 0) > 0:
                return "Gene", "HGNC"
    
    # SNP patterns
    if re.match(r'^RS\d+$', entity):
        return "SNP", "DBSNP"
    
    # Process/pathway patterns
    process_patterns = [
        r'.*_PATHWAY$',
        r'.*_AGGREGATION$',
        r'.*_PRIMING$',
        r'.*_RESPONSE$',
        r'RESPONSE_TO_.*',
        r'.*_SIGNALING$',
        r'.*_CASCADE$',
        r'.*_METABOLISM$'
    ]
    
    if any(re.match(pattern, entity) for pattern in process_patterns):
        return "Biological_Process", "GO"
        
    # Complex/structure patterns
    if "_COMPLEX" in entity:
        return "Protein_Complex", "GO"
    
    # Phenotype patterns
    phenotype_patterns = [
        r'.*_PHENOTYPE$',
        r'.*_DISORDER$',
        r'.*_DISEASE$',
        r'.*_SYNDROME$',
        r'.*_TOXICITY$',
        r'.*_INJURY$',
        r'.*_LESION$'
    ]
    
    if any(re.match(pattern, entity) for pattern in phenotype_patterns):
        return "Phenotype", "HP"
    
    # Let other cases fall through to comprehensive ontology search
    return None, None
    

def query_hgnc_fuzzy(entity):
    """Enhanced HGNC query with better gene symbol matching."""
    entity = str(entity).strip().upper()
    
    # Common gene name patterns
    if re.match(r'^[A-Z0-9]{2,8}$', entity):
        headers = {"Accept": "application/json"}
        urls = [
            f"https://rest.genenames.org/search/symbol/{entity}",
            f"https://rest.genenames.org/search/alias_symbol/{entity}"
        ]
        
        for url in urls:
            try:
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    results = response.json()
                    if results.get("response", {}).get("numFound", 0) > 0:
                        return "Gene", "HGNC"
            except:
                continue
    
    return None, None

def query_go_terms(entity):
    """Query Gene Ontology for biological processes."""
    url = f"https://www.ebi.ac.uk/ols/api/ontologies/go/terms?q={entity}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            results = response.json()
            terms = results.get("_embedded", {}).get("terms", [])
            
            for term in terms:
                label = term.get("label", "").lower()
                if fuzz.ratio(entity.lower(), label) > 70:
                    return "Biological_Process", "GO"
    except:
        pass
    return None, None
def query_ols_fuzzy(entity, threshold=60):
    """Enhanced OLS query with better MESH support."""
    entity = str(entity).strip()
    
    # First try MESH for medical terms
    mesh_result = query_mesh_terms(entity)
    if mesh_result[0]:
        return mesh_result
    
    # Prioritize certain ontologies based on entity characteristics
    ontology_priorities = {
        "MESH": ("Disease", 1),      # Added MESH as high priority
        "MONDO": ("Disease", 2),
        "DOID": ("Disease", 3),
        "CHEBI": ("Chemical", 1),
        "GO": ("Biological_Process", 1),
        "HP": ("Phenotype", 1),
        "UBERON": ("Anatomy", 1),
    }
    
    """Query OLS with chemical-aware prioritization."""
    entity = str(entity).strip()
    
    # First check CHEBI specifically for potential chemicals
    try:
        response = requests.get(
            f"https://www.ebi.ac.uk/ols/api/search?q={entity}&ontology=chebi",
            timeout=10
        )
        if response.status_code == 200:
            results = response.json()
            docs = results.get("response", {}).get("docs", [])
            for doc in docs:
                if fuzz.ratio(entity.lower(), doc.get("label", "").lower()) > 80:
                    return "Chemical", "CHEBI"
    except:
        pass

    """
    Query OLS for ontology information with fuzzy matching.
    
    Args:
        entity (str): Entity name to search for
        threshold (int): Minimum similarity score (0-100) to consider a match
    
    Returns:
        tuple: (label, namespace) or (None, "Unknown") if no match found
    """
    entity = str(entity).strip()
    base_url = "https://www.ebi.ac.uk/ols/api/search"
    
    try:
        response = requests.get(
            f"{base_url}?q={entity}&rows=20",
            timeout=10
        )
        
        if response.status_code != 200:
            return None, "Unknown"
            
        results = response.json()
        docs = results.get("response", {}).get("docs", [])
        
        best_match = None
        highest_score = 0
        best_ontology = "Unknown"
        
        ontology_priorities = {
            # Disease ontologies
            "MONDO": ("Disease", 1),
            "DOID": ("Disease", 2),
            "MESH": ("Disease", 3),
            
            # Gene/protein ontologies
            "HGNC": ("Gene", 1),
            "PR": ("Protein", 2),
            
            # Biological process ontologies
            "GO": ("Biological_Process", 1),
            "PW": ("Pathway", 2),
            
            # Phenotype ontologies
            "HP": ("Phenotype", 1),
            "MP": ("Phenotype", 2),
            
            # Chemical ontologies
            "CHEBI": ("Chemical", 1),
            
            # Anatomy ontologies
            "UBERON": ("Anatomy", 1),
            
            # Cell ontologies
            "CL": ("Cell", 1)
        }
        
        for doc in docs:
            label = doc.get("label", "")
            ontology = doc.get("ontology_prefix", "")
            
            if label and ontology in ontology_priorities:
                score = fuzz.ratio(entity.lower(), label.lower())
                
                # Adjust score based on ontology priority
                priority_boost = (10 - ontology_priorities[ontology][1]) / 10
                adjusted_score = score * priority_boost
                
                if adjusted_score > highest_score:
                    highest_score = adjusted_score
                    best_match = ontology_priorities[ontology][0]
                    best_ontology = ontology
        
        if highest_score >= threshold:
            return best_match, best_ontology
            
    except Exception as e:
        print(f"Error querying OLS for {entity}: {e}")
    
    return None, "Unknown"

def query_all_ontologies(entity):
    """Query multiple ontologies with flexible fuzzy matching."""
    def normalize_term(term):
        """Simple normalization: lowercase and remove separators."""
        return re.sub(r'[_\-]', ' ', term.lower()).strip()

    try:
        # Normalize the search term
        search_term = normalize_term(entity)
        
        # Ontology priorities and their mappings
        ontology_configs = [
            ("mesh", {"score_boost": 1.2, "default_type": "Disease"}),
            ("go", {"score_boost": 1.1, "default_type": "Biological_Process"}),
            ("doid", {"score_boost": 1.1, "default_type": "Disease"}),
            ("hp", {"score_boost": 1.0, "default_type": "Phenotype"}),
            ("chebi", {"score_boost": 1.1, "default_type": "Chemical"}),
            ("mondo", {"score_boost": 1.0, "default_type": "Disease"}),
        ]
        
        best_overall_score = 0
        best_overall_result = None
        best_match = None
        
        # Query each ontology
        for ontology, config in ontology_configs:
            response = requests.get(
                f"https://www.ebi.ac.uk/ols/api/search?q={search_term}&ontology={ontology}&rows=50",
                timeout=10
            )
            
            if response.status_code == 200:
                results = response.json()
                docs = results.get("response", {}).get("docs", [])
                
                for doc in docs:
                    label = doc.get("label", "").lower()
                    synonyms = doc.get("synonym", [])
                    if isinstance(synonyms, str):
                        synonyms = [synonyms]
                    
                    terms_to_check = [label] + [s.lower() for s in synonyms if isinstance(s, str)]
                    
                    for term in terms_to_check:
                        # Use multiple fuzzy matching algorithms
                        token_set = fuzz.token_set_ratio(search_term, term)
                        partial = fuzz.partial_ratio(search_term, term)
                        token_sort = fuzz.token_sort_ratio(search_term, term)
                        
                        # Take the highest score and apply ontology-specific boost
                        score = max(token_set, partial, token_sort) * config["score_boost"]
                        
                        if score > best_overall_score:
                            best_overall_score = score
                            best_match = term
                            
                            # Determine type based on ontology and tree numbers
                            if ontology == "go":
                                # Check GO aspects (P, F, C)
                                namespace = doc.get("obo_id", "")
                                if "GO:P" in str(namespace):
                                    best_overall_result = ("Biological_Process", "GO")
                                elif "GO:F" in str(namespace):
                                    best_overall_result = ("Molecular_Function", "GO")
                                elif "GO:C" in str(namespace):
                                    best_overall_result = ("Cellular_Component", "GO")
                                else:
                                    best_overall_result = ("Biological_Process", "GO")
                            elif ontology == "mesh":
                                tree_numbers = doc.get("obo_id", [])
                                if isinstance(tree_numbers, str):
                                    tree_numbers = [tree_numbers]
                                
                                for tree_num in tree_numbers:
                                    if tree_num.startswith(('MESH:C', 'MESH:D')):
                                        best_overall_result = ("Disease", "MESH")
                                    elif tree_num.startswith('MESH:F'):
                                        best_overall_result = ("Biological_Process", "MESH")
                                    elif tree_num.startswith('MESH:G'):
                                        best_overall_result = ("Phenotype", "MESH")
                                    elif tree_num.startswith('MESH:E'):
                                        best_overall_result = ("Clinical_Feature", "MESH")
                                    elif tree_num.startswith('MESH:B'):
                                        best_overall_result = ("Biological_Process", "MESH")
                            else:
                                best_overall_result = (config["default_type"], ontology.upper())

            if best_overall_score >= 60:  # Threshold check after each ontology
                print(f"Matched '{entity}' to '{best_match}' in {ontology} (score: {best_overall_score})")
                return best_overall_result
                        
    except Exception as e:
        print(f"Error in ontology query for {entity}: {e}")
    
    return None, None

def determine_entity_label(entity):
    """Enhanced entity label determination with better pattern matching."""
    if not isinstance(entity, str):
        return "Unknown", "Unknown"
    
    entity = entity.strip()
    
    # First try pattern matching
    label, namespace = determine_entity_type(entity)
    if label:
        return label, namespace
    
    # Try HGNC for genes
    label, namespace = query_hgnc_fuzzy(entity)
    if label:
        return label, namespace
    
    # Try GO for biological processes
    if "_" in entity:
        label, namespace = query_go_terms(entity.replace("_", " "))
        if label:
            return label, namespace
    
    # Additional patterns for specific cases
    if "_COMPLEX" in entity.upper():
        return "Protein_Complex", "GO"
    
    if any(word in entity.upper() for word in ["TOXICITY", "INJURY", "LESION"]):
        return "Phenotype", "HP"
    
    if "VIRUS" in entity.upper():
        return "Organism", "NCBITaxon"
    
    # For remaining unknown cases, try general ontology search
    # For remaining unknown cases, try comprehensive ontology search
    label, namespace = query_all_ontologies(entity)
    if label:
        return label, namespace


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
                    session.execute_write(self.create_relationship, row)
                except Exception as e:
                    print(f"Error uploading row {idx + 1}: {e}")

    @staticmethod
    def create_relationship(tx, row):
        start_entity = sanitize_label(row['Subject'])
        end_entity = sanitize_label(row['Object'])

        # Skip if either node name is empty or just whitespace
        if not start_entity or not end_entity or start_entity.isspace() or end_entity.isspace():
            print(f"Skipping relationship creation due to empty node name: Subject='{start_entity}', Object='{end_entity}'")
            return

        # Get labels and namespaces using the new determine_entity_label function
        start_label, start_namespace = determine_entity_label(row['Subject'])
        end_label, end_namespace = determine_entity_label(row['Object'])

        rel_type = sanitize_label(row['Relation'])
        pmid = str(row['pmid'])
        evidence = str(row['evidence'])
        source = str(row['Source'])

        print(f"Creating relationship: {start_entity} ({start_label}) -> {end_entity} ({end_label})")

        query = f"""
        MERGE (start:{start_label} {{name: $start_entity}})
        MERGE (end:{end_label} {{name: $end_entity}})
        MERGE (start)-[rel:{rel_type} {{
            pmid: $pmid,
            evidence: $evidence,
            source: $source
        }}]->(end)
        """
        
        tx.run(
            query,
            start_entity=start_entity,
            end_entity=end_entity,
            pmid=pmid,
            evidence=evidence,
            source=source,
        )

def main():
    # Load the triples from the CSV file
    file_path = 'all-dbs/cleaned_all_db_association.csv'
    data = pd.read_csv(file_path)

    # Fill NaN values and ensure all fields are strings
    data.fillna("Unknown", inplace=True)
    triples = data[['Subject', 'Subject_Namespace', 'Object', 'Object_Namespace', 'Relation', 'pmid', 'evidence', 'Source']]

    # Initialize the Neo4j uploader
    neo4j_uploader = Neo4jUploader(
        uri= "neo4j+s://09f8d4e9.databases.neo4j.io", #"bolt://localhost:7687",  # Replace with your Neo4j instance URI
        user="neo4j",                # Replace with your Neo4j username
        password= "pass" # Replace with your Neo4j password
    )

    try:
        # Upload the triples
        neo4j_uploader.upload_triples(triples)
    finally:
        # Close the connection
        neo4j_uploader.close()

if __name__ == "__main__":
    main()

# %% GWAS enrichment of database
import pandas as pd
from fuzzywuzzy import fuzz, process
from neo4j import GraphDatabase
import logging
import re

class Neo4jGWASEnricher:
    def __init__(self, shared_variants_file: str, fuzzy_threshold=65):
        """Initialize Neo4j driver and load shared variants data."""
        self.driver = GraphDatabase.driver(
            "neo4j+s://09f8d4e9.databases.neo4j.io", #"bolt://localhost:7687",
            auth=("neo4j", "a8axV58SA-bajOWDieMaYFcf_U6NhG929g3atbsJQxg") # "12345678")
        )
        self.fuzzy_threshold = fuzzy_threshold

        # Load shared variants data
        self.shared_variants_data = pd.read_excel(shared_variants_file)

        logging.info(f"Loaded shared variants data with {len(self.shared_variants_data)} entries")

    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()

    def add_shared_variant_edges(self, tx, trait: str, variant_info: dict):
        """Add edges between traits and shared variants."""
        try:
            query = """
            MATCH (t:Disease {name: $trait})
            MERGE (v:SNP {id: $variant_id})
            ON CREATE SET 
                v.gene = $gene_name,
                v.risk_allele = $risk_allele,
                v.p_value = $p_value,
                v.chromosome = $chromosome,
                v.position = $position
            MERGE (v)-[:IS_RISK_SNP_FOR]->(t)
            """
            tx.run(query,
                   trait=trait,
                   variant_id=variant_info['Variant ID'],
                   gene_name=variant_info['Gene Name'],
                   risk_allele=variant_info['Risk Allele'],
                   p_value=variant_info['P-Value'],
                   chromosome=variant_info['Chromosome'],
                   position=variant_info['Position'])
            logging.info(f"Added SNP {variant_info['Variant ID']} for trait {trait}")
        except Exception as e:
            logging.error(f"Error adding variant {variant_info['Variant ID']} for trait {trait}: {e}")

    def enrich_graph(self):
        """Enrich the graph with shared variants data."""
        with self.driver.session() as session:
            # Fetch all traits in the graph and normalize them
            logging.info("Fetching traits from the graph...")
            trait_query = "MATCH (t:Disease) RETURN t.name AS name"
            graph_traits = {self.normalize_disease_name(row['name']): row['name']
                            for row in session.run(trait_query)}

            # Process each row in the shared variants data
            processed_count = 0
            skipped_count = 0
            for _, row in self.shared_variants_data.iterrows():
                covid_trait = row['DISEASE/TRAIT_COVID']
                neuro_trait = row['DISEASE/TRAIT_NEURO']
                variant_id = row['SNPS']
                gene_name = row.get('MAPPED_GENE_COVID', row.get('MAPPED_GENE_NEURO'))
                risk_allele = row.get('STRONGEST SNP-RISK ALLELE_COVID', row.get('STRONGEST SNP-RISK ALLELE_NEURO'))
                p_value = row.get('P-VALUE_COVID', row.get('P-VALUE_NEURO'))
                chromosome = row['CHR_ID_COVID']
                position = row['CHR_POS_COVID']

                variant_info = {
                    'Variant ID': variant_id,
                    'Gene Name': gene_name,
                    'Risk Allele': risk_allele,
                    'P-Value': p_value,
                    'Chromosome': chromosome,
                    'Position': position
                }

                # Normalize and match COVID trait
                normalized_covid_trait = self.normalize_disease_name(covid_trait)
                covid_match = process.extractOne(normalized_covid_trait, graph_traits.keys(), scorer=fuzz.ratio)

                if covid_match and covid_match[1] >= self.fuzzy_threshold:
                    original_covid_trait = graph_traits[covid_match[0]]
                    session.write_transaction(self.add_shared_variant_edges, original_covid_trait, variant_info)
                    processed_count += 1
                else:
                    skipped_count += 1
                    logging.info(f"Skipped COVID trait: '{covid_trait}' (normalized: '{normalized_covid_trait}')")

                # Normalize and match Neuro trait
                normalized_neuro_trait = self.normalize_disease_name(neuro_trait)
                neuro_match = process.extractOne(normalized_neuro_trait, graph_traits.keys(), scorer=fuzz.ratio)

                if neuro_match and neuro_match[1] >= self.fuzzy_threshold:
                    original_neuro_trait = graph_traits[neuro_match[0]]
                    session.write_transaction(self.add_shared_variant_edges, original_neuro_trait, variant_info)
                    processed_count += 1
                else:
                    skipped_count += 1
                    logging.info(f"Skipped Neuro trait: '{neuro_trait}' (normalized: '{normalized_neuro_trait}')")

            logging.info(f"Completed processing {processed_count} entries. Skipped {skipped_count} entries.")

    @staticmethod
    def normalize_disease_name(disease_name):
        """Normalize disease names for consistent matching."""
        if not disease_name:
            return ""
        disease_name = re.sub(r'\([^)]*\)', '', disease_name)  # Remove parenthetical text
        disease_name = re.sub(r'[^a-zA-Z0-9\s]', '', disease_name)  # Remove special characters
        return disease_name.upper().strip()

# Main function
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        shared_variants_file = "GWAS/shared-variants.xlsx"  # Update with actual file path
        enricher = Neo4jGWASEnricher(shared_variants_file)

        logging.info("Enriching Neo4j graph with shared variants data...")
        enricher.enrich_graph()
        logging.info("Graph enrichment completed successfully.")
    except Exception as e:
        logging.error(f"Error during enrichment: {e}")
    finally:
        if 'enricher' in locals():
            enricher.close()

if __name__ == "__main__":
    main()

# %%
