
# Upload hypothesis space  to Neo4j AuraDB
from operator import delitem
import pandas as pd
import requests
from rapidfuzz import fuzz, process
from neo4j import GraphDatabase
import os
import re
from difflib import get_close_matches

def sanitize_label(label):
    """Sanitize labels for Neo4j compatibility."""
    return str(label).replace("-", "_").replace(" ", "_").replace(".", "_")

def determine_entity_type(entity):
    """
    Pre-check the entity string for obvious entity types before making API calls.
    Returns (is_likely_gene, suggested_label) tuple.
    """
    entity = entity.strip().upper()
    
    # Expanded disease indicators
    disease_indicators = {
        # Common disease suffixes and terms
        "SYNDROME", "DISEASE", "DISORDER", "IMMUNODEFICIENCY", 
        "ITIS", "OSIS", "PATHY", "CANCER", "TUMOR", "DEFICIENCY",
        # COVID-specific terms
        "COVID", "SARS", "SEQUELA", "LONG COVID", "POST-COVID",
        # General medical conditions
        "CONDITION", "INFECTION", "COMPLICATION"
    }
    
    # Common phenotype terms
    phenotype_indicators = {
        # Clinical manifestations
        "PHENOTYPE", "ABNORMAL", "MORPHOLOGY", "APPEARANCE", "TRAIT", 
        "PRESENTATION", "MANIFESTATION", "CLINICAL", "SYMPTOMS", "FEATURES",
        # Physical characteristics
        "LOSS", "MEGALY", "TROPHY", "PLASIA", "GRADE",
        # Common phenotype terms
        "DISABILITY", "IMPAIRMENT", "DEFICIT", "DYSFUNCTION",
        # Inheritance patterns
        "INHERITANCE", "DOMINANT", "RECESSIVE", "LINKED",
        # Specific phenotypes
        "SEIZURE", "ATAXIA", "PALSY", "DYSTROPHY", "WEAKNESS",
        "RETARDATION", "DEGENERATION", "DEFICIT",
        # Post-disease sequelae
        "SEQUELA", "COMPLICATION", "AFTER", "POST"
    }
    
    # Process indicators
    process_indicators = {
        "PATHWAY", "SIGNALING", "INFLAMMATION", "RESPONSE", 
        "REGULATION", "METABOLISM", "SYNTHESIS", "ACTIVITY"
    }
    
    # Protein/molecule indicators
    protein_indicators = {
        "CYTOKINE", "CYTOKINES", "INTERLEUKIN", "CHEMOKINE",
        "FACTOR", "PROTEIN", "RECEPTOR", "HORMONE", "ENZYME"
    }
    
    # Known gene patterns
    gene_indicators = {"APOE4", "BRCA1", "TP53"}
    
    # First check for COVID-related terms
    if any(covid_term in entity for covid_term in ["COVID", "SARS-COV", "SEQUELA OF COVID"]):
        return False, "Disease"
    
    # Check for disease terms
    if any(term in entity for term in disease_indicators):
        return False, "Disease"
    
    # Check for phenotype indicators
    if any(term in entity for term in phenotype_indicators):
        return False, "Phenotype"
    
    # Check for protein/molecule indicators
    if any(term in entity for term in protein_indicators):
        return False, "Protein"
    
    # Check for process indicators
    if any(term in entity for term in process_indicators):
        return False, "Biological_Process"
    
    # Check for gene patterns last
    if (entity.startswith("APOE") or 
        any(gene in entity for gene in gene_indicators) or
        (len(entity) <= 5 and entity.isalnum())):  # Most gene symbols are short
        return True, "Gene"
    
    # Default case
    return False, None

def query_hpo_fuzzy(entity, threshold=0.6):
    """
    Enhanced query for Human Phenotype Ontology (HPO) terms with proper case handling.
    Returns (label_type, namespace, hpo_id) tuple.
    """
    # Clean and format the search term - keep both original and lowercase versions
    original_term = entity.strip()
    search_term = original_term.lower()
    
    # Debug print
    print(f"Searching HPO for term: {original_term} (lowercase: {search_term})")
    
    # Generate variations of the term
    variations = [
        search_term,                     # lowercase
        original_term,                   # original case
        original_term.upper(),           # uppercase
        original_term.title(),           # Title Case
        search_term.replace("-", " "),   # spaces instead of hyphens
        search_term.replace(" ", "-"),   # hyphens instead of spaces
    ]
    
    # Remove duplicates while preserving order
    variations = list(dict.fromkeys(variations))
    print(f"Trying variations: {variations}")
    
    # Try exact match first with all variations
    for variant in variations:
        url = f"https://www.ebi.ac.uk/ols/api/ontologies/hp/terms?q={variant}&exact=true"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                results = response.json()
                terms = results.get("_embedded", {}).get("terms", [])
                
                if terms:
                    for term in terms:
                        label = term.get("label", "")
                        ontology_id = term.get("obo_id")
                        
                        # Debug print
                        print(f"Found term: {label} with ID: {ontology_id}")
                        
                        # Compare lowercase versions for matching
                        if label.lower() == search_term:
                            return "Phenotype", "HP", ontology_id
        except Exception as e:
            print(f"Error in exact match query: {str(e)}")
    
    # Try partial match if exact match fails
    for variant in variations:
        url = f"https://www.ebi.ac.uk/ols/api/ontologies/hp/terms?q={variant}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                results = response.json()
                terms = results.get("_embedded", {}).get("terms", [])
                
                if terms:
                    for term in terms:
                        label = term.get("label", "")
                        ontology_id = term.get("obo_id")
                        
                        # Compare with fuzzy matching
                        score = fuzz.ratio(search_term, label.lower())
                        print(f"Checking {label} (score: {score})")
                        
                        if score >= threshold * 100:
                            return "Phenotype", "HP", ontology_id
        except Exception as e:
            print(f"Error in partial match query: {str(e)}")
    
    # Try OLS search as final fallback
    url = f"https://www.ebi.ac.uk/ols/api/search?q={search_term}&ontology=hp"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            results = response.json()
            docs = results.get("response", {}).get("docs", [])
            
            if docs:
                for doc in docs:
                    label = doc.get("label", "")
                    ontology_id = doc.get("obo_id")
                    score = fuzz.ratio(search_term, label.lower())
                    
                    print(f"OLS result: {label} (score: {score})")
                    
                    if score >= threshold * 100:
                        return "Phenotype", "HP", ontology_id
    except Exception as e:
        print(f"Error in OLS fallback search: {str(e)}")
    
    return None, None, None


def query_disease_ontology_fuzzy(entity, threshold=0.7):
    """
    Query Disease Ontology (DO) for approximate string matching.
    Returns (label_type, namespace, ontology_id) tuple.
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
                    ontology_id = term.get("obo_id") or term.get("short_form")
                    return "Disease", "DO", ontology_id
    
    return None, None, None

def query_hgnc_fuzzy(entity):
    """
    Query HGNC database for gene symbols.
    Returns (label_type, namespace, hgnc_id) tuple.
    """
    entity = entity.strip().upper()
    headers = {"Accept": "application/json"}
    
    # Try exact symbol match
    url = f"https://rest.genenames.org/search/symbol/{entity}"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        results = response.json()
        if results.get("response", {}).get("numFound", 0) > 0:
            doc = results["response"]["docs"][0]
            hgnc_id = doc.get("hgnc_id")
            return "Gene", "HGNC", hgnc_id
            
    # Try alias search
    url = f"https://rest.genenames.org/search/alias_symbol/{entity}"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        results = response.json()
        if results.get("response", {}).get("numFound", 0) > 0:
            doc = results["response"]["docs"][0]
            hgnc_id = doc.get("hgnc_id")
            return "Gene", "HGNC", hgnc_id
    
    return None, None, None

def query_gene_ontology_fuzzy(entity, threshold=0.7):
    """
    Query Gene Ontology (GO) for approximate string matching.
    Returns (label_type, namespace, go_id) tuple.
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
                    ontology_id = term.get("obo_id") or term.get("short_form")
                    return "Biological_Process", "GO", ontology_id

    return None, None, None
def query_mammalian_phenotype_fuzzy(entity, threshold=0.7):
    """
    Query Mammalian Phenotype Ontology (MP) for phenotype terms.
    Returns (label_type, namespace, mp_id) tuple.
    """
    url = f"https://www.ebi.ac.uk/ols/api/ontologies/mp/terms?q={entity.lower()}"
    response = requests.get(url)
    
    if response.status_code == 200:
        results = response.json()
        terms = results.get("_embedded", {}).get("terms", [])
        
        if terms:
            for term in terms:
                label = term.get("label", "").lower()
                if entity.lower() in label or \
                   get_close_matches(entity.lower(), [label], n=1, cutoff=threshold):
                    ontology_id = term.get("obo_id") or term.get("short_form")
                    return "Phenotype", "MP", ontology_id
    
    return None, None, None

def query_mesh_nih(entity):
    """
    Query NIH MESH API for terms and classify based on qualifiers.
    Returns (label_type, namespace, mesh_id) tuple.
    """
    search_term = entity.strip().replace(" ", "+")
    base_url = "https://id.nlm.nih.gov/mesh"
    
    # Try exact match for MESH term
    exact_url = f"{base_url}/lookup/descriptor?label={search_term}&match=exact&limit=1"
    
    try:
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(exact_url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            if data:  # If we got any results
                mesh_id = data[0].get('resource', '').split('/')[-1]
                
                # Get detailed information about the term
                detail_url = f"{base_url}/lookup/details?descriptor={mesh_id}"
                detail_response = requests.get(detail_url, headers=headers)
                
                if detail_response.status_code == 200:
                    details = detail_response.json()
                    
                    # Check qualifiers to infer category
                    qualifiers = details.get('qualifiers', [])
                    print("Qualifiers:", qualifiers)  # Debugging the qualifiers
                    
                    # Check if the term has specific categories
                    if any(q.get("label", "").lower() in ["protein", "cytokine", "enzyme"] for q in qualifiers):
                        return "Protein", "MESH", f"MESH:{mesh_id}"
                    elif any(q.get("label", "").lower() in ["pathology", "disease", "disorder", "syndrome"] for q in qualifiers):
                        return "Disease", "MESH", f"MESH:{mesh_id}"
                    elif any(q.get("label", "").lower() in ["physiology","pathway", "process", "signaling"] for q in qualifiers):
                        return "Biological_Process", "MESH", f"MESH:{mesh_id}"
                    else:
                        # If no clear classification, return a general term like "Chemical"
                        return "Chemical", "MESH", f"MESH:{mesh_id}"
        
        # If exact match fails, try partial match
        partial_url = f"{base_url}/lookup/descriptor?label={search_term}&match=contains&limit=5"
        response = requests.get(partial_url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                mesh_id = data[0].get('resource', '').split('/')[-1]
                detail_url = f"{base_url}/lookup/details?descriptor={mesh_id}"
                detail_response = requests.get(detail_url, headers=headers)
                
                if detail_response.status_code == 200:
                    details = detail_response.json()
                    
                    # Check qualifiers to infer category
                    qualifiers = details.get('qualifiers', [])
                    print("Qualifiers:", qualifiers)  # Debugging the qualifiers
                    
                    if any(q.get("label", "").lower() in ["chemistry","protein", "cytokine", "enzyme"] for q in qualifiers):
                        return "Chemical", "MESH", f"MESH:{mesh_id}"
                    elif any(q.get("label", "").lower() in ["pathology", "disease", "disorder", "syndrome", "prevention"] for q in qualifiers):
                        return "Disease", "MESH", f"MESH:{mesh_id}"
                    elif any(q.get("label", "").lower() in ["immunology", "epidemiology", "physiology","pathway", "process", "signaling"] for q in qualifiers):
                        return "Biological_Process", "MESH", f"MESH:{mesh_id}"
                    else:
                        return "Chemical", "MESH", f"MESH:{mesh_id}"
    
    except Exception as e:
        print(f"Error querying NIH MESH API: {str(e)}")
    
    return None, None, None

def query_protein_ontology_fuzzy(entity, threshold=0.7):
    """
    Query Protein Ontology (PRO) for protein terms.
    Returns (label_type, namespace, pro_id) tuple.
    """
    url = f"https://www.ebi.ac.uk/ols/api/ontologies/pr/terms?q={entity.lower()}"
    response = requests.get(url)
    
    if response.status_code == 200:
        results = response.json()
        terms = results.get("_embedded", {}).get("terms", [])
        
        if terms:
            for term in terms:
                label = term.get("label", "").lower()
                synonyms = term.get("synonyms", [])
                
                # Check label and synonyms for matches
                if (fuzz.ratio(entity.lower(), label) >= threshold * 100 or 
                    any(fuzz.ratio(entity.lower(), syn.lower()) >= threshold * 100 for syn in synonyms)):
                    ontology_id = term.get("obo_id") or term.get("short_form")
                    return "Protein", "PR", ontology_id
    
    # Try UniProt as fallback
    url = f"https://rest.uniprot.org/uniprotkb/search?query={entity.lower()}&format=json"
    response = requests.get(url)
    
    if response.status_code == 200:
        results = response.json()
        if results.get("results"):
            uniprot_id = results["results"][0]["primaryAccession"]
            return "Protein", "UniProt", uniprot_id
            
    return None, None, None

def query_ols_fuzzy(entity, threshold=0.5):
    """
    Enhanced OLS query function with better handling of compound terms and reduced threshold.
    Returns (label_type, namespace, ontology_id) tuple.
    """
    # Clean the search term
    search_term = entity.lower().strip()
    
    # First try Disease Ontology
    label, namespace, ontology_id = query_disease_ontology_fuzzy(entity, threshold)
    if label:
        return label, namespace, ontology_id

    # Then try Gene Ontology
    label, namespace, ontology_id = query_gene_ontology_fuzzy(entity, threshold)
    if label:
        return label, namespace, ontology_id

    # Try direct OLS query
    url = f"https://www.ebi.ac.uk/ols/api/search?q={search_term}&local=true"
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            results = response.json()
            docs = results.get("response", {}).get("docs", [])
            
            if docs:
                # Sort by score and exact matches first
                scored_docs = []
                for doc in docs:
                    label = doc.get("label", "").lower()
                    ontology = doc.get("ontology_prefix", "")
                    
                    # Calculate match score
                    exact_match = search_term == label
                    contains_match = search_term in label or label in search_term
                    fuzzy_score = fuzz.ratio(search_term, label)
                    
                    # Boost score for relevant ontologies
                    ontology_boost = 1.2 if ontology in {"GO", "MESH", "DOID", "PR", "CHEBI"} else 1.0
                    
                    final_score = (fuzzy_score * ontology_boost) + (50 if exact_match else 0) + (25 if contains_match else 0)
                    
                    scored_docs.append((final_score, doc))
                
                # Sort by score
                scored_docs.sort(reverse=True, key=lambda x: x[0])
                
                # Process the best match
                if scored_docs:
                    best_score, best_doc = scored_docs[0]
                    
                    if best_score >= threshold * 100:
                        ontology = best_doc.get("ontology_prefix", "")
                        ontology_id = best_doc.get("obo_id") or best_doc.get("short_form")
                        
                        # Map ontology to label type
                        if ontology in {"DOID", "MESH", "MESHD"}:
                            # Check if it's inflammation-related
                            if "inflammation" in search_term or "itis" in search_term:
                                return "Biological_Process", ontology, ontology_id
                            return "Disease", ontology, ontology_id
                        elif ontology == "GO":
                            return "Biological_Process", ontology, ontology_id
                        elif ontology == "PR":
                            return "Protein", ontology, ontology_id
                        elif ontology == "CHEBI":
                            return "Chemical", ontology, ontology_id
                        
                        # Additional checks for biological processes
                        label = best_doc.get("label", "").lower()
                        if any(term in label for term in ["process", "regulation", "pathway", "inflammation"]):
                            return "Biological_Process", ontology, ontology_id
    
    except Exception as e:
        print(f"Error querying OLS: {str(e)}")
    
    # Process-specific fallback for inflammation terms
    if "inflammation" in search_term or "itis" in search_term:
        return "Biological_Process", "GO", None
    
    return None, None, None

def determine_entity_label(entity):
    """
    Updated function to include phenotype detection.
    Returns (label_type, namespace, ontology_id) tuple.
    """
    """
    Updated function to prioritize phenotype detection properly.
    Returns (label_type, namespace, ontology_id) tuple.
    """

    # First check for COVID-specific cases
    if any(term in entity.upper() for term in ["COVID", "SARS-COV", "SEQUELA OF COVID"]):
        # Try Disease Ontology first
        label, namespace, ontology_id = query_disease_ontology_fuzzy(entity)
        if label:
            return sanitize_label(label), sanitize_label(namespace), ontology_id
        
        # Try MESH as fallback for COVID terms
        label, namespace, ontology_id = query_mesh_nih(entity)
        if label:
            return "Disease", "MESH", ontology_id
        
        # If no specific ID found, still classify as Disease
        return "Disease", "MESH", None

    # First check for obvious entity types
    is_likely_gene, suggested_label = determine_entity_type(entity)
    
    # Prioritize phenotype detection first
    if suggested_label == "Phenotype":
        # Try HPO first
        label, namespace, ontology_id = query_hpo_fuzzy(entity)
        if label:
            return sanitize_label(label), sanitize_label(namespace), ontology_id
            
        # Try MP as fallback
        label, namespace, ontology_id = query_mammalian_phenotype_fuzzy(entity)
        if label:
            return sanitize_label(label), sanitize_label(namespace), ontology_id
            
        # If we're confident it's a phenotype but couldn't get an ID, return this
        return "Phenotype", "HP", None
    
    # Only check MESH if we're not confident it's a phenotype
    label, namespace, ontology_id = query_mesh_nih(entity)
    if label and namespace and ontology_id:
        # Double check if MESH term is actually a phenotype
        if any(term in entity.lower() for term in [
            "inheritance", "loss", "megaly", "trophy", "plasia", "grade", 
            "disability", "impairment", "deficit", "dysfunction", "seizure", 
            "ataxia", "palsy", "dystrophy", "weakness", "retardation"
        ]):
            # Try HPO again for these terms
            pheno_label, pheno_ns, pheno_id = query_hpo_fuzzy(entity)
            if pheno_id:
                return pheno_label, pheno_ns, pheno_id
            return "Phenotype", "HP", None
        return sanitize_label(label), sanitize_label(namespace), ontology_id

    # Rest of the function remains the same...
    if is_likely_gene:
        label, namespace, ontology_id = query_hgnc_fuzzy(entity)
        if label:
            return sanitize_label(label), sanitize_label(namespace), ontology_id
       
    # Try NIH MESH
    label, namespace, ontology_id = query_mesh_nih(entity)
    if label and namespace and ontology_id:
        # Check if MESH term is actually a phenotype
        if any(term in entity.lower() for term in [
            "loss", "megaly", "trophy", "plasia", "grade", "disability",
            "impairment", "deficit", "dysfunction", "seizure", "ataxia",
            "palsy", "dystrophy", "weakness", "retardation"
        ]):
            return "Phenotype", "MESH", ontology_id
        return sanitize_label(label), sanitize_label(namespace), ontology_id

    # Try NIH MESH first
    label, namespace, ontology_id = query_mesh_nih(entity)
    if label and namespace and ontology_id:
        return sanitize_label(label), sanitize_label(namespace), ontology_id

    # First check for obvious entity types
    is_likely_gene, suggested_label = determine_entity_type(entity)
    
    # Check for phenotypes
    if suggested_label == "Phenotype":
        # Try HPO first for phenotype terms
        label, namespace, ontology_id = query_hpo_fuzzy(entity)
        if label:
            return sanitize_label(label), sanitize_label(namespace), ontology_id
    
    if suggested_label == "Protein":
        label, namespace, ontology_id = query_protein_ontology_fuzzy(entity)
        if label:
            return sanitize_label(label), sanitize_label(namespace), ontology_id
            
        return "Protein", "PR", None
       
    if is_likely_gene:
        label, namespace, ontology_id = query_hgnc_fuzzy(entity)
        if label:
            return sanitize_label(label), sanitize_label(namespace), ontology_id

    # Try specific ontologies based on suggested label
    if suggested_label:
        if suggested_label == "Disease":
            label, namespace, ontology_id = query_disease_ontology_fuzzy(entity)
            if label:
                return sanitize_label(label), sanitize_label(namespace), ontology_id
        elif suggested_label == "Biological_Process":
            label, namespace, ontology_id = query_gene_ontology_fuzzy(entity)
            if label:
                return sanitize_label(label), sanitize_label(namespace), ontology_id
    
    # Try OLS with improved handling
    label, namespace, ontology_id = query_ols_fuzzy(entity, threshold=0.4)
    if label:
        return sanitize_label(label), sanitize_label(namespace), ontology_id

    # Text analysis fallback
    text_indicators = {
        r".*kine[s]?$": ("Protein", "PR"),
        r".*ase$|.*itis$|.*osis$": ("Disease", "MESH"),
        r".*ation$|.*ing$|.*sis$": ("Biological_Process", "GO"),
        r".*in$|.*or$": ("Protein", "PR"),
        r".*way$": ("Pathway", "PW"),
        r".*cell.*|.*cyte.*": ("Cell", "CL"),
        r".*phenotype$|.*abnormal.*|.*trait.*": ("Phenotype", "HP")  # Added phenotype patterns
    }
    
    for pattern, (label, ontology) in text_indicators.items():
        if re.search(pattern, entity.lower()):
            if label == "Phenotype":
                pheno_label, pheno_ns, pheno_id = query_hpo_fuzzy(entity)
                if pheno_id:
                    return pheno_label, pheno_ns, pheno_id
            elif label == "Protein":
                prot_label, prot_ns, prot_id = query_protein_ontology_fuzzy(entity)
                if prot_id:
                    return prot_label, prot_ns, prot_id
            return label, ontology, None
            
    # Final fallback
    if any(term in entity.lower() for term in ["phenotype", "trait", "clinical", "symptoms"]):
        pheno_label, pheno_ns, pheno_id = query_hpo_fuzzy(entity)
        if pheno_id:
            return pheno_label, pheno_ns, pheno_id
        return "Phenotype", "HP", None
    elif any(term in entity.lower() for term in ["cytokine", "cytokines", "interleukin", "chemokine"]):
        prot_label, prot_ns, prot_id = query_protein_ontology_fuzzy(entity)
        if prot_id:
            return prot_label, prot_ns, prot_id
        return "Protein", "PR", None
    elif "process" in entity.lower() or "activity" in entity.lower():
        return "Biological_Process", "GO", None
    elif any(term in entity.lower() for term in ["disease", "syndrome", "disorder", "itis"]):
        return "Disease", "MESH", None
    else:
        return "Biological_Process", "GO", None

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

        # Determine labels, namespaces, and ontology IDs
        start_label, start_namespace, start_ontology_id = determine_entity_label(start_entity)
        end_label, end_namespace, end_ontology_id = determine_entity_label(end_entity)

        # Validate labels
        if not start_label or not end_label:
            raise ValueError(f"Could not determine label for entities: {start_entity} -> {end_entity}")

        # Sanitize labels
        start_label = sanitize_label(start_label)
        end_label = sanitize_label(end_label)

        # Create nodes query with proper handling of ontology IDs
        create_nodes_query = """
        MERGE (start:%s {name: $start_entity})
        ON CREATE SET 
            start.namespace = $start_namespace,
            start.ontology_id = $start_ontology_id,
            start.created_at = timestamp()
        ON MATCH SET 
            start.last_seen = timestamp(),
            start.ontology_id = CASE 
                WHEN $start_ontology_id <> 'UNKNOWN' THEN $start_ontology_id 
                ELSE start.ontology_id 
            END
            
        MERGE (end:%s {name: $end_entity})
        ON CREATE SET 
            end.namespace = $end_namespace,
            end.ontology_id = $end_ontology_id,
            end.created_at = timestamp()
        ON MATCH SET 
            end.last_seen = timestamp(),
            end.ontology_id = CASE 
                WHEN $end_ontology_id <> 'UNKNOWN' THEN $end_ontology_id 
                ELSE end.ontology_id 
            END
        """ % (start_label, end_label)

        try:
            # Create/update nodes with ontology IDs
            tx.run(
                create_nodes_query,
                start_entity=start_entity,
                end_entity=end_entity,
                start_namespace=start_namespace or "UNKNOWN",
                end_namespace=end_namespace or "UNKNOWN",
                start_ontology_id=start_ontology_id if start_ontology_id else "UNKNOWN",
                end_ontology_id=end_ontology_id if end_ontology_id else "UNKNOWN"
            )

            # Create relationship query (unchanged)
            relationship_query = """
            MATCH (start:%s {name: $start_entity})
            MATCH (end:%s {name: $end_entity})
            MERGE (start)-[rel:%s {pmid: $pmid}]->(end)
            ON CREATE SET 
                rel.evidence = $evidence,
                rel.source = $source,
                rel.created_at = timestamp()
            """ % (start_label, end_label, rel_type)

            tx.run(
                relationship_query,
                start_entity=start_entity,
                end_entity=end_entity,
                pmid=str(pmid),
                evidence=str(evidence),
                source=str(source)
            )
            print(f"Successfully processed: {start_entity} ({start_label}, {start_ontology_id}) -> {end_entity} ({end_label}, {end_ontology_id})")
        except Exception as e:
            print(f"Error processing: {start_entity} -> {end_entity}")
            print(f"Labels: {start_label} -> {end_label}")
            print(f"Error: {str(e)}")
            raise e
def main():
    # Load CSV file with triples
    script_directory = os.path.dirname(os.path.abspath(__file__))
    #file_path = os.path.join(script_directory, "PrimeKG/paths-primekg-hypothesis.xlsx")
    #file_path = os.path.join(script_directory, "hypothesis_pmid_evidences.csv")
    #file_path = os.path.join(script_directory, "all-dbs/cleaned_all_db_association-test.csv")
    file_path = os.path.join(script_directory, "all-dbs/shared-mechanism-export1.csv")
    data = pd.read_csv(file_path, encoding="utf-8")
    #data = pd.read_excel(file_path)

    # Check if required columns exist
    required_columns = ["Start Node", "End Node", "REL_type", "Evidences", "Source", "PMID"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Fill NaN values
    data.fillna("Unknown_Entity", inplace=True)
    
    # Create triples dataframe with exact column names from your data
    triples = data[["Start Node", "End Node", "REL_type", "PMID", "Evidences", "Source"]]
    
    data.fillna("Unknown_Entity", inplace=True)

    # Prepare the data for upload
    triples = data[["Source", "Start Node", "End Node", "REL_type", "PMID", "Evidences"]]

    #Upload data to Neo4jAura DB REMOTELY
    neo4j_uploader = Neo4jUploader(
        "neo4j+s://09f8d4e9.databases.neo4j.io",
        "neo4j",
        "password"
    )

    try:
        neo4j_uploader.upload_triples(triples)
        print("Data upload to Neo4j completed successfully.")
    finally:
        neo4j_uploader.close()

if __name__ == "__main__":
    main()
