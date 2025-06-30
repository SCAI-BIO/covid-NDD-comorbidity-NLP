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