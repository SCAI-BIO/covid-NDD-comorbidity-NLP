import os
import logging
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from neo4j import GraphDatabase

@dataclass
class Neo4jConfig:
    """Configuration for Neo4j connection."""
    uri: str = "neo4j+s://09f8d4e9.databases.neo4j.io"
    user: str = "neo4j"
    password: str = "your_password"  # Replace with actual password

class BiomedicalDataPipeline:
    """Main pipeline class for processing and uploading biomedical data."""
    
    def __init__(self, config: Neo4jConfig):
        """Initialize the pipeline with Neo4j configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging format and level."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('pipeline.log')
            ]
        )

    def process_hypothesis_data(self, file_path: str) -> bool:
        """Process and upload hypothesis data."""
        self.logger.info("Starting hypothesis data processing")
        try:
            data = pd.read_csv(file_path, encoding="latin1")
            data.fillna("Unknown_Entity", inplace=True)
            triples = data[["Source", "Start Node", "End Node", "REL_type", 
                          "PMID", "Evidences"]]
            
            with Neo4jUploader(self.config.uri, self.config.user, 
                             self.config.password) as uploader:
                uploader.upload_triples(triples)
            
            self.logger.info("Hypothesis data processing completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing hypothesis data: {e}")
            return False

    def process_pathway_data(self, file_path: str) -> bool:
        """Process and upload pathway data."""
        self.logger.info("Starting pathway data processing")
        try:
            data = pd.read_csv(file_path)
            data.fillna("Unknown", inplace=True)
            triples = data[['Subject', 'Subject_Namespace', 'Object', 
                          'Object_Namespace', 'Relation', 'pmid', 
                          'evidence', 'Source']]
            
            with Neo4jUploader(self.config.uri, self.config.user, 
                             self.config.password) as uploader:
                uploader.upload_triples(triples)
            
            self.logger.info("Pathway data processing completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing pathway data: {e}")
            return False

    def process_gwas_data(self, file_path: str) -> bool:
        """Process and upload GWAS data."""
        self.logger.info("Starting GWAS data processing")
        try:
            enricher = Neo4jGWASEnricher(
                file_path,
                uri=self.config.uri,
                user=self.config.user,
                password=self.config.password
            )
            enricher.enrich_graph()
            enricher.close()
            
            self.logger.info("GWAS data processing completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing GWAS data: {e}")
            return False

class DataPipelineRunner:
    """Class to manage the execution of the data pipeline."""
    
    def __init__(self, config: Optional[Neo4jConfig] = None):
        self.config = config or Neo4jConfig()
        self.pipeline = BiomedicalDataPipeline(self.config)
        
    def run(self, 
            triple_file: str = "hypothesis_pmid_evidences.csv",
            pathway_file: str = "all-dbs/cleaned_all_db_association.csv",
            gwas_file: str = "GWAS/shared-variants.xlsx") -> None:
        """
        Run the complete pipeline.
        
        Args:
            triple_file: Path to hypothesis data file
            pathway_file: Path to pathway data file
            gwas_file: Path to GWAS data file
        """
        # Dictionary to track processing status
        status = {
            "hypothesis": False,
            "pathway": False,
            "gwas": False
        }
        
        # Process hypothesis data
        if os.path.exists(triple_file):
            status["hypothesis"] = self.pipeline.process_hypothesis_data(
                triple_file
            )
        else:
            logging.warning(f"Hypothesis file not found: {triple_file}")
            
        # Process pathway data
        if os.path.exists(pathway_file):
            status["pathway"] = self.pipeline.process_pathway_data(
                pathway_file
            )
        else:
            logging.warning(f"Pathway file not found: {pathway_file}")
            
        # Process GWAS data
        if os.path.exists(gwas_file):
            status["gwas"] = self.pipeline.process_gwas_data(
                gwas_file
            )
        else:
            logging.warning(f"GWAS file not found: {gwas_file}")
            
        # Report final status
        successful = sum(status.values())
        total = len(status)
        logging.info(f"Pipeline completed: {successful}/{total} processes successful")
        
        # Log details for failed processes
        failed = [name for name, success in status.items() if not success]
        if failed:
            logging.warning(f"Failed processes: {', '.join(failed)}")

def main():
    """Main entry point for the pipeline."""
    # Configure Neo4j connection
    config = Neo4jConfig(
        uri="neo4j+s://09f8d4e9.databases.neo4j.io",
        user="neo4j",
        password="your_password"  # Replace with actual password
    )
    
    # Initialize and run pipeline
    runner = DataPipelineRunner(config)
    
    # Run with default file paths
    runner.run()
    
    # Alternatively, specify custom file paths:
    # runner.run(
    #     triple_file="custom/path/hypothesis.csv",
    #     pathway_file="custom/path/pathway.csv",
    #     gwas_file="custom/path/gwas.xlsx"
    # )

if __name__ == "__main__":
    main()
