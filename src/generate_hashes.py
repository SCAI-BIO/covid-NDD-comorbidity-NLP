#!/usr/bin/env python3
"""
SHA256 Hash Generation and Verification Script for COVID-NDD Knowledge Graphs
Generates reproducible hashes for all knowledge graph sources used in the analysis.
"""

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import logging
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class KnowledgeGraphInfo:
    """Data class for knowledge graph metadata."""
    name: str
    version: str
    file_path: str
    file_size_bytes: int
    sha256_hash: str
    download_date: str
    download_url: Optional[str] = None
    description: Optional[str] = None
    license: Optional[str] = None
    preprocessing_applied: Optional[List[str]] = None

class HashGenerator:
    """Generate and verify SHA256 hashes for knowledge graph files."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.hash_dir = Path("data/hashes")
        self.hash_dir.mkdir(parents=True, exist_ok=True)
        
    def calculate_file_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Calculate SHA256 hash for a file."""
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            raise
    
    def get_file_size(self, file_path: Path) -> int:
        """Get file size in bytes."""
        return file_path.stat().st_size
    
    def generate_kg_info(self, name: str, version: str, file_path: str, 
                        download_url: str = None, description: str = None,
                        license: str = None, preprocessing: List[str] = None) -> KnowledgeGraphInfo:
        """Generate complete metadata for a knowledge graph file."""
        
        full_path = self.data_dir / file_path
        
        if not full_path.exists():
            logger.warning(f"File not found: {full_path}")
            # Return placeholder for missing files
            return KnowledgeGraphInfo(
                name=name,
                version=version,
                file_path=file_path,
                file_size_bytes=0,
                sha256_hash="FILE_NOT_FOUND",
                download_date=datetime.now().isoformat(),
                download_url=download_url,
                description=description,
                license=license,
                preprocessing_applied=preprocessing or []
            )
        
        logger.info(f"Processing {name} v{version}...")
        
        # Calculate hash and size
        file_hash = self.calculate_file_hash(full_path)
        file_size = self.get_file_size(full_path)
        
        return KnowledgeGraphInfo(
            name=name,
            version=version,
            file_path=file_path,
            file_size_bytes=file_size,
            sha256_hash=file_hash,
            download_date=datetime.now().isoformat(),
            download_url=download_url,
            description=description,
            license=license,
            preprocessing_applied=preprocessing or []
        )
    
    def generate_all_hashes(self) -> Dict[str, KnowledgeGraphInfo]:
        """Generate hashes for all known knowledge graph sources."""
        
        # Define all knowledge graphs used in the project
        kg_definitions = [
            {
                "name": "DrugBank",
                "version": "5.1.12",
                "file_path": "drugbank/drugbank_v5.1.12.xml",
                "download_url": "https://go.drugbank.com/releases/5-1-12",
                "description": "Comprehensive drug and drug target database",
                "license": "DrugBank License (academic use)",
                "preprocessing": ["XML parsing", "entity extraction"]
            },
            {
                "name": "OpenTargets",
                "version": "24.06",
                "file_path": "opentargets/opentargets_24.06.json.gz",
                "download_url": "https://platform.opentargets.org/downloads",
                "description": "Target-disease associations",
                "license": "CC0 1.0",
                "preprocessing": ["JSON parsing", "filtering by score"]
            },
            {
                "name": "DisGeNET",
                "version": "24.2",
                "file_path": "disgenet/all_gene_disease_associations_v24.2.tsv",
                "download_url": "https://www.disgenet.org/downloads",
                "description": "Gene-disease associations",
                "license": "CC BY-NC-SA 4.0",
                "preprocessing": ["TSV parsing", "score filtering"]
            },
            {
                "name": "INDRA",
                "version": "1.0",
                "file_path": "indra/indra_statements_v1.0.pkl",
                "download_url": "https://indra.readthedocs.io/",
                "description": "Integrated biological knowledge statements",
                "license": "BSD 2-Clause",
                "preprocessing": ["statement extraction", "evidence filtering"]
            },
            {
                "name": "PrimeKG",
                "version": "2023-release",
                "file_path": "primekg/kg.csv",
                "download_url": "https://github.com/mims-harvard/PrimeKG",
                "description": "Multimodal biomedical knowledge graph",
                "license": "MIT",
                "preprocessing": ["CSV parsing", "relation filtering"]
            },
            {
                "name": "COVID-19 Literature",
                "version": "2024-07",
                "file_path": "covid_literature/covid_papers_2024_07.json",
                "description": "COVID-19 related publications",
                "license": "Various (depends on publisher)",
                "preprocessing": ["NLP processing", "entity extraction", "relation mining"]
            },
            {
                "name": "NDD Literature",
                "version": "2024-07",
                "file_path": "ndd_literature/ndd_papers_2024_07.json",
                "description": "Neurodegenerative disease publications",
                "license": "Various (depends on publisher)",
                "preprocessing": ["NLP processing", "entity extraction", "relation mining"]
            }
        ]
        
        results = {}
        
        for kg_def in kg_definitions:
            kg_info = self.generate_kg_info(**kg_def)
            results[kg_info.name] = kg_info
            
        return results
    
    def save_hash_metadata(self, kg_info_dict: Dict[str, KnowledgeGraphInfo], 
                          output_file: str = "kg_versions_and_hashes.json"):
        """Save hash metadata to JSON file."""
        
        output_path = self.hash_dir / output_file
        
        # Convert to serializable format
        serializable_data = {
            "generated_date": datetime.now().isoformat(),
            "generator_version": "1.0.0",
            "knowledge_graphs": {
                name: asdict(info) for name, info in kg_info_dict.items()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Hash metadata saved to {output_path}")
        
        # Also create a simplified version for the README
        self.create_readme_table(kg_info_dict)
    
    def create_readme_table(self, kg_info_dict: Dict[str, KnowledgeGraphInfo]):
        """Create markdown table for README."""
        
        readme_path = self.hash_dir / "README_table.md"
        
        with open(readme_path, 'w') as f:
            f.write("| Source | Version | SHA256 Hash | File Size (MB) | Status |\n")
            f.write("|--------|---------|-------------|----------------|--------|\n")
            
            for name, info in kg_info_dict.items():
                status = "✅ Available" if info.sha256_hash != "FILE_NOT_FOUND" else "❌ Missing"
                size_mb = round(info.file_size_bytes / (1024*1024), 2) if info.file_size_bytes > 0 else "N/A"
                hash_display = info.sha256_hash[:16] + "..." if len(info.sha256_hash) > 16 else info.sha256_hash
                
                f.write(f"| {name} | {info.version} | `{hash_display}` | {size_mb} | {status} |\n")
        
        logger.info(f"README table saved to {readme_path}")
    
    def verify_hashes(self, metadata_file: str = "kg_versions_and_hashes.json") -> bool:
        """Verify existing hashes against current files."""
        
        metadata_path = self.hash_dir / metadata_file
        
        if not metadata_path.exists():
            logger.error(f"Metadata file not found: {metadata_path}")
            return False
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        all_valid = True
        
        for name, info in data["knowledge_graphs"].items():
            file_path = self.data_dir / info["file_path"]
            
            if not file_path.exists():
                logger.warning(f"File missing: {file_path}")
                all_valid = False
                continue
            
            current_hash = self.calculate_file_hash(file_path)
            stored_hash = info["sha256_hash"]
            
            if current_hash == stored_hash:
                logger.info(f"✅ {name}: Hash verified")
            else:
                logger.error(f"❌ {name}: Hash mismatch!")
                logger.error(f"   Expected: {stored_hash}")
                logger.error(f"   Current:  {current_hash}")
                all_valid = False
        
        return all_valid

def main():
    """Main function with CLI interface."""
    
    parser = argparse.ArgumentParser(description="Generate and verify SHA256 hashes for knowledge graphs")
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing knowledge graph files")
    parser.add_argument("--verify", action="store_true", help="Verify existing hashes instead of generating new ones")
    parser.add_argument("--output", default="kg_versions_and_hashes.json", help="Output filename for hash metadata")
    
    args = parser.parse_args()
    
    generator = HashGenerator(args.data_dir)
    
    if args.verify:
        logger.info("Verifying existing hashes...")
        if generator.verify_hashes(args.output):
            logger.info("✅ All hashes verified successfully")
            sys.exit(0)
        else:
            logger.error("❌ Hash verification failed")
            sys.exit(1)
    else:
        logger.info("Generating new hashes...")
        kg_info = generator.generate_all_hashes()
        generator.save_hash_metadata(kg_info, args.output)
        
        # Print summary
        total_files = len(kg_info)
        available_files = sum(1 for info in kg_info.values() if info.sha256_hash != "FILE_NOT_FOUND")
        
        logger.info(f"Summary: {available_files}/{total_files} files processed successfully")
        
        if available_files < total_files:
            logger.warning("Some files are missing. Please download them to complete the verification.")

if __name__ == "__main__":
    main()