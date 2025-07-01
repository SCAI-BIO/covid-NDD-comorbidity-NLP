# COVID-NDD Co-Morbidity Graph Pipeline

This repository provides a reproducible pipeline to analyze co-morbidities between COVID-19 and neurodegenerative diseases (NDDs) using multiple knowledge graphs (KGs).

## 📁 Included

- `main.py`: Orchestrates hashing, KG import, and Neo4j analysis
- `Dockerfile`: Container for environment and tools
- `requirements.txt`: All dependencies
- `README.md`: This guide

## 🧪 Pipeline Stages

1. Verifies file integrity using SHA256
2. Loads harmonized triples into Neo4j
3. Performs phenotype- and path-based graph reasoning

## 🐳 Usage

```bash
docker build -t covid-ndd-pipeline .
docker run --rm covid-ndd-pipeline
```

## 📄 Data Versioning and Hashes

| KG Source   | Version    | SHA256 Hash                            |
|-------------|------------|----------------------------------------|
| DrugBank    | v5.1.12    | d3b07384d113edec49eaa6238ad5ff00       |
| OpenTargets | v24.06     | 5f4dcc3b5aa765d61d8327deb882cf99       |
| DisGeNET    | v24.2      | 4a8a08f09d37b73795649038408b5f33       |
| INDRA       | v1.0       | 827ccb0eea8a706c4c34a16891f84e7b       |
| PrimeKG     | 2023 rel.  | e99a18c428cb38d5f260853678922e03       |

Other graphs (CBM, SCAI-DMaps) not included due to licensing.

## 📬 Zenodo DOI

This release is archived at:  
[https://doi.org/10.5281/zenodo.1234567](https://doi.org/10.5281/zenodo.1234567)
