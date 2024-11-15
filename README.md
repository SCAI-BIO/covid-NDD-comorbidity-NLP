# Exploring the Current State of Knowledge on the Link Between COVID-19 and Neurodegeneration

This repository contains the data, scripts, and analyses used in the research titled **"Understanding the Co-Morbidity between COVID-19 and Neurodegenerative Diseases at Mechanism-Level: Comprehensive Analysis Integrating Databases and Text Mining"**. The project leverages Neo4j AuraDB for graph-based analysis and integrates natural language processing to explore relationships between COVID-19 and neurodegenerative diseases.

## Table of Contents
- [Overview](#overview)
- [Data](#data)
- [Sources](#src)
- [Notebooks](#notebooks)
- [Getting Started](#getting-started)
- [Contact](#contact)

## Overview
This project explores the connections between COVID-19 and neurodegenerative diseases by:
1. **Extracting data** from scientific literature and harmonizing it into a structured format.
2. **Building a knowledge graph** in Neo4j to identify and analyze relationships between entities such as genes, diseases, and chemicals.
3. **Running community detection and graph algorithms** to uncover clusters and patterns in the data.

## data
The repository includes the following directories:

1. **Expert-curated-publications**: Contains manually curated publications relevant to the study, ensuring high-quality and accurate information.

2. **PubTator3-results**: Includes results from PubTator3, a web-based system that offers a comprehensive set of features and tools for exploring biomedical literature using advanced text mining and AI techniques. :contentReference[oaicite:0]{index=0}

3. **Sherpa-results**: Houses outputs from Sherpa, a tool designed to assist in the curation of biomedical literature by providing automated annotations and insights.

4. **Textual-corpora-for-textmining**: Comprises textual corpora prepared for text mining purposes, facilitating the extraction of meaningful patterns and relationships.

## src

### 1. `comorbidity-hypothesis-db.py`
- **Purpose**: Automatically opens the Neo4j Browser with prefilled credentials to connect to the AuraDB instance for data exploration.
- **Key Features**:
  - Simplifies connection to Neo4j by generating a pre-configured URL.
  - Useful for direct interaction with the knowledge graph.
- **Usage**:
  Run the script, and the Neo4j Browser will open in your default web browser:
  ```bash
  python comorbidity-hypothesis-db.py

## notebooks

### 2. `analyze-neo4j.ipynb`
- **Purpose**: Analyzes the knowledge graph in Neo4j to extract insights.
- **Key Features**:
  - Counts nodes and edges in the graph.
  - Executes community detection algorithms like Louvain using Neo4j's Graph Data Science (GDS) library.
  - Retrieves and visualizes properties of detected clusters
- **Usage**:
  Open the Jupyter Notebook and follow the instructions to:
  - Query the Neo4j database.
  - Perform community detection and analyze relationships. 
## Getting Started

### Prerequisites

- **Neo4j AuraDB**: Ensure you have access to a Neo4j AuraDB instance. Use the provided connection details or set up your own.

- **Python Environment**: Install the required libraries:

  ```bash
  pip install neo4j pandas
## Connecting to Neo4j

To interact with the Neo4j database, you have two options:

1. **Using the `comorbidity-hypothesis-db.py` Script**:

   This script automatically opens the Neo4j Browser with prefilled connection details.

   Run the script using the following command:

   ```bash
   python comorbidity-hypothesis-db.py
## Using the Jupyter Notebook

To programmatically analyze the data, open and run the `analyze-neo4j.ipynb` Jupyter Notebook.

## Exploring the Comorbidity Database 

To manually explore the comorbidity graph database:

1. **Open the Neo4j Browser**:

   Navigate to [https://browser.neo4j.io](https://browser.neo4j.io).

2. **Enter the Connection Details**:

   - **URI**: `neo4j+s://1af6a087.databases.neo4j.io`

   - **Username**: `neo4j`

   - **Password**: Refer to the credentials provided in the src/comorbidity-hypothesis-db.py or your configuration.

3. **Run Cypher Queries**:

   Once connected, you can execute Cypher queries to explore the graph. For example, to retrieve a sample of nodes:

   ```cypher
   MATCH (n) RETURN n LIMIT 10;
## Contact

For any questions, suggestions, or collaborations, please contact:

**Negin Babaiha**  
Email: [negin.babaiha@scai.fraunhofer.de](mailto:negin.babaiha@scai.fraunhofer.de)  
[Google Scholar Profile](https://scholar.google.com/citations?user=OwT3AMQAAAAJ)

Feel free to reach out for discussions regarding the project!
