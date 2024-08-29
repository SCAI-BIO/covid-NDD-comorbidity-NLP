
# Exploring the Current State of Knowledge on the Link Between COVID-19 and Neurodegeneration
This repository contains the data and source code used in our research titled "Exploring the Current State of Knowledge on the Link Between COVID-19 and Neurodegeneration using Natural language Processing and Knowledge Graphs".

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Models](#models)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Structure

```plaintext
covid-NDD-comorbidity-NLP/
│
├── README.md                        # Project overview and instructions
├── LICENSE                          # License information
├── requirements.txt                 # Python dependencies
├── src/                             # Core source code for the project
│   ├── data_processing/             # Scripts for data extraction and preprocessing
│   ├── model/                       # Model training and inference scripts
│   ├── utils/                       # Utility functions and helper scripts
│   └── main.py                      # Main script to run the project
│
├── data/                            # Data directory
│   ├── raw/                         # Raw data (images and publication figures)
│   ├── processed/                   # Processed data ready for modeling
│   └── samples/                     # Sample images for testing
│
├── output/                          # Output directory
│   ├── results/                     # Extracted entities and triples
│   └── logs/                        # Logs from model runs and other processes
│
├── notebooks/                       # Jupyter notebooks for experiments and analysis
│   ├── exploration.ipynb            # Data exploration and initial analysis
│   ├── model_training.ipynb         # Model training and evaluation
│   └── results_analysis.ipynb       # Analysis of model results
│
├── scripts/                         # Standalone scripts for specific tasks
│   ├── download_data.py             # Script to download data from sources
│   └── preprocess_images.py         # Script to preprocess images
│
└── tests/                           # Unit and integration tests
    ├── test_data_processing.py      # Tests for data processing functions
    └── test_model.py  


