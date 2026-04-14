# Ni_Mo_CatBot Optimization (Public)

This repository contains scripts and notebooks for investigating Ni–Mo catalyst optimization runs.

## Overview

The workflow consists of two main steps:
1. Extract and structure raw experimental data
2. Analyze and visualize the results

## Data Extraction

Before performing any analysis, the raw chemical data must be extracted and stored in a structured format.

Run the following script:

    python extract_and_store_raw_data.py

**Note:**  
Make sure to update the data directory in the script so it matches the location of the data on your machine.

## Output

Running the extraction script generates `.json` files for each optimization run, including:

- EIS data  
- Stability cycling data  
- ECSA data  

This is performed for all optimization strategies:
- UCB (β = 1)  
- UCB (β = 5)  
- Hybrid approach  

All output files are saved to a user-defined directory.

## Data Analysis

To explore and analyze the optimization results, open:

- `Ni_Mo_data_analysis.ipynb`

This notebook provides an overview of the optimization runs and their outcomes.

For more detailed analysis, use:

- `detailed_data_analysis.ipynb`

This notebook serves as a starting point for deeper investigation of the dataset, and uses the generated data .json data created from "extract_and_store_raw_data.py".