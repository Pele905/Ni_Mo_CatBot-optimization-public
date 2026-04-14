# Ni_Mo_CatBot-optimization-public

Script to investigate the NiMo optimization run performed. In order to analyze the data from the optimization run go to Ni_Mo_data_analysis.ipynb

To extract and store the raw chemical data run the extract_and_store_raw_data.py. Remember to change the folder where the data is located to match your PC.
By running this script you get one json file for EIS data, one for stability cycling and one for ECSA. This will happen for all optimization runs (UCB beta = 1, UCB beta = 5 and the hybrid approach)

The data can then be loaded and more in-depth investigation into the NiMo data can be performed. This is done in the script detailed_data_analysis.ipynb as an example