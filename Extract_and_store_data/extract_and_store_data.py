import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
import time
import gc
import os
from pathlib import Path
###############################################################################
### WARNING:
### Due to a software update on 26 January, this script is only compatible
### with data generated prior to this date.
###############################################################################


# List where the datasets are 
#data_first_set = "/Volumes/Raw_lab_data/B310/323-Electrochemistry/3413-Catbot-3425/EC_data_CatBot/Ni_Mo_optimization_run_01_05_25"
#data_beta_5 = "/Volumes/Raw_lab_data/B310/323-Electrochemistry/3413-Catbot-3425/EC_data_CatBot/Ni_Mo_Optimization_Jonas/Beta_5"
data_beta_1 = "/Volumes/Raw_lab_data/B310/323-Electrochemistry/3413-Catbot-3425/EC_data_CatBot/Ni_Mo_Optimization_Jonas/Beta_1"

root_save_path_json = "/Users/pvifr/Desktop/ElectrochemicalDataAnalysis/Ni_Mo_CatBot optimization public/Datasets"

keywords = ["Ni_Mo_beta_1"] #["Ni_Mo_beta_5", "Ni_Mo_beta_1", "Ni_Mo_og"]

folders = [
    #data_first_set,
    data_beta_1, 
    #data_beta_5
]

for folder, keyword in zip(folders, keywords):
    
    keywords = ["Ni_Mo_beta_1"] #["Ni_Mo_og", "Ni_Mo_beta_1", "Ni_Mo_beta_5"]
    ECSA_json = os.path.join(root_save_path_json, f"ECSA_complete_{keyword}.json")
    EIS_json = os.path.join(root_save_path_json, f"EIS_complete_{keyword}.json")
    Stability_json = os.path.join(root_save_path_json, f"Stability_complete_{keyword}.json")

    for subfolder in os.listdir(folder):
        if "exp" in subfolder:
            
            init = time.time()
            extract_all_data_from_experiment(I_stabilities=[100, 10, 1], 
                                            folderpath=os.path.join(folder, subfolder), 
                                            ECSA_json_path=ECSA_json ,
                                            Stability_json_path=Stability_json, 
                                            EIS_json_path=EIS_json, 
                                            use_idxs_for_ECSA=False)

            total_processing_time = time.time() - init
            print("Processing time:", total_processing_time)

            gc.collect()
            
