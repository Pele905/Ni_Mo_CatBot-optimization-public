import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
import time
import gc
import os
from pathlib import Path


# Folders where you store the UCB beta = 1, UCB beta = 5, and Strategy 3 (Hybridg strategies)
# List where the datasets are 
root_path = "/Volumes/ENRGK/Raw_lab_data/B313/307-Automated Synthesis/3413-Catbot-3425/EC_data_CatBot"
data_hybrid = root_path + "/Ni_Mo_optimization_run_01_05_25"
data_beta_5 = root_path + "/Ni_Mo_Optimization_Jonas/Beta_5"
data_beta_1 = root_path + "/Ni_Mo_Optimization_Jonas/Beta_1"

root_path_random = "/Volumes/Elements/Random experiments baseline"
data_random_1 = "/Volumes/Elements/Random experiments baseline/seed_42"
data_random_2 = "/Volumes/Elements/Random experiments baseline/seed_1"
root_save_path_json = "/Users/pvifr/Desktop/ElectrochemicalDataAnalysis/Ni_Mo_CatBot optimization public/Datasets"

keywords = ["Ni_Mo_hybrid", "Ni_Mo_beta_1", "Ni_Mo_beta_5"]

folders = [
    data_hybrid,
    data_beta_1, 
    data_beta_5
]

if 1 == 2:
    for folder, keyword in zip(folders, keywords):
        
        ECSA_json = os.path.join(root_save_path_json, f"ECSA_complete_{keyword}.json")
        EIS_json = os.path.join(root_save_path_json, f"EIS_complete_{keyword}.json")
        Stability_json = os.path.join(root_save_path_json, f"Stability_complete_{keyword}.json")

        for subfolder in os.listdir(folder):
            if "exp" in subfolder:
                
                init = time.time()
                extract_all_data_from_experiment(I_stabilities=[100, 50, 20, 15, 10, 5, 2, 1, 0.5], 
                                                folderpath=os.path.join(folder, subfolder), 
                                                ECSA_json_path=ECSA_json ,
                                                Stability_json_path=Stability_json, 
                                                EIS_json_path=EIS_json, 
                                                use_idxs_for_ECSA=False)

                total_processing_time = time.time() - init
                print("Processing time:", total_processing_time)

                gc.collect()
                

# Random experimentation
for folder, keyword in zip([data_random_1, data_random_2], ["random_seed_42", "random_seed_1"]):

    ECSA_json = os.path.join(root_save_path_json, f"ECSA_complete_{keyword}.json")
    EIS_json = os.path.join(root_save_path_json, f"EIS_complete_{keyword}.json")
    Stability_json = os.path.join(root_save_path_json, f"Stability_complete_{keyword}.json")

    for subfolder in os.listdir(folder):
        if "exp" in subfolder:
            
            init = time.time()
            extract_data_from_Ni_Mo_v2(current_densities_stability=[100, 50, 20, 15, 10, 5, 2, 1, 0.5], 
                                            folderpath=os.path.join(folder, subfolder), 
                                            ECSA_json_path=ECSA_json ,
                                            Stability_json_path=Stability_json, 
                                            EIS_json_path=EIS_json)

            total_processing_time = time.time() - init
            print("Processing time:", total_processing_time)

            gc.collect()
