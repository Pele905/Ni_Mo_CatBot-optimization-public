import pandas as pd
import os
import json 
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from analysis_scripts import (extract_CV_data_general_protocol, extract_GEIS_data_general_protocol)


def analyze_data_after_testing(datafile_path = None, 
                               CV_stability_cycling_scan_rate = 50, 
                               return_IR = False, 
                               save_folder_path_data_file = "", 
                               return_EIS_raw_data = False
                               ):

    experiment_name = datafile_path.split("\\")[-1].split(".csv")[0]
        
    try:
        df = pd.read_csv(datafile_path)
    except Exception as e:
        print("CSV datafile could not be opened")
        print(e)
        
    
    if save_folder_path_data_file == "":
        save_folder_path_data_file = os.path.dirname(datafile_path)
    
    try:
        for file in os.listdir(save_folder_path_data_file):
            if "temperature" in file:
                temperature_json_path = os.path.join(save_folder_path_data_file, file) # if you want to investigate the temoperature file
            if "dep" in file:
                deposition_csv_path =  os.path.join(save_folder_path_data_file, file)# if you want to investigate the deposition potential file
            if "parameter" in file:
                parameter_dict_path = os.path.join(save_folder_path_data_file, file)# if you want to investigate the parameter files
            
    except:
        pass
    
    # Extract the stability cycling CVs, as well as the ECSA dictionary
    stability_cycling_CVs, ECSA_dict = extract_CV_data_general_protocol(df=df, 
                                                                        experiment_name=experiment_name, 
                                        scan_rates_ECSA_mV_s=[320, 160, 80, 40, 20, 10, 5], 
                                        scan_rates_stability_mV_s=CV_stability_cycling_scan_rate, 
                                        
                                        )
    # Get the IR correction as well as the EIS raw data. 
    IR_correction, EIS_raw_data = extract_GEIS_data_general_protocol(df=df, 
                                                                     use_GEIS_i_for_IR=0,
                                        experiment_name=experiment_name, 
                                        save_plotted_data = False, 
                                        save_path = save_folder_path_data_file)
    
    if return_EIS_raw_data:
        return stability_cycling_CVs, ECSA_dict, IR_correction, EIS_raw_data
    
    if return_IR:
        return stability_cycling_CVs, ECSA_dict, IR_correction
    
    return stability_cycling_CVs, ECSA_dict

    



