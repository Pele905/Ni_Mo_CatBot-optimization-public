import matplotlib.pyplot as plt 
import numpy as np
from analysis_scripts import (get_forwards_backwards_CV_scan, get_stability_data_from_stability_cycling, )
#from extract_electrochemical_data_from_datasets import analyze_data_after_testing

import os
import numpy as np
from datetime import datetime
import json 
import pandas as pd 
import sys 
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from analysis_scripts import (extract_CV_data_general_protocol, extract_GEIS_data_general_protocol)

# Here in this example, we extract all the data, and then store it in the proper data folder 

def save_json_safely(json_path, 
                     data):
    
    existing_data = {}
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                existing_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load existing JSON due to: {e}. Proceeding with new data.")

    # Merge new data into existing (overwriting same scan rates)
    existing_data.update(data)

    # Save updated ECSA JSON
    if json_path:
        try:
            with open(json_path, 'w') as f:
                json.dump(existing_data, f, indent=4)
            #print(f"Stability data saved to {json_path}")
        except Exception as e:
            print(f"Error saving Stability JSON: {e}")



def transform_ECSA_CV_to_cap_current(I_mA, E_WE, CV_midpoint = 25):

    I_mA_decreasing, E_mV_decreasing, I_mA_increasing, E_mV_increasing = get_forwards_backwards_CV_scan(I_mA, E_we_CV_i=E_WE)

    idx_mid_inc = np.argmin(np.abs(np.array(E_mV_increasing) - CV_midpoint))
    idx_mid_dec = np.argmin(np.abs(np.array(E_mV_decreasing) - CV_midpoint))
    
    cap_current = (abs(I_mA_increasing[idx_mid_inc]) + abs(I_mA_decreasing[idx_mid_dec])) / 2

    return cap_current


def transform_ECSA_dict_to_ECSA(ECSA_dict, 
                                ECSA_json_path = "", 
                                plot_ECSA = False, 
                                save_path_ECSA = "", 
                                scan_rate_cutoff = 10, upper_scan_rate = 320):

    exp_name = list(ECSA_dict.keys())[0]
    data = ECSA_dict[exp_name]

    ECSA_cap_current_dicts = {exp_name: {}}
    scan_rates = []
    cap_currents_init = []
    cap_currents_final = []
    if plot_ECSA:
        fig, (ax_init, ax_final) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
        ax_init.set_title("Initial Scans", fontsize=16)
        ax_final.set_title("Final Scans", fontsize=16)

    for scan_rate in data.keys():
        
        # Something seems wrong with the scan rate at 20....
        if scan_rate_cutoff <= scan_rate <= upper_scan_rate:
            
            data_scan_str = list(data[scan_rate])
            mid_idx = int(len(data_scan_str) / 2) - 1
            
            last_scans_initial = data[scan_rate][data_scan_str[mid_idx]]
            last_scans_final = data[scan_rate][data_scan_str[-1]]
            E_WE_init = last_scans_initial['Working Electrode Voltage [mV]']
            E_WE_final = last_scans_final['Working Electrode Voltage [mV]']

            I_mA_init = last_scans_initial['Current [mA]']
            I_mA_final = last_scans_final['Current [mA]']

            I_mA_decreasing_initial, E_mV_decreasing_initial, I_mA_increasing_initial, E_mV_increasing_initial = get_forwards_backwards_CV_scan(I_mA_init, E_we_CV_i=E_WE_init)

            I_mA_decreasing_final, E_mV_decreasing_final, I_mA_increasing_final, E_mV_increasing_final = get_forwards_backwards_CV_scan(I_mA_final, E_we_CV_i=E_WE_final)

            mid_idx_dec_final = int(len(E_mV_decreasing_final) / 2) - 1
            mid_idx_inc_final = int(len(E_mV_increasing_final) / 2) - 1
            mid_idx_dec_initial = int(len(E_mV_decreasing_initial) / 2) - 1
            mid_idx_inc_initial = int(len(E_mV_increasing_initial) / 2) - 1

            mid_val_dec_final = E_mV_decreasing_final[mid_idx_dec_final]
            mid_val_inc_final = E_mV_increasing_final[mid_idx_inc_final]
            mid_val_dec_initial = E_mV_decreasing_initial[mid_idx_dec_initial]
            mid_val_inc_initial = E_mV_increasing_initial[mid_idx_inc_initial]

            idx_mid_dec_final = np.argmin(np.abs(np.array(E_mV_decreasing_final) - mid_val_dec_final))
            idx_mid_inc_final = np.argmin(np.abs(np.array(E_mV_increasing_final) - mid_val_inc_final))
            idx_mid_dec_initial = np.argmin(np.abs(np.array(E_mV_decreasing_initial) - mid_val_dec_initial))
            idx_mid_inc_initial = np.argmin(np.abs(np.array(E_mV_increasing_initial) - mid_val_inc_initial))

            cap_current_init = (I_mA_increasing_initial[idx_mid_inc_initial] - I_mA_decreasing_initial[idx_mid_dec_initial]) / 2
            cap_current_final = (I_mA_increasing_final[idx_mid_inc_final] - I_mA_decreasing_final[idx_mid_dec_final]) / 2

            ECSA_cap_current_dicts[exp_name][scan_rate] = {"Cap current init [mA]" : cap_current_init, 
                                                "Cap current final [mA]" : cap_current_final,
                                                "Scan rate [mV/s]" : scan_rate
                                                }

            # Save values
            scan_rates.append(scan_rate)
            cap_currents_init.append(cap_current_init)
            cap_currents_final.append(cap_current_final)
            try:
                if plot_ECSA:
                    ax_init.scatter(E_WE_init, I_mA_init, label=f"{scan_rate} mV/s {data_scan_str[mid_idx]}")
                    ax_final.scatter(E_WE_final, I_mA_final, label=f"{scan_rate} mV/s, ")
            except:
                pass

    if plot_ECSA:
        ax_init.set_xlabel("WE [mV]", fontsize=20)
        ax_init.set_ylabel("Current [mA]", fontsize=20)
        ax_final.set_xlabel("WE [mV]", fontsize=20)
        ax_final.legend()
        ax_init.legend()
        plt.tight_layout()
        plt.savefig(save_path_ECSA.split(".png")[0] + "_CVs_ECSAs_initial_and_final.png")
        
    scan_rates_np = np.array(scan_rates)
    init_np = np.array(cap_currents_init)
    final_np = np.array(cap_currents_final)

    # Linear fits
    fit_init = np.polyfit(scan_rates_np, init_np, 1)
    fit_final = np.polyfit(scan_rates_np, final_np, 1)

    x_fit = np.linspace(min(scan_rates_np), max(scan_rates_np), 100)
    y_fit_init = np.polyval(fit_init, x_fit)
    y_fit_final = np.polyval(fit_final, x_fit)

    if plot_ECSA:

        # Plotting
        plt.figure(figsize=(12, 10))
        plt.scatter(scan_rates_np, init_np, color='blue', label='Initial capacitive current')
        plt.plot(x_fit, y_fit_init, 'b--', label=f'Fit (init), slope={fit_init[0] * 10 ** (6):.1f} mu F')

        plt.scatter(scan_rates_np, final_np, color='red', label='Final capacitive current')
        plt.plot(x_fit, y_fit_final, 'r--', label=f'Fit (final), slope={fit_final[0] * 10 ** (6):.1f} mu F')

        plt.xlabel('Scan rate [mV/s]', fontsize = 20)
        plt.ylabel('Capacitive current [mA]', fontsize = 20)

        plt.legend()
        plt.grid(True)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.tight_layout()


        if save_path_ECSA != "":
            plt.savefig(save_path_ECSA)

        plt.close()

        # Plotting closeup of small value 
        plt.figure(figsize=(12, 10))
        plt.scatter(scan_rates_np, final_np, color='red', label='Final capacitive current')
        plt.plot(x_fit, y_fit_final, 'r--', label=f'Fit (final), slope={fit_final[0] * 10 ** (6):.1f} mu F')

        plt.xlabel('Scan rate [mV/s]', fontsize = 20)
        plt.ylabel('Capacitive current [mA]', fontsize = 20)

        plt.legend()
        plt.grid(True)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.tight_layout()
        if save_path_ECSA != "":
            plt.savefig(save_path_ECSA.split(".png")[0] + "_closeup_final.png")
        plt.close()

    ECSA_cap_current_dicts[exp_name]["ECSA initial F"] = fit_init[0]
    ECSA_cap_current_dicts[exp_name]["ECSA final F"] = fit_final[0]
    ECSA_cap_current_dicts[exp_name]["ECSA_final / ECSA_initial"] = fit_final[0] / fit_init[0]
    save_json_safely(json_path=ECSA_json_path, 
                     data=ECSA_cap_current_dicts)


def get_idsx_ECSA_before_after(scan_rates = [5, 10, 20, 40, 80, 160, 320], 
                               n_scans = 100):
    ECSA_before = {}
    ECSA_after = {}
    for i, scan_rate in enumerate(scan_rates):
        ECSA_before[scan_rate] = np.arange(2 + 5 * i, 7 + 5 * i)
        
    # Idxs for running EIS, LSV, stability cycling CVs + LSV
    idx_offset = 36 + 1 + n_scans + ECSA_before[scan_rate][-1]
    for i, scan_rate in enumerate(scan_rates):
        ECSA_after[scan_rate] = np.arange(2 + 5 * i + idx_offset, 7 + 5 * i + idx_offset)
    return ECSA_before, ECSA_after

def transform_df_to_ECSA_dict(ECSA_idxs_before, 
                         ECSA_idxs_after, 
                         experiment_name,
                         df):

    sorted_CVs = {}

    for scan_rate in ECSA_idxs_before:
        sorted_CVs[scan_rate] = {}
        for i in ECSA_idxs_before[scan_rate]:
            CV_V_i = df[df["Step number"] == i]['Working Electrode Voltage [V]'].to_numpy()
            CV_I_i = df[df["Step number"] == i]['Current [A]'].to_numpy()
            sorted_CVs[scan_rate][f"Scan {i}"] = {'Current [mA]': list(CV_I_i * 1000), 
                                            'Working Electrode Voltage [mV]': list(1000 * CV_V_i)}
    
    for scan_rate in ECSA_idxs_after:   
        for i in ECSA_idxs_after[scan_rate]:

            CV_V_i = df[df["Step number"] == i]['Working Electrode Voltage [V]'].to_numpy()
            CV_I_i = df[df["Step number"] == i]['Current [A]'].to_numpy()
            sorted_CVs[scan_rate][f"Scan {i}"] = {'Current [mA]': list(CV_I_i * 1000), 
                                            'Working Electrode Voltage [mV]': list(1000 * CV_V_i)}
    return {experiment_name : sorted_CVs}



def load_json_safely(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None



def extract_CVs_GEIS(datafile_path = None, 
                               CV_stability_cycling_scan_rate = 50, 
                               save_folder_path_data_file = "", 
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
                                        scan_rates_ECSA_mV_s=[320, 160, 80, 40, 20, 10, 5], # Scan rate for the CVs to estimate ECSA in mV/s
                                        scan_rates_stability_mV_s=CV_stability_cycling_scan_rate, # Scan rate during stability cycling is 50 mV / s 
                                        
                                        )
    # Get the IR correction as well as the EIS raw data. 
    IR_correction, EIS_raw_data = extract_GEIS_data_general_protocol(df=df, 
                                                                     use_GEIS_i_for_IR=0,
                                        experiment_name=experiment_name, 
                                        save_plotted_data = False, 
                                        save_path = save_folder_path_data_file)
    

    return stability_cycling_CVs, ECSA_dict, IR_correction, EIS_raw_data


def extract_all_data_from_experiment(I_stabilities, 
                                     folderpath = "",
                                     savepath_ECSA = "", 
                                     ECSA_json_path = "", 
                                     Stability_json_path = "", 
                                     EIS_json_path = "", 
                                     plot_ECSA = False,
                                     use_idxs_for_ECSA = False, 
                        ):
    
    
    '''
        Now also works with using resistance, i.e R = U / I where U is overpotential, and I is the current density at that overpotential
        This is for Ni-Cr paper 
    '''
    overpotential_evolution = {}    
    
    # The experiment is not used for training, skip it 
    try:
        for file in os.listdir(folderpath):
            if file.endswith(".csv") and "Testing" in file:
                file_path = os.path.join(folderpath, file)
            if file.endswith(".json") and "parameter" in file:
                
                with open(os.path.join(folderpath, file), 'r') as file:
                    parameter_dict = json.load(file)
                #print("We make it here")
                
        # Here we get the stability cycling CVs, ECSA_cycling_dicts, IR correction, and raw EIS data
        stability_cycling_CVs, ECSA_cycling_dict, IR, EIS_raw_data = extract_CVs_GEIS(
            datafile_path=file_path,
            CV_stability_cycling_scan_rate=50)

        exp_name = list(ECSA_cycling_dict.keys())[0]

        # Get the stability dict 
        Stability_dict = get_stability_data_from_stability_cycling(
            stability_cycling_dict=stability_cycling_CVs,
            current_densities=I_stabilities,
            get_stability_from="forward_scan",
            IR_correction = IR,
            )
        
        exp_name_param_dict = list(parameter_dict.keys())[0]
        param_dict_clean = parameter_dict[exp_name_param_dict]
        "Experiment start time"
        timestamp_str = param_dict_clean["Experiment start time"]
        timestamp = datetime.strptime(timestamp_str, "%d.%m.%Y at %H:%M")
        timestamp_iso = timestamp.isoformat()

        overpotential_evolution[exp_name] = {
        "params":  param_dict_clean,  # All the experimental params 
        "timestamp" : timestamp_iso, 
        "calibration CV peak [mV]" : 0,
        "exp location PC" : str(folderpath), 
        "ML optimization params": {
            
            "Deposition time [s]": param_dict_clean["deposition_time_s"],
            "Deposition current density [mA/cm2]": param_dict_clean["deposition_current_density_cm2"],
            "Temperature_deposition [C]": param_dict_clean["Deposition_T_K"],
            **{
                f"Concentrations {species} [mol/L]": value
                for species, value in param_dict_clean["Concentrations [mol/L]"].items()
            },
        }, 
        "Cycling results" : Stability_dict, 
        "IR correction" : IR
        
        }
        
        # Trying to save tje EIS and stability json files 
        save_json_safely(json_path=Stability_json_path, 
                        data = overpotential_evolution)

        save_json_safely(json_path= EIS_json_path, 
                        data=EIS_raw_data)

        if use_idxs_for_ECSA:

            n_scans = 100 
            df = pd.read_csv(file_path)
            scan_rates = [5, 10, 20, 40, 80, 160, 320]
            if n_scans == 100:
                scan_rates = [2, 5, 10, 20, 40, 80, 160, 320]
                # In this case we also do the CV scans with 2 mV/s
            ECSA_before, ECSA_after = get_idsx_ECSA_before_after(n_scans=n_scans, scan_rates=scan_rates)
            
            ECSA_dict = transform_df_to_ECSA_dict(ECSA_before, 
                        ECSA_after, 
                        exp_name,
                        df)
            
            ECSA_cycling_dict = ECSA_dict
        transform_ECSA_dict_to_ECSA(ECSA_dict=ECSA_cycling_dict, 
                                    save_path_ECSA=savepath_ECSA, 
                                    ECSA_json_path = ECSA_json_path, 
                                    plot_ECSA = plot_ECSA, 
                                    scan_rate_cutoff=1, 
                                    upper_scan_rate = 320)
        
        plt.close("all")
        del df
        del stability_cycling_CVs 
        del ECSA_cycling_dict
        
    except Exception as e:
        print("Error loading experiment", e)

