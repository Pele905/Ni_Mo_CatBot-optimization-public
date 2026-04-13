import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import zscore
from scipy import interpolate
import matplotlib.pyplot as plt
import os
import json 
from scipy import stats
from scipy.optimize import minimize

# All the scripts used here are called upon to analyze the specific Ni - Cr data

# Interpolates IR correction from EIS
def get_interpolation_EIS(Re_Z, Im_Z, start_from = 0, use_mask = True, save_plotted_data = False, 
                          
                          save_path = ""):
    '''
        Interpolates the curve for EIS, so that we can calculate the ohmic drop 
    '''


    #compute distance to zero
    distance_to_zero = np.abs(Im_Z)
    #make new mask
    below_re_z_mask = Re_Z < 1.1
    Re_Z_below =Re_Z[below_re_z_mask]
    Im_Z_below = Im_Z[below_re_z_mask]
    distance_below = distance_to_zero[below_re_z_mask]
    #soret by distance to Im_Z =0 and select 10 nearest
    nearest_indices = np.argsort(distance_below)[:10]
    nearest_points = np.column_stack((Re_Z_below[nearest_indices],Im_Z_below[nearest_indices]))

    def Line_distance(params):
        a, b = params #line: y=ax+b
        distances = []
        for point in nearest_points:
            x, y = point
            distance = abs(a*x-y+b)/np.sqrt(a**2+1)
            distances.append(distance)
        
        return sum(distances)
        
    #initial gues for line params
    initial_params = [1,0]
    result = minimize(Line_distance, initial_params, method='Nelder-Mead')
    a_opt, b_opt =result.x

    root3 =-b_opt/a_opt
    if abs(a_opt) < 0.5: #if the gradient of the intersecting line is smaller then 0.5
        root3 = np.mean(nearest_points,axis=0)[0]
    
    print(root3, "This is root 3")
    
    if save_plotted_data:
        # Plot the original data
        plt.figure(figsize=(12, 9))
        plt.scatter(Re_Z, Im_Z, label="Original Data", color="blue")
        # Mark the root on the plot
        #plt.scatter(root1, 0, label=f"Root at {root1:.2f} Mask 1", color="red", zorder=5)

        #plt.scatter(root2, 0, label=f"Root at {root2:.2f} Mask 2", color="green", zorder=5)
        # Customize the plot

        plt.scatter(root3,0, label=f"Root at {root3:.2f} Mask 3", color="black", zorder=5)

        plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
        plt.axvline(root3, color="purple", linewidth=0.8, linestyle="--")
        plt.legend(fontsize=20, loc = "upper right")
        plt.xlabel("Re(Z)", fontsize=18)
        plt.ylabel("Im(Z)", fontsize=18)
        plt.title("Filtered Data and Interpolation", fontsize=16)
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "EIS_interpolation.png"))
        plt.close()

        # Plot the original data
        plt.figure(figsize=(12, 9))

        plt.scatter(Re_Z, Im_Z, label="Original Data", color="blue")
        # Mark the root on the plot
        #plt.scatter(root1, 0, label=f"Root at {root1:.2f} Mask 1", color="red", zorder=5)
        # Add root 3 
        #plt.scatter(root2, 0, label=f"Root at {root2:.2f} Mask 2", color="green", zorder=5)

        plt.scatter(root3,0, label=f"Root at {root3:.2f} Mask 3", color="black", zorder=5)
        # Customize the plot
        plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
        plt.axvline(root3, color="purple", linewidth=0.8, linestyle="--")
        plt.legend(fontsize=20, loc = "upper right")
        plt.xlabel("Re(Z)", fontsize=18)
        plt.ylabel("Im(Z)", fontsize=18)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.grid(alpha=0.5)
        plt.xlim(0, 2)
        plt.ylim(-1, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "EIS_interpolation_closeup.png"))
        plt.close()
    # return root3
    if root3 != 0:
        return root3
    # Could not capture the root, EIS is ill defined
    return 0

def get_ohmic_resistance_from_EIS(df, EIS_dict_name = "", experiment_name = "", idx_GEIS = [], plot_data = False):
    
    if idx_GEIS == []:
        Re_Z = df[df["Step name"] == "Galvanostatic EIS"]['Re_Z'].to_numpy()
        Im_Z = -df[df["Step name"] == "Galvanostatic EIS"]['Im_Z'].to_numpy()
    else:
        Re_Z = df[(df["Step name"] == "Galvanostatic EIS") & (df["Step number"] <= idx_GEIS[1])]['Re_Z'].to_numpy()
        Im_Z = -df[(df["Step name"] == "Galvanostatic EIS") & (df["Step number"] <= idx_GEIS[1])]['Im_Z'].to_numpy()
        frequencies = df[(df["Step name"] == "Galvanostatic EIS") & (df["Step number"] <= idx_GEIS[1])]['Frequency [Hz]'].to_numpy()

    try:
        ohmic_res = get_interpolation_EIS(Re_Z, Im_Z, save_plotted_data = save_plotted_data)
        print('Ohmic resistance: ', ohmic_res)
    except:
        
        ohmic_res = get_interpolation_EIS(Re_Z, Im_Z, start_from=8)
        print('Ohmic resistance: ', ohmic_res)
    
    EIS_dict_i = {}

    if os.path.exists(EIS_dict_name):
        with open(EIS_dict_name, 'r') as file:
            
            EIS_dict = json.load(file)

    else:
        EIS_dict = {}

    EIS_dict_i[experiment_name] = {'Re_Z': list(Re_Z), 'Im_Z': list(Im_Z), "IR correction": ohmic_res}

    EIS_dict.update(EIS_dict_i)
    with open(EIS_dict_name, "w") as outfile: 
        json.dump(EIS_dict, outfile)
    
    if plot_data:
        plt.figure(figsize=(12, 9))
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.xlabel("Re_Z [Ohm]", fontsize = 18)
        plt.ylabel("-Im_Z [Ohm]", fontsize = 18)
        plt.scatter(Re_Z, Im_Z)
        plt.show()
    return ohmic_res

def get_forwards_backwards_CV_scan(I_mA_CV_i, E_we_CV_i):
    '''
    Get the forward and backwards CV scan data from the measured current and potential
    '''
    E_we_forward = []
    I_mA_forward = []
    E_we_backwards = []
    I_mA_backwards = []
    
    for i in range(1, len(E_we_CV_i)):
        if E_we_CV_i[i] > E_we_CV_i[i - 1]:
            E_we_forward.append(E_we_CV_i[i])
            I_mA_forward.append(I_mA_CV_i[i])
        else:
            E_we_backwards.append(E_we_CV_i[i])
            I_mA_backwards.append(I_mA_CV_i[i])
    
    return I_mA_backwards, E_we_backwards, I_mA_forward, E_we_forward

def extract_CV_data_from_stability_cycling(df, 
                                           CV_stability_idx = [67, 166],
                                             CV_cycling_dict_path = "", 
                                             save_file_name = "", 
                                             experiment_name = "", 
                                             plot_CVs = False):
    cmap = plt.get_cmap("coolwarm")
    
    
    CV = df[df["Step number"].between(CV_stability_idx[0], CV_stability_idx[1])]
    unique_steps = CV["Step number"].unique()
    colors = cmap(np.linspace(0, 1, len(unique_steps) + 1))
    Es = []
    Is = []
        
    CV_cycling_dict_exp_n = {experiment_name: {}}
    
    if plot_CVs:
        plt.figure(figsize=(12,9 ))
    for i, (step, color) in enumerate(zip(unique_steps, colors), start=1):
        # Filter data for this step
        step_data = CV[CV["Step number"] == step]
        E_V_scan_i = step_data["Working Electrode Voltage [V]"].to_numpy()
        I_A_scan_i = step_data["Current [mA]"].to_numpy()

        # Get forward and backward scan data
        I_A_decreasing, E_V_decreasing, I_A_increasing, E_V_increasing = get_forwards_backwards_CV_scan(I_A_scan_i, E_V_scan_i)
        
        # Add cycle data to the experiment dictionary
        CV_cycling_dict_exp_n[experiment_name][f"Cycle {i}"] = {
            'I_A_increasing': list(I_A_increasing), 
            'I_A_decreasing': list(I_A_decreasing), 
            'E_V_decreasing': list(E_V_decreasing), 
            'E_V_increasing': list(E_V_increasing)
        }
        if plot_CVs:
            plt.plot(E_V_scan_i * 1000, I_A_scan_i * 1000, color = color)

    if plot_CVs:
        plt.xlabel("Working electrode [V]", fontsize = 18)
        plt.ylabel("Current [mA]", fontsize = 18)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        # Save the plot if a file name is provided
        if save_file_name:
            plt.savefig(save_file_name)
        plt.show()

    # Load the existing CV_cycling_dict if it exists, or create an empty one
    if os.path.exists(CV_cycling_dict_path):
        with open(CV_cycling_dict_path, 'r') as infile:
            CV_cycling_dict = json.load(infile)
    else:
        CV_cycling_dict = {}
    
    CV_cycling_dict.update(CV_cycling_dict_exp_n)
    with open(CV_cycling_dict_path, 'w') as outfile:
        json.dump(CV_cycling_dict, outfile, indent=4)

def get_WE_V_at_current_densities(LSV_I, LSV_V, desired_current_densities = [], A = 0.975):
    
    WE_voltages = []
    for current_density in desired_current_densities:
        #print(np.array(LSV_I * A) - current_density)
        idx_argmin = np.argmin(np.abs(np.array(LSV_I * A) - current_density))
        #print(current_density)
        #print(np.min(np.array(LSV_I * A) - current_density))
        WE_voltages.append(LSV_V[idx_argmin])
    
    return WE_voltages

def extract_GEIS_data_general_protocol(df,
                                       use_GEIS_i_for_IR = 0, 
                                       experiment_name = "", 
                                       plot_data = False, 
                                       save_plotted_data = False, 
                                       save_path = None):
    '''
        Extract the general GEIS data for a given protocol 
    '''

    filtered_df = df[df["Step name"] == "Galvanostatic EIS"]
    
    # Get all unique step numbers corresponding to "Potential Linear Sweep"
    unique_step_numbers = sorted(filtered_df["Step number"].unique())

    splits = np.where(np.diff(unique_step_numbers) > 1)[0] + 1

    # Split the array at those indices
    split_values = np.split(unique_step_numbers, splits)

    EIS_dict_i = {experiment_name : {}}
    for i, group in enumerate(split_values):
        
        # Filter rows corresponding to the current group of step numbers
        group_df = filtered_df[filtered_df["Step number"].isin(group)]
        
        # Extract Re_Z and Im_Z for the current group
        Re_Z = group_df['Re_Z'].to_numpy()
        Im_Z = -group_df['Im_Z'].to_numpy()
        
        if save_plotted_data:
            plt.figure(figsize=(12, 9))
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            plt.xlabel("Re_Z [Ohm]", fontsize = 18)
            plt.ylabel("-Im_Z [Ohm]", fontsize = 18)
            plt.scatter(Re_Z, Im_Z)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"GEIS_{i}.png"))
            plt.close()
        if plot_data:
            plt.figure(figsize=(12, 9))
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            plt.xlabel("Re_Z [Ohm]", fontsize = 18)
            plt.ylabel("-Im_Z [Ohm]", fontsize = 18)
            plt.scatter(Re_Z, Im_Z)
            plt.show()
            
            

        EIS_dict_i[experiment_name][f"GEIS_{i + 1}"] = {"Re_Z": list(Re_Z), 
                                                        "Im_Z": list(Im_Z)}
        
        
        if i == use_GEIS_i_for_IR:
            try:
                ohmic_res = get_interpolation_EIS(Re_Z, Im_Z, use_mask = True, save_plotted_data = save_plotted_data, 
                                                  save_path=save_path)
                
            except:
                try:
                    ohmic_res = get_interpolation_EIS(Re_Z, Im_Z, start_from=8, use_mask = True, save_plotted_data = save_plotted_data, 
                                                  save_path=save_path)
                    
                except:
                    ohmic_res = 0
    

    return ohmic_res, EIS_dict_i

def sort_CVs_based_on_scan_rates(df, 
                                 CV_steps,
                                 stability_cycling_scan_rate = 50, 
                                 ECSA_cycling_data = [320, 160, 80, 40, 20, 10, 5]
                                 
                                 ):


    sorted_CVs = {}
    
    print(ECSA_cycling_data)
    for CV_step_i in CV_steps:
        
        CV_V_i = df[df["Step number"] == CV_step_i]['Working Electrode Voltage [V]'].to_numpy()
        CV_I_i = df[df["Step number"] == CV_step_i]['Current [A]'].to_numpy()
        time = df[df["Step number"] == CV_step_i]['Timestamp'].to_numpy()[2: len(CV_I_i) - 2]
        
        CV_V_i_scan_rate_det = df[df["Step number"] == CV_step_i]['Working Electrode Voltage [V]'].to_numpy()[2: len(CV_I_i) - 2]
        
        dV = np.diff(CV_V_i_scan_rate_det)  # Change in voltage
        dt = np.diff(time)    # Change in time
        scan_rates = np.abs(dV / dt) * 1000  # Convert V/s to mV/s
        
        rounded_scan_rates = np.round(scan_rates / 10) * 10
        # Round the average scan rate to the nearest 10
        rounded_scan_rate = int(np.round(np.mean(rounded_scan_rates) / 10) * 10)
        if rounded_scan_rate not in ECSA_cycling_data and rounded_scan_rate != stability_cycling_scan_rate:
            rounded_scan_rate = ECSA_cycling_data[np.argmin(rounded_scan_rate - np.array(ECSA_cycling_data))]
            # Ensure the rounded_scan_rate key exists
        if rounded_scan_rate not in sorted_CVs:
            sorted_CVs[rounded_scan_rate] = {}
        
        sorted_CVs[rounded_scan_rate][f"Scan {CV_step_i}"] = {'Current [mA]': list(CV_I_i * 1000), 
                                            'Working Electrode Voltage [mV]': list(1000 * CV_V_i)}

        
    try:

        #print(sorted_CVs[stability_cycling_scan_rate], "We try to print the stability cycling scan rate")
        stability_cycling_CVs = sorted_CVs[stability_cycling_scan_rate]
    except:
        for scan_rate in sorted_CVs.keys():
            if scan_rate not in ECSA_cycling_data:
                stability_cycling_CVs = sorted_CVs[scan_rate]
    ECSA_cycling_CVs = {scan_rate: scans for scan_rate, scans in sorted_CVs.items() if scan_rate != stability_cycling_scan_rate}
    #stability_cycling_CVs = {scan_rate: scans for scan_rate, scans in sorted_CVs.items() if scan_rate not in ECSA_cycling_data}
    #print(len(matching_CVs), "This is matching CV length")
    return stability_cycling_CVs, ECSA_cycling_CVs

def extract_CV_data_general_protocol(df, 
                                            experiment_name = "", 
                                            scan_rates_ECSA_mV_s = [320, 160, 80, 40, 20, 10, 5], 
                                            scan_rates_stability_mV_s = 50, 
                                            ):
    '''
        TODO:
            Plot the CVs
            Plot the ECSAs extracted from the ECSA scan rates (linear interpolation)
            Save the figures in the correct places
    '''
    
    filtered_df = df[df["Step name"] == "Cyclic Voltammetry"]

    CV_steps_unique_numbers = filtered_df["Step number"].unique()

    stability_cycling_CVs, ECSA_cycling_CVs = sort_CVs_based_on_scan_rates(df=filtered_df,
                                                  CV_steps=CV_steps_unique_numbers, 
                                                  stability_cycling_scan_rate=scan_rates_stability_mV_s, 
                                                  ECSA_cycling_data=scan_rates_ECSA_mV_s)
    
    stability_cycling_CVs = {experiment_name: stability_cycling_CVs}
    ECSA_cycling_CVs = {experiment_name: ECSA_cycling_CVs}

    return stability_cycling_CVs, ECSA_cycling_CVs

def get_stability_data_from_stability_cycling(stability_cycling_dict, 
                                              current_densities = [50], 
                                              geometric_surface_area = 0.97,
                                              get_stability_from = "forward_scan",
                                              plot_CV_data = True, 
                                            IR_correction = 0
                                              ):
    
    
    Es = {"Overpotentials at " + f"{current_density} mA/cm2": [] for current_density in current_densities}
    Is = {"Measured current density at " + f"{current_density} mA/cm2": [] for current_density in current_densities}
    Rs = {"Resistance at " + f"{current_density} mA/cm2": [] for current_density in current_densities}

    experiment_name = list(stability_cycling_dict.keys())[0]

    # Plot each step with a unique colorss
    if plot_CV_data: 
        plt.figure(figsize=(12, 8))
    for i, step_i in enumerate(stability_cycling_dict[experiment_name].keys()):
        
        step_data = stability_cycling_dict[experiment_name][step_i]
        
        closest_index = 0
        E_mV_scan_i = np.array(step_data['Working Electrode Voltage [mV]'])
        I_mA_scan_i = np.array(step_data["Current [mA]"])
        I_mA_decreasing, E_mV_decreasing, I_mA_increasing, E_mV_increasing = get_forwards_backwards_CV_scan(I_mA_scan_i, E_we_CV_i=E_mV_scan_i)

        voltage_interpolated = np.linspace(np.min(E_mV_scan_i), np.max(E_mV_scan_i), 10000)

        if get_stability_from == "forward_scan":
            inter = interpolate.interp1d(E_mV_increasing, np.array(I_mA_increasing) / geometric_surface_area, fill_value="extrapolate")
        elif get_stability_from =="backward_scan":
            inter = interpolate.interp1d(E_mV_decreasing, np.array(I_mA_decreasing) / geometric_surface_area, fill_value="extrapolate")
        
        current_density_interpolation = inter(voltage_interpolated)
        for current_density in current_densities:
            closest_index = np.argmin(np.abs(current_density_interpolation + current_density))
            overpotential_current_density_scan_i = voltage_interpolated[closest_index] - IR_correction * current_density_interpolation[closest_index] * geometric_surface_area
            exact_current_density_scan_i = current_density_interpolation[closest_index]
            resistance_at_current_density_scan_i = (overpotential_current_density_scan_i) / exact_current_density_scan_i # R = U / I

            Es["Overpotentials at " + f"{current_density} mA/cm2"].append(overpotential_current_density_scan_i)
            Is["Measured current density at " + f"{current_density} mA/cm2"].append(exact_current_density_scan_i)
            Rs["Resistance at " + f"{current_density} mA/cm2"].append(resistance_at_current_density_scan_i)
    
    combined_dict = {**Es, **Is, **Rs}
    return combined_dict


def transform_CVs_to_stability_metrics(stability_cycling_CVs, 
                                              current_densities = 50, 
                                              geometric_surface_area = 0.97,
                                              get_stability_from = "forward_scan",
                                              plot_stability_data = True,
                                              plot_CV_data = True, 
                                              cmap = plt.get_cmap("coolwarm"), 
                                            save_path = None, 
                                            IR_correction = 0
                                              ):

    Es = {"Overpotentials at " + f"{current_density} mA/cm2": [] for current_density in current_densities}
    Is = {"Measured current density at " + f"{current_density} mA/cm2": [] for current_density in current_densities}
    Rs = {"Resistance at " + f"{current_density} mA/cm2": [] for current_density in current_densities}
    scan_rate = float(
          round(float(stability_cycling_CVs[0]['params']['variables']['dEdt']) * 1000, 2)
    )  # [mV/s]
    for i, CV_scan in enumerate(stability_cycling_CVs[0]["data"]):
    
        I_mA_cm2_scan_i = CV_scan['current_density [mA/cm2]']
        WE_mV_scan_i = CV_scan['voltage [mV]']

        I_mA_decreasing, E_mV_decreasing, I_mA_increasing, E_mV_increasing = get_forwards_backwards_CV_scan(I_mA_cm2_scan_i, 
                                                                                                            WE_mV_scan_i)
        voltage_interpolated = np.linspace(np.min(WE_mV_scan_i), np.max(WE_mV_scan_i), 10000)

        if get_stability_from == "forward_scan":
            inter = interpolate.interp1d(E_mV_increasing, np.array(I_mA_increasing) / geometric_surface_area, fill_value="extrapolate")
        elif get_stability_from =="backward_scan":
            inter = interpolate.interp1d(E_mV_decreasing, np.array(I_mA_decreasing) / geometric_surface_area, fill_value="extrapolate")
        
        current_density_interpolation = inter(voltage_interpolated)
        for current_density in current_densities:
            closest_index = np.argmin(np.abs(current_density_interpolation + current_density))
            overpotential_current_density_scan_i = voltage_interpolated[closest_index] - IR_correction * current_density_interpolation[closest_index] * geometric_surface_area
            exact_current_density_scan_i = current_density_interpolation[closest_index]
            resistance_at_current_density_scan_i = (overpotential_current_density_scan_i) / exact_current_density_scan_i # R = U / I

            Es["Overpotentials at " + f"{current_density} mA/cm2"].append(overpotential_current_density_scan_i)
            Is["Measured current density at " + f"{current_density} mA/cm2"].append(exact_current_density_scan_i)
            Rs["Resistance at " + f"{current_density} mA/cm2"].append(resistance_at_current_density_scan_i)
    
    combined_dict = {**Es, **Is, **Rs}
    return combined_dict



def group_consecutive_scans(scans):
    grouped_scans = []
    current_group = [scans[0]]  # Start with the first scan

    for i in range(1, len(scans)):
        # Extract scan numbers for comparison
        current_scan = int(scans[i].split(" ")[1])
        previous_scan = int(scans[i - 1].split(" ")[1])

        # Check if scans are consecutive
        if current_scan == previous_scan + 1:
            current_group.append(scans[i])
        else:
            # Break the group and start a new one
            grouped_scans.append(current_group)
            current_group = [scans[i]]

    # Add the last group
    grouped_scans.append(current_group)
    return grouped_scans

def extract_ECSA_data_general_protocol(ECSA_dict, 
                                       experiment_name="", 
                                       plot_data = False, 
                                       ECSA_dict_path = None, 
                                       save_plotted_data = False, 
                                    save_path = None, 
                                    use_scan_rates = [20, 40, 80, 160, 320]):
    
        
    
    avg_cap_current_before = []
    avg_cap_current_after = []

    cmap = plt.get_cmap("coolwarm")

    colors = cmap(np.linspace(0, 1, 8))
    j = 0
    for scan_rate in ECSA_dict[experiment_name].keys():
        j += 1
        scan_rate_data = ECSA_dict[experiment_name][scan_rate]
        grouped_scans = group_consecutive_scans(list(scan_rate_data.keys()))

        before_cycling_i = [int(grouped_scans[0][-1].split(" ")[1]), int(grouped_scans[-1][-1].split(" ")[1])]
        
        idx_before = np.argmin(before_cycling_i)
        for i, scan_group in enumerate(grouped_scans):
            
            final_scan = np.argmax(scan_group)
            
            scan_final = scan_rate_data[scan_group[final_scan]]
            I_mA_scan_final = np.array(scan_final['Current [mA]'])
            E_we_mV_scan_final = np.array(scan_final['Working Electrode Voltage [mV]'])
            
            mask = (E_we_mV_scan_final >= 700) & (E_we_mV_scan_final <= 800)  # Boolean mask for the range
            I_mA_scan_truncated = I_mA_scan_final[mask]
            E_we_mV_scan_truncated = E_we_mV_scan_final[mask]

            diffs = np.diff(E_we_mV_scan_truncated)  # Calculate differences
            E_we_mV_scan_forward = E_we_mV_scan_truncated[:-1][diffs > 0]  # Use only matching indices
            E_we_mV_scan_backward = E_we_mV_scan_truncated[:-1][diffs < 0][1:]  # Use only matching indices

            I_mA_scan_forward = I_mA_scan_truncated[:-1][diffs > 0]
            I_mA_scan_backward = I_mA_scan_truncated[:-1][diffs < 0][1:]

            min_length = min(len(I_mA_scan_forward), len(I_mA_scan_backward))

            # Slice both arrays to the shortest length
            I_mA_scan_forward = I_mA_scan_forward[:min_length]
            I_mA_scan_backward = I_mA_scan_backward[:min_length]
            E_we_mV_scan_forward = E_we_mV_scan_forward[:min_length]
            E_we_mV_scan_backward = E_we_mV_scan_backward[:min_length]
            #plt.scatter(E_we_mV_scan_forward, I_mA_scan_forward, c = colors[j])
            #plt.scatter(E_we_mV_scan_backward, I_mA_scan_backward, c = colors[j], label = f"{scan_rate} f")
            
            # Now calculate the mean as requested
            avg_cap_current = np.mean(np.abs(I_mA_scan_forward) + np.abs(I_mA_scan_backward)) / 2
           
            if i == idx_before:
                
                avg_cap_current_before.append(avg_cap_current)
            else:
                avg_cap_current_after.append(avg_cap_current)
    #plt.legend(loc = "upper left")
    #plt.show()
    if save_plotted_data:
        
        try:
            plt.figure(figsize=(12, 9))

            # Kun inkluder scan rates som vi sætter
            filtered_scan_rates = sorted([rate for rate in ECSA_dict[experiment_name].keys() if rate in use_scan_rates])


            # Prepare data for "before cycling"
            filtered_avg_cap_current_before = [avg_cap_current_before[sorted(list(ECSA_dict[experiment_name].keys())).index(rate)] 
                                            for rate in filtered_scan_rates]
            
            b_bef, a_bef = np.polyfit(sorted(filtered_scan_rates), sorted(filtered_avg_cap_current_before), deg=1)
            plt.scatter(sorted(filtered_scan_rates), sorted(filtered_avg_cap_current_before), color=colors[0], 
                        label=f"Before cycling ECSA = {np.round(b_bef, 5)}")
            plt.plot(filtered_scan_rates, np.array(filtered_scan_rates) * b_bef + a_bef, color=colors[0])

            # Prepare data for "after cycling"
            filtered_avg_cap_current_after = [avg_cap_current_after[sorted(list(ECSA_dict[experiment_name].keys())).index(rate)] 
                                            for rate in filtered_scan_rates]

            b_aft, a_aft = np.polyfit(sorted(filtered_scan_rates), sorted(filtered_avg_cap_current_after), deg=1)
            plt.scatter(sorted(filtered_scan_rates), sorted(filtered_avg_cap_current_after), color=colors[1], 
                        label=f"After cycling ECSA = {np.round(b_aft, 5)}")
            plt.plot(sorted(filtered_scan_rates), sorted(np.array(filtered_scan_rates)) * b_aft + a_aft, color=colors[1])

        except:
            print("Failed to plot after data")

        plt.title(experiment_name[:60])
        plt.ylabel("Current [mA]", fontsize = 20)
        plt.xlabel("Scan rate [mV/s]", fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.xticks(fontsize = 20)
        plt.legend(loc = "upper left", fontsize = 20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "ECSA_before_after.png"))
        if plot_data:
            plt.show()
        plt.close()

    return avg_cap_current_after, avg_cap_current_before, sorted(list(ECSA_dict[experiment_name].keys()))[:-1], sorted(list(ECSA_dict[experiment_name].keys()))
