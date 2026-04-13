import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
from scipy import interpolate
import matplotlib.pyplot as plt
import os
import json 
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d, median_filter

# All the scripts used here are called upon to analyze the specific Ni - Cr data

# Interpolates IR correction from EIS
def get_interpolation_EIS(Re_Z, Im_Z):
    '''
        Interpolates the curve for EIS, so that we can calculate the ohmic drop 
    '''
    Re_Z, Im_Z = np.array(Re_Z), np.array(Im_Z)
    #compute distance to zero
    distance_to_zero = np.abs(Im_Z)
    #make new mask
    below_re_z_mask = Re_Z < 1.1 # Needs to be adjusted to the needs of the user 
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
        root3 = np.round(np.mean(nearest_points,axis=0)[0], 2)

    
    if root3 != 0:
        return np.round(root3, 2)
    # Could not capture the root, EIS is ill defined
    return 0

def get_forwards_backwards_CV_scan(I_mA_CV_i, E_we_CV_i):
    '''
    Get the forward and backwards scans from the CV data, that is where the voltage increases and decreases and split it up 
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

def make_tafel_plot_LSV(current, voltage):
    target_currents = [-0.5, -1, -2, -5, -8]
    
    # Find indices of the closest current values
    closest_indices = [np.argmin(np.abs(current * 1000 - target)) for target in target_currents]
    
    # Extract the closest current values and their corresponding voltages
    closest_currents = np.array([current[i] for i in closest_indices])
    corresponding_voltages = np.array([voltage[i] for i in closest_indices])
    
    # Plot the Tafel plot (log of absolute current)
    b, a = np.polyfit(np.log10(np.abs(closest_currents)), corresponding_voltages,deg = 1)
    print("Tafel slope:", b)
    plt.figure()
    plt.plot(np.log10(np.abs(closest_currents)), corresponding_voltages, 'o-', label="Tafel plot")
    plt.xlabel("log(Current) [log(mA)]", fontsize=12)
    plt.ylabel("Voltage [V]", fontsize=12)
    plt.title("Tafel Plot", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return 

def save_LSV_to_dict(LSV_dict_path = "", LSV = [], experiment_name = "", LSV_dict_experiment_i = None):

    if LSV_dict_experiment_i != None:
        LSV_dict_i = {experiment_name : {}}
        for i, LSV_i in enumerate(LSV_dict_experiment_i.keys()):
            LSV_I = LSV_dict_experiment_i[LSV_i]["LSV [I]"]
            LSV_V = LSV_dict_experiment_i[LSV_i]["LSV [V]"]
            LSV_dict_i[experiment_name][f"LSV_{i}"] = {"LSV [I]": list(LSV_I),
                                                       "LSV [V]": list(LSV_V)}
            
    else:
        
        LSV_dict_i = {experiment_name : {
            "LSV [I]": list(LSV[0]),
            "LSV [V]": list(LSV[1])
        }}
    if os.path.exists(LSV_dict_path):
        with open(LSV_dict_path, 'r') as file:
            LSV_dict = json.load(file)
            
    else:
        LSV_dict = {}
    

    LSV_dict.update(LSV_dict_i)
    with open(LSV_dict_path, "w") as outfile: 
        json.dump(LSV_dict, outfile)
        
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
    Es_nonIR = {"Overpotentials at " + f"{current_density} mA/cm2 [non IR corrected]": [] for current_density in current_densities}
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
            Es_nonIR["Overpotentials at " + f"{current_density} mA/cm2 [non IR corrected]"].append(voltage_interpolated[closest_index])
    combined_dict = {**Es, **Is, **Rs, **Es_nonIR}
    return combined_dict

def extract_CP_data_general_protocol(df,
                                     CP_dict_path, 
                                      save_file_name = None, 
                                      experiment_name = "", 
                                      plot_data = False, 
                                      A = 0.975, 
                                      IR_correction = 0.4,
                                      save_plotted_data = False, 
                                        save_path = None
                                     ):
    
    '''
        We set the area of the working electrode to be equal to 0.975 

    '''
    filtered_df = df[df["Step name"] == "Constant Current"]

    # Get all unique step numbers corresponding to "Potential Linear Sweep"
    CP_dict_i = {}
    unique_step_numbers = filtered_df["Step number"].unique()
    CP_dict_i[experiment_name] = {}
    for step_i in unique_step_numbers:
        
        CP_V = filtered_df[filtered_df["Step number"] == step_i]['Working Electrode Voltage [V]'].to_numpy()
        CP_t = filtered_df[filtered_df["Step number"] == step_i]['Timestamp'].to_numpy()
        CP_I_A = filtered_df[filtered_df["Step number"] == step_i]['Current [mA]'].to_numpy() # For some reason when we save data is stored in mA
        current_density = np.round(np.mean(CP_I_A) / A * 1000) # Current in mA/cm2 * 1000 
        CP_dict_i[experiment_name][f"Current_density_mA_cm2_{current_density}_step_{step_i}"] = {'Working Electrode Voltage [mV] IR corrected': 
                                                                                                 list(np.abs(CP_V) * 1000 - IR_correction * np.mean(CP_I_A)), 
                                                                                                 'Timestamp' : list(CP_t), 
                                                                                                 "Current [mA]" : list(CP_I_A * 1000), 
                                                                                                 "IR_correction" : IR_correction,
                                                                                                 "Area used for current density" : A
                                                                                                 }
        
        if save_plotted_data:
            plt.figure(figsize=(12, 9))
            plt.plot(CP_t - CP_t[0], CP_V)
            plt.xlabel("Time [s]", fontsize = 18)
            plt.ylabel("Working electrode voltage [V]", fontsize = 18)
            plt.title(f"CP at {current_density} mA_cm2", fontsize = 18)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path,f"CP_mA_cm2_{current_density}_step_{step_i}.png"))
            plt.close()
        
        if plot_data:
            plt.figure(figsize=(12, 9))
            plt.plot(CP_t - CP_t[0], CP_V)
            plt.xlabel("Time [s]", fontsize = 18)
            plt.ylabel("Working electrode voltage [V]", fontsize = 18)
            plt.title(f"CP at {current_density} mA_cm2", fontsize = 18)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            plt.tight_layout()
            plt.show()
        
    
    if os.path.exists(CP_dict_path):
        with open(CP_dict_path, 'r') as infile:
            CP_dict = json.load(infile)
    else:
        CP_dict = {}

    CP_dict.update(CP_dict_i)
    with open(CP_dict_path, 'w') as outfile:
        json.dump(CP_dict, outfile, indent=4)

def extract_LSV_data_general_protocol(df, LSV_dict_path, 
                                      save_file_name = None, 
                                      experiment_name = "", 
                                      plot_data = False, 
                                      save_plotted_data = None, 
                                        save_path = None,
                                        extract_WE_potential_LSV_cycle = None, 
                                        extract_at_current_density = [-100], 
                                        IR_correction = 0
                                        ):
        # Filter the DataFrame for rows where "Step name" is "Potential Linear Sweep"
    filtered_df = df[df["Step name"] == "Potential Linear Sweep"]

    # Get all unique step numbers corresponding to "Potential Linear Sweep"
    LSV_dict = {}
    unique_step_numbers = sorted(filtered_df["Step number"].unique())
    for i, step_number in enumerate(unique_step_numbers):
        LSV_V = df[df["Step number"] == step_number]['Working Electrode Voltage [V]'].to_numpy()
        LSV_I = df[df["Step number"] == step_number]['Current [mA]'].to_numpy()
        
        LSV_dict[f"LSV_{i}"] = {
            "LSV [I]": LSV_I,
            "LSV [V]": LSV_V
        }
        if i == extract_WE_potential_LSV_cycle:

            WE_potentials = get_WE_V_at_current_densities(LSV_I=LSV_I * 1000, 
                                        LSV_V = LSV_V * 1000, 
                                        desired_current_densities=extract_at_current_density, 
                                        A = 0.975)
            print(WE_potentials[0], "Non IR corrected")
            WE_potentials = WE_potentials[0] - IR_correction * extract_at_current_density[0]
            print(WE_potentials, 'IR corrected')
        if save_plotted_data:
            plt.figure(figsize=(12, 9))
            plt.plot(LSV_V * 1000, LSV_I * 1000, label="LSV data")
            plt.xlabel("E applied [V]", fontsize=20)
            plt.ylabel("Current [mA]", fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(loc="upper left")
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"LSV_step_{step_number}_nr_{i}.png"))
            plt.close()
        if plot_data:
            #make_tafel_plot_LSV(LSV_I, LSV_V)
            plt.figure(figsize=(12, 9))
            plt.plot(LSV_V, LSV_I, label="LSV data")
            plt.xlabel("E applied [V]", fontsize=18)
            plt.ylabel("Current [mA]", fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.legend(loc="upper left")
            plt.show()


    save_LSV_to_dict(LSV_dict_path=LSV_dict_path, 
                     experiment_name=experiment_name, 
                     LSV_dict_experiment_i=LSV_dict)
    # Save the plot if a filename is provided
    
    return WE_potentials

def transform_CVs_to_ECSA(
                          CV_series = [], 
                          names = ["Cap current final [mA]"], 
                          scan_idx = -1,
                          plot_CVs = True):
    '''
        Transforms CV from different scan rates into an ECSA value. 
        The shape of each element in the CV_series needs to be the following form
        CV_scans = [ {
            "params" : 'variables' : {"dEdt" : 0.05},  # The CVs at that specific scan rate
            "data" : [{'current_density [mA/cm2]' : [...], 'voltage [mV]' : [...]},.... {}, {}]
        },  .... ]
    '''
    ECSA_dict = {}

    cmap = plt.get_cmap("coolwarm")

    if plot_CVs:
        plt.figure(figsize=(13, 10))
    
    for i, CV_series_i in enumerate(CV_series):
        ###### Find the voltage midpoint 
        slowest_scan = min(
            (scan for series in CV_series for scan in series),
            key=lambda s: float(s['params']['variables']['dEdt'])
        )

        slow_scan_data = slowest_scan["data"][scan_idx]
        E_ref = np.asarray(slow_scan_data['voltage [mV]'])

        E_mid_global = 0.5 * (E_ref.max() + E_ref.min())
        for CV_scans_rate in CV_series_i:
            
            scan_rate = float(
                round(float(CV_scans_rate['params']['variables']['dEdt']) * 1000, 2)
            )  # [mV/s]
            
            ECSA_dict.setdefault(scan_rate, {})

            scan_no_i_at_scan_rate = CV_scans_rate["data"][scan_idx]
            I_mA_cm2 = scan_no_i_at_scan_rate['current_density [mA/cm2]']
            WE_mV = scan_no_i_at_scan_rate['voltage [mV]']

            # Split forward / backward scans
            I_mA_decreasing, E_mV_decreasing, \
            I_mA_increasing, E_mV_increasing = get_forwards_backwards_CV_scan(I_mA_cm2, WE_mV)

            idx_mid_decreasing = np.argmin(np.abs(E_mV_decreasing - E_mid_global))
            idx_mid_increasing = np.argmin(np.abs(E_mV_increasing - E_mid_global))

            cap_current_scan_rate_i = abs(
                I_mA_increasing[idx_mid_increasing] - 
                I_mA_decreasing[idx_mid_decreasing]) / 2

            ECSA_dict[scan_rate][names[i]] = cap_current_scan_rate_i

            # ---------- PLOTTING ----------
            if plot_CVs:
                color = cmap(i / max(1, len(CV_series) - 1))

                # Full CV
                plt.plot(
                    WE_mV, I_mA_cm2,
                    color=color,
                    alpha=0.6,
                    label=f"{names[i]} – {scan_rate} mV/s"
                )

                # Mark midpoint currents
                plt.scatter(
                    [E_mV_increasing[idx_mid_increasing],
                     E_mV_decreasing[idx_mid_decreasing]],
                    [I_mA_increasing[idx_mid_increasing],
                     I_mA_decreasing[idx_mid_decreasing]],
                    color='red',
                    s=60,
                    zorder=5
                )

        if plot_CVs:
            plt.xlabel("Potential (mV)", fontsize=18)
            plt.ylabel("Current density (mA/cm²)", fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.grid(alpha=0.3)
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.show()

    return ECSA_dict

def transform_EIS_data_to_dict(EIS_measurements = [], 
                               names = ["GEIS_1", "GEIS_2"], 
                               plot_EIS = False, 
                               EIS_save_path = None):
    EIS_dict = {}
    for i, EIS_measurement_i in enumerate(EIS_measurements):
        EIS_measurement_i = EIS_measurement_i[0]["data"]
        Re_Z = [m["Re_Z"][0] for m in EIS_measurement_i]
        Im_Z = [-m["Im_Z"][0] for m in EIS_measurement_i]
        freq = [m["frequency [Hz]"][0] for m in EIS_measurement_i]
        try:
            ohmic_res = get_interpolation_EIS(Re_Z, Im_Z)
        except Exception as e:
            print(e)
            ohmic_res = 0
        if plot_EIS:
            # Colormap based on the index of the point
            cmap = plt.get_cmap("coolwarm")
            color = cmap(0.0)   # blue end of coolwarm

            plt.figure(figsize=(10, 8))
            # Plot initial
            plt.scatter(Re_Z, Im_Z, color=color, cmap=cmap, edgecolors='k', s=80)
            plt.xlabel("Re(Z) (Ω)", fontsize=20)
            plt.ylabel("Im(Z) (Ω)", fontsize=20)
            plt.scatter(ohmic_res, 0.0, marker='x', color='red',s=160, linewidths=3, label=r'$R_\Omega$ = ' + str(ohmic_res)) # Mark a cross for the ohmic_res we find 
            plt.grid(True, alpha=0.3)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=16)
            plt.show()
        EIS_dict[names[i]] = {
            "Re_Z": Re_Z,
            "Im_Z": Im_Z,
            "frequency [Hz]": freq,
            "IR correction": ohmic_res
        }
    return EIS_dict

def get_peaks_from_CVs(CV_file_name, 
                       plot_data = False):
    
    '''
        Takes as input the file name where the CVs are located, and from this extracts the CVs, and detects all the peaks in 
        the CVs
    '''

    Ni_calib_data = pd.read_csv(CV_file_name, sep=",")
    
    WE = Ni_calib_data[(Ni_calib_data["Step name"] == "Cyclic Voltammetry") & 
                    (Ni_calib_data["Step number"] == 2)]["Working Electrode Voltage [V]"].to_numpy() * 1000

    I_mA = Ni_calib_data[(Ni_calib_data["Step name"] == "Cyclic Voltammetry") & 
                        (Ni_calib_data["Step number"] == 2)]["Current [A]"].to_numpy() * 1000

    # Mask data where WE > 0 mV and <= 1000 mV
    mask = (WE <= 1000) & (WE > 0)
    WE_filtered = WE[mask]
    I_mA_filtered = I_mA[mask]

    # Apply median filter to remove spikes
    filtered_I_mA = median_filter(I_mA_filtered, size=5)

    # Apply Gaussian smoothing
    smoothed_I_mA = gaussian_filter1d(filtered_I_mA, sigma=70)  

    # Find peaks
    peaks, properties = find_peaks(
        smoothed_I_mA, 
        height=0.2, 
        distance=50, 
        prominence=0.5,  
        width=5  
    )
    if plot_data == True:
    
        plt.figure(figsize=(10, 7))
        plt.plot(WE_filtered, I_mA_filtered, label="Raw Data", alpha=0.5)
        plt.plot(WE_filtered, smoothed_I_mA, label="Smoothed Data", linewidth=2)
        plt.scatter(WE_filtered[peaks], smoothed_I_mA[peaks], color='red', label="Detected Peaks")
        plt.xlabel("Voltage (mV)")
        plt.ylabel("Current (mA)")
        plt.legend()
        plt.show()

    return WE_filtered[peaks]

