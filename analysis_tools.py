import pandas as pd 
import numpy as np 
import json
import os 
from datetime import datetime


def get_df_from_ML_optimization(
    path_suggested_exps=None,
    path_input_params=None,
    path_goal_params=None,
    feature_set=None,
    ):
    

    # ================================================================
    # Default feature set (IDENTICAL to original hard-coded features)
    # ================================================================
    DEFAULT_FEATURE_SET = {
        "NiSO4 (mol/L)": ("input", "Conc Ni/Mo 10:1 liquid"),
        "Na2Mo (mol/L)": ("input", "Conc Mo/Ni 10:1 liquid"),
        "H2SO4 (mol/L)": ("input", "Conc H2SO4"),
        "Dep t (s)": ("input", "Dep time [s]"),
        "Dep I (mA/cm²)": ("input", "Current density mA/cm2"),
        "Dep T (C)": ("input", "Dep electrolye T [C]"),
        "integrated_area": ("goal", "Integrated stability at 10 [mA/cm2]"),
        "ML_optimization": ("meta", "ML_optimization"),
        "timestamp": ("meta", "timestamp"),
        "qUpperConfidence_beta": ("meta", "beta"),
    }

    if feature_set is None:
        feature_set = DEFAULT_FEATURE_SET

    # ================================================================
    # Load beta values from suggested experiments
    # ================================================================
    timestamp_to_beta = {}

    if path_suggested_exps is not None:
        for file in os.listdir(path_suggested_exps):
            if file.endswith(".json") and "suggested" in file:
                with open(os.path.join(path_suggested_exps, file), "r") as f:
                    suggested_exp = json.load(f)

                timestamp = list(suggested_exp.keys())[0]
                timestamp_key = list(suggested_exp.keys())[-1]

                try:
                    beta = suggested_exp[timestamp]["ML-outputs"]["Model params [beta]"]["beta"]
                except Exception:
                    beta = suggested_exp[timestamp]["ML-outputs"]["Model params [beta]"]

                timestamp_str = suggested_exp[timestamp_key]
                dt = datetime.strptime(timestamp_str, "%d.%m.%Y_%H-%M")
                formatted_str = dt.strftime("%d.%m.%Y_%H-%M")

                timestamp_to_beta[formatted_str] = beta

    timestamp_beta_dt = {
        datetime.strptime(ts, "%d.%m.%Y_%H-%M"): beta
        for ts, beta in timestamp_to_beta.items()
    }

    def find_beta_near(dt_query, tolerance_sec=60):
        for dt_key, beta in timestamp_beta_dt.items():
            if abs((dt_key - dt_query).total_seconds()) <= tolerance_sec:
                return beta
        return 0

    # ================================================================
    # Load input and goal parameters
    # ================================================================
    with open(path_input_params, "r") as f:
        input_params = json.load(f)

    with open(path_goal_params, "r") as f:
        goal_params = json.load(f)

    data_rows = []

    # ================================================================
    # Main loop
    # ================================================================
    for experiment_i_goals, experiment_i_inp in zip(goal_params, input_params):

        timestamp_str_exp_i = experiment_i_goals.split("start_time_")[-1].split("_")[0]
        dt = datetime.strptime(timestamp_str_exp_i, "%d.%m.%Y at %H:%M")
        formatted = dt.strftime("%d.%m.%Y_%H-%M")

        beta_value = find_beta_near(dt)

        if beta_value == 0:
            ML_optimization = "qMaxValueEntropySearch"
        else:
            ML_optimization = "qUpperConfidenceBound"

        # ------------------------------------------------------------
        # Feature construction (default or user-defined)
        # ------------------------------------------------------------
        features = {}

        for feature_name, spec in feature_set.items():

            if callable(spec):
                features[feature_name] = spec(
                    input_params[experiment_i_inp],
                    goal_params[experiment_i_goals],
                    beta_value,
                    formatted,
                    ML_optimization,
                )

            else:
                source, key = spec

                if source == "input":
                    features[feature_name] = input_params[experiment_i_inp][key]

                elif source == "goal":
                    features[feature_name] = goal_params[experiment_i_goals][key]

                elif source == "meta":
                    if key == "timestamp":
                        features[feature_name] = formatted
                    elif key == "beta":
                        features[feature_name] = beta_value
                    elif key == "ML_optimization":
                        features[feature_name] = ML_optimization

        data_rows.append(features)


    # Convert to DataFrame
    df = pd.DataFrame(data_rows)

    # Optional: preview the DataFrame
    df = df[df["integrated_area"] != 0]

    return df
