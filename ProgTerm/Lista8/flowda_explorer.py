import numpy as np
import pandas as pd

configurations = pd.read_json("flowda_data/MF_C3b_60/configurations.json")
instruments = pd.read_json("flowda_data/MF_C3b_60/instruments.json")
measurements = pd.read_json("flowda_data/MF_C3b_60/measurements.json")

exp_ids = measurements["exp_id"]
exp_ids.drop_duplicates(inplace=True)

instrument_ids = measurements["instr_id"]
instrument_ids.drop_duplicates(inplace=True)

print(instrument_ids)

multiflow_dict = {
    "exp_id": [],
    "WC": [],
    "USG": [],
    "USL": [],
    "WC": [],
    "DP_DX_FRIC": [],
}

for exp_id in exp_ids:
    multiflow_dict["exp_id"].append(exp_id)

    index = measurements.index[
        (measurements["exp_id"] == exp_id) & (measurements["instr_id"] == "wc")
    ]
    wc_value = measurements["value"][index].values[0]
    multiflow_dict["WC"].append(wc_value)

    index = measurements.index[
        (measurements["exp_id"] == exp_id) & (measurements["instr_id"] == "usl")
    ]
    usl_value = measurements["value"][index].values[0]
    multiflow_dict["USL"].append(usl_value)

    index = measurements.index[
        (measurements["exp_id"] == exp_id) & (measurements["instr_id"] == "usg")
    ]
    usg_value = measurements["value"][index].values[0]
    multiflow_dict["USG"].append(usg_value)

    index = measurements.index[
        (measurements["exp_id"] == exp_id) & (measurements["instr_id"] == "dp_dx_fric")
    ]
    dp_dx_fric_value = measurements["value"][index].values[0]
    multiflow_dict["DP_DX_FRIC"].append(dp_dx_fric_value)

multiflow_df = pd.DataFrame(multiflow_dict)