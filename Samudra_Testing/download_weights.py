import sys
sys.path.append("../samudra/")

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
#import torch
from huggingface_hub import hf_hub_download, snapshot_download

# Experiment inputs and outputs
DEPTH_LEVELS = ['2_5',
 '10_0',
 '22_5',
 '40_0',
 '65_0',
 '105_0',
 '165_0',
 '250_0',
 '375_0',
 '550_0',
 '775_0',
 '1050_0',
 '1400_0',
 '1850_0',
 '2400_0',
 '3100_0',
 '4000_0',
 '5000_0',
 '6000_0']

INPT_VARS = {
    "3D_thermo_dynamic_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ]
    + ["zos"],
    "3D_thermo_all": [
        k + str(j)
        for k in ["thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ]
    + ["zos"]
}
EXTRA_VARS = {
    "3D_all_hfds_anom": ["tauuo", "tauvo", "hfds", "hfds_anomalies"]
}
OUT_VARS = {
    "3D_thermo_dynamic_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ]
    + ["zos"],
    "3D_thermo_all": [
        k + str(j)
        for k in ["thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ]
    + ["zos"]
}
VARS = {
    "3D_thermo_dynamic_all": ['uo', 'vo', 'thetao', 'so', 'zos'],
    "3D_thermo_all": ['thetao', 'so', 'zos'],
}
MODEL_PATHS = {
    "3D_thermo_dynamic_all": "samudra_thermo_dynamic_seed1.pt",
    "3D_thermo_all": "samudra_thermo_seed1.pt",
}


exp_num_in = "3D_thermo_all" # "3D_thermo_all" or "3D_thermo_dynamic_all"
exp_num_extra = "3D_all_hfds_anom"
exp_num_out = exp_num_in

model_path = MODEL_PATHS[exp_num_out]

hist = 1
N_samples = 2850 # Used for train
N_val = 50 # Used for validation
N_test = 600 # Used for testing

inputs_str = INPT_VARS[exp_num_in]
extra_in_str = EXTRA_VARS[exp_num_extra]
outputs_str = OUT_VARS[exp_num_out]
var_ls = VARS[exp_num_out]

levels = len(DEPTH_LEVELS)

N_atm = len(extra_in_str)  # Number of atmosphere variables
N_in = len(inputs_str)
N_extra = N_atm  # Number of atmosphere variables
N_out = len(outputs_str)

num_in = int((hist + 1) * N_in + N_extra)
num_out = int((hist + 1) * len(outputs_str))

print("Number of inputs: ", num_in)
print("Number of outputs: ", num_out) 

# Getting start and end indices of train and test
s_train = hist
e_train = s_train + N_samples
e_test = e_train + N_val  

import xarray as xr

from pathlib import Path

weights_path = Path(model_path)
hf_hub_download(
    repo_id="M2LInES/Samudra",
    filename=weights_path.name,
    local_dir=".",
)    
"""Section 'Starting...' took 0.0000 seconds
Total elapsed time: 0.0000 seconds
----------------------------------------
Matplotlib is building the font cache; this may take a moment.
Section 'Imports complete!' took 14.5374 seconds
Total elapsed time: 14.5374 seconds
----------------------------------------
Number of inputs:  82
Number of outputs:  78
Section 'Setting up variables complete!' took 0.0001 seconds
Total elapsed time: 14.5375 seconds
----------------------------------------
Section 'Mean downloaded' took 63.0526 seconds
Total elapsed time: 77.5901 seconds
----------------------------------------
Section 'Standard eviations downloaded' took 22.4860 seconds
Total elapsed time: 100.0761 seconds
----------------------------------------
Section 'Downloaded data for time window (2901, 3503)' took 229.5630 seconds
Total elapsed time: 329.6391 seconds
----------------------------------------"""