#!/usr/bin/env python
# coding: utf-8

# # Samudra with Parallelization Profiling

import time

class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.last_checkpoint = self.start_time

    def checkpoint(self, section_name):
        current_time = time.time()
        section_duration = current_time - self.last_checkpoint
        total_duration = current_time - self.start_time
        print(f"Section '{section_name}' took {section_duration:.4f} seconds")
        print(f"Total elapsed time: {total_duration:.4f} seconds")
        print("-" * 40)
        self.last_checkpoint = current_time

# Start the clock
timer = Timer()

timer.checkpoint("Starting...")

import sys
sys.path.append("../samudra/")

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download

# Import the timed model classes
from timer_model import TimedSamudra, reset_timers, get_timing_stats, timed_generate_model_rollout

timer.checkpoint("Imports")

# ## Configs

if torch.cuda.is_available():
    print(f"Using device: {torch.cuda.get_device_name()}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

timer.checkpoint("Torch config")

# #### Experiment Constants

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

exp_num_in = "3D_thermo_dynamic_all" # "3D_thermo_all" or "3D_thermo_dynamic_all"
exp_num_extra = "3D_all_hfds_anom"
exp_num_out = exp_num_in

model_path = MODEL_PATHS[exp_num_out]

# ### Data Configs

hist = 1
N_samples = 2850 # Used for train
N_val = 50 # Used for validation
N_test = 4 # Used for testing
# Should take 1 min/2 N_test

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

timer.checkpoint("Set constants")

import xarray as xr

# Replace these local zarr paths if required
data_mean_file = "./data_mean.zarr"
data_std_file = "./data_std.zarr"
data_file = "./data.zarr"

if not Path(data_mean_file).exists():
    raise Exception("This file should exist already!")
    data_mean = xr.open_dataset("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4_means", engine='zarr', chunks={})
    data_mean.to_zarr(data_mean_file, mode="w")

data_mean = xr.open_zarr("./data_mean.zarr")

if not Path(data_std_file).exists():
    data_std = xr.open_dataset("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4_stds", engine='zarr', chunks={})
    data_std.to_zarr(data_std_file, mode="w")
data_std = xr.open_zarr(data_std_file)

if not Path(data_file).exists():
    data = xr.open_dataset("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4", engine='zarr', chunks={})
    data = data.isel(time=slice(e_test, e_test+hist+1+N_test)) # We only require the data in the test period
    data.to_zarr(data_file, mode="w")
data = xr.open_zarr(data_file)
data

timer.checkpoint("Data retrieved")

# We require the wet mask to pass to the data loader
from utils import extract_wet
wet_zarr = data.wetmask
wet = extract_wet(wet_zarr, outputs_str, hist)
print("Wet resolution:", wet.shape)

from data_loaders import Test

test_data = Test(
    data,
    inputs_str,
    extra_in_str,
    outputs_str,
    wet,
    data_mean,
    data_std,
    N_test,
    hist,
    0,
    long_rollout=True,
    device=device,
)

timer.checkpoint("Dataloader run")

# ## Build Model
# Instead of the original Samudra model, we'll use the TimedSamudra for profiling

# Create the timed model instead of the original
model = TimedSamudra(n_out=num_out, ch_width=[num_in]+[200,250,300,400], wet=wet.to(device), hist=hist)

timer.checkpoint("Model create")

# ### Load weights
weights_path = Path(model_path)
if not weights_path.exists():
    hf_hub_download(
        repo_id="M2LInES/Samudra",
        filename=weights_path.name,
        local_dir=".",
    )

timer.checkpoint("Weights retrieved")

model.load_state_dict(torch.load(weights_path, map_location=torch.device(device))["model"])
model = model.to(device)

timer.checkpoint("Weights loaded to model")

# Reset timers before running inference
reset_timers()

# Use the timed_generate_model_rollout function instead of the original
model_pred, timing_report = timed_generate_model_rollout(
    N_test,
    test_data,
    model,
    hist,
    N_out,
    N_extra,
    initial_input=None,
)

timer.checkpoint("Generate forward rollout")

# Print the parallelization timing results
print("\n===== PARALLELIZATION ANALYSIS =====")
print(f"Total execution time: {timing_report['total_time']:.4f} seconds")
print(f"Parallelizable time: {timing_report['parallel_time']:.4f} seconds ({timing_report['parallel_percentage']:.2f}%)")
print(f"Sequential time: {timing_report['sequential_time']:.4f} seconds ({100-timing_report['parallel_percentage']:.2f}%)")

# Print detailed component timings
print("\n===== COMPONENT BREAKDOWN =====")
for component, times in timing_report['detailed_stats'].items():
    par_time = times.get('parallel', 0)
    seq_time = times.get('sequential', 0)
    comp_total = par_time + seq_time
    if comp_total > 0:
        par_pct = (par_time / comp_total * 100) if comp_total > 0 else 0
        print(f"{component}:")
        print(f"  Total: {comp_total:.4f}s, Parallel: {par_time:.4f}s ({par_pct:.2f}%), Sequential: {seq_time:.4f}s ({100-par_pct:.2f}%)")

# Save the timing report to a file
import json
with open('parallelization_profile.json', 'w') as f:
    json.dump(timing_report, f, indent=2)
print("\nDetailed timing report saved to 'parallelization_profile.json'")

# #### Convert the prediction and ground truth data to the correct format useful for plotting and comparison
from utils import post_processor, convert_train_data
ds_prediction = xr.DataArray(
            data=model_pred,
            dims=["time", "x", "y", "var"]
        )
ds_prediction = ds_prediction.to_dataset(name="predictions")
ds_groundtruth = data.isel(time=slice(hist+1,hist+1+N_test))
ds_prediction = post_processor(ds_prediction, ds_groundtruth, var_ls)
ds_groundtruth = convert_train_data(ds_groundtruth)

timer.checkpoint("Data post-processing")

# ## Save rollout
ds_prediction.to_zarr(exp_num_in + '_prediction.zarr', mode='w')
timer.checkpoint("Predictions made")
