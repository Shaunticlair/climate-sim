#!/usr/bin/env python
# coding: utf-8

# # Samudra Perturbation
# 
# This script demonstrates how to use the SamudraAdjoint model to compute state sensitivities
# using the finite difference (perturbation) approach.

import time
import random
import sys
import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

import setup
import model_adjoint

timer = setup.Timer()

### PARAMETERS ###

# t=0 is the start of 2014
# We want t_end to be December 2015
t_start = 0 
t_end = 72 # Approximately a year
#t_months = [t_end - 6*i for i in range(1,13)]  # Months in a year
t_months = [0,18,36,54] # Every 3 months from the start
print("Months in a year:", t_months)

# Define the final latitude and longitude for the output
final_lat = 90
final_lon = 180

in_lats = list(range(90-2, 90+2+1 ))
in_lons = list(range(180-2, 180+2+1 ))
# Model choice
hist = 1
N_test = 40  # Timesteps to use for testing
state_in_vars_config = "3D_thermo_dynamic_all"
state_out_vars_config = state_in_vars_config
boundary_vars_config = "3D_all_hfds_anom"

# Batch and channels to study
batch = 0
initial_channels = [76]#, 153, 154, 155, 156, 157]
final_channel = 76

# Times to compute sensitivity for
in_times = t_months

# Perturbation size for finite difference calculation
perturbation_size = 1e-4

### SETUP STAGE ###

# Configure the environment for CUDA or CPU, and set random seeds for reproducibility
device = setup.torch_config_cuda_cpu_seed() 

# Based on our model, get our list of variables
list_list_str, list_num_channels = setup.choose_model(state_in_vars_config, boundary_vars_config, hist)

# Unpack
input_list_str, boundary_list_str, output_list_str, vars_list_str = list_list_str
num_input_channels, num_output_channels = list_num_channels

# Get indices for loading data
s_train, e_train, s_test, e_test = setup.compute_indices(hist=hist, N_samples=2850, N_val=50, N_test=N_test)

timer.checkpoint("Environment configured")

# Load the data
test_data, wet, data_mean, data_std = setup.load_data(s_test, e_test, N_test,
                                                     input_list_str, boundary_list_str, output_list_str,
                                                     hist=hist, device=device)

timer.checkpoint("Data loaded")

# Initialize SamudraAdjoint with the same parameters as the original model
adjoint_model = model_adjoint.SamudraAdjoint(
    wet=wet.to(device),
    hist=hist,
    ch_width=[num_input_channels] + [200, 250, 300, 400],
    n_out=num_output_channels
)

adjoint_model = setup.load_weights(adjoint_model, state_out_vars_config, device=device)

timer.checkpoint("Model loaded")

print(f"Our data has the shape {test_data[0][0].shape}")

### Choosing which sensitivities to calculate ###

# Structure source coordinates as a dictionary mapping timesteps to list of coordinates
source_coords_dict = {}
for t in in_times:
    source_coords_list = []
    for ch_num in initial_channels:
        # For each source time and channel, we'll compute sensitivities for specific points
        source_coords_list.append((batch, ch_num, final_lat, final_lon))
    source_coords_dict[t] = source_coords_list

# Structure target coordinates as a dictionary mapping timesteps to list of coordinates
target_coords_dict = {
    t_end: [(batch, final_channel, final_lat, final_lon)]
}

print(f"Computing sensitivity from sources at times {list(source_coords_dict.keys())}")
print(f"Computing sensitivity to targets at times {list(target_coords_dict.keys())}")

### COMPUTE SENSITIVITY USING FINITE DIFFERENCE METHOD ###

def save_sensitivity_results(source_coords_dict, target_coords_dict, sensitivity_results):
    """
    Save sensitivity results between source coordinates and target coordinates
    """
    print(f"Saving sensitivity results: {len(sensitivity_results)} sensitivities")
    
    for target_time, target_coords_list in target_coords_dict.items():
        for target_coord in target_coords_list:
            for source_time, source_coords_list in source_coords_dict.items():
                for source_coord in source_coords_list:
                    index = (target_time, target_coord, source_time, source_coord)
                    
                    if index in sensitivity_results:
                        _, target_ch, target_lat, target_lon = target_coord
                        _, source_ch, source_lat, source_lon = source_coord
                        
                        sensitivity_value = sensitivity_results[index]
                        
                        # Convert to NumPy array if it's a tensor
                        if isinstance(sensitivity_value, torch.Tensor):
                            sensitivity_nparray = sensitivity_value.cpu().numpy()
                        else:
                            sensitivity_nparray = np.array([sensitivity_value])
                        
                        # Save the sensitivity to a file
                        filename = f'perturbation_sensitivity_chin[{source_ch}]_chout[{target_ch}]_t[{source_time},{target_time}].npy'
                        np.save(filename, sensitivity_nparray)
                        print(f"Saved sensitivity for {filename}: {sensitivity_nparray.item():.6e}")
                    else:
                        print(f"No sensitivity data for index {index}")

timer.checkpoint("Setup complete")

# Run the sensitivity computation using finite difference perturbation
sensitivity_results = adjoint_model.compute_fd_sensitivity(
    test_data,
    source_coords_dict=source_coords_dict,
    target_coords_dict=target_coords_dict,
    perturbation_size=perturbation_size,
    device=device,
    use_checkpointing=True,
    timer=timer
)

timer.checkpoint("Finished computing perturbation sensitivity")

# Save the sensitivity results
save_sensitivity_results(
    source_coords_dict=source_coords_dict,
    target_coords_dict=target_coords_dict,
    sensitivity_results=sensitivity_results
)

timer.checkpoint("Finished saving sensitivity results")

print("============================================")
print("    Perturbation Sensitivity Analysis Done  ")
print("============================================")