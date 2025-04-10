#!/usr/bin/env python
# coding: utf-8

# # Samudra Adjoint
# 
# This script demonstrates how to use the SamudraAdjoint model to compute state sensitivities.
# The adjoint method allows us to efficiently calculate how changes in the initial state
# affect the final state, which is useful for data assimilation and sensitivity analysis.

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

### SETUP STAGE ###

hist = 1
N_test = 40 # Timesteps to use for testing
state_in_vars_config="3D_thermo_dynamic_all"
state_out_vars_config=state_in_vars_config
boundary_vars_config="3D_all_hfds_anom"

# Configure the environment for CUDA or CPU, and set random seeds for reproducibility.
device = setup.torch_config_cuda_cpu_seed() 

# Based on our model, we get our list of variables
list_list_str, list_num_channels = setup.choose_model(state_in_vars_config, boundary_vars_config, hist)

# Unpack
input_list_str, boundary_list_str, output_list_str, vars_list_str = list_list_str
num_input_channels, num_output_channels = list_num_channels

# Get indices for loading data
s_train, e_train, s_test, e_test = setup.compute_indices(hist=hist, N_samples=2850, N_val=50, N_test=N_test)

# Load the data
test_data, wet, data_mean, data_std = setup.load_data(s_test, e_test, N_test,
                                                        input_list_str, boundary_list_str, output_list_str,
                                                        hist=hist, device=device)


# Initialize SamudraAdjoint with the same parameters as the original model
adjoint_model = model_adjoint.SamudraAdjoint(
    wet=wet.to(device),
    hist=hist,
    ch_width=[num_input_channels] + [200, 250, 300, 400],
    n_out=num_output_channels
)

adjoint_model = setup.load_weights(adjoint_model, state_out_vars_config, 
                 device=device)


print(f"Our data has the shape {test_data[0][0].shape}")

### SELECT SENSITIVITY PARAMETERS ###

# Time steps for sensitivity analysis
initial_time = 0        # Starting time step
final_time =   20          # Ending time step 

# Choose channel index to study
initial_channel_index = 0 
final_channel_index = 0


# Define regions of interest in the ocean: latitude and longitude indices

lats = np.arange(0, 180)
lons = np.arange(0, 360)

    
# Create indices for the full grid
initial_indices = []
c = initial_channel_index
for h in lats:
    for w in lons:
        initial_indices.append([0, c, h, w])

# For the output, we'll just look at a single point
final_lat = 90
final_lon = 180
final_indices = [[0, final_channel_index, final_lat, final_lon]]  # Final indices for sensitivity computation
print(f"Final point for sensitivity: {final_indices}")  

print(f"Computing sensitivity matrix for {len(initial_indices)} initial points and {len(final_indices)} final points")


### COMPUTE SENSITIVITY AND SAVE RESULTS###


# Compute the sensitivity matrix using the more efficient method
sensitivity_matrix = adjoint_model.compute_state_sensitivity(
    test_data,
    initial_indices=initial_indices,
    final_indices=final_indices,
    initial_time=initial_time,
    final_time=final_time,
    device=device,
    use_checkpointing=True  # Set to True for larger computation
)

# Print the results
print(f"Computed sensitivity matrix shape: {sensitivity_matrix.shape}")

# Reshape the sensitivity matrix back to latitude-longitude grid for plotting
lat_size = len(lats)
lon_size = len(lons)
sensitivity_grid = sensitivity_matrix.reshape(lat_size, lon_size)


# Convert sensitivity grid to numpy for masking
sensitivity_grid_np = sensitivity_grid.cpu().numpy()

# Write to file 
sensitivity_output_file = Path(f'adjoint_sensitivity_matrix_t={initial_time},{final_time}.npy')
if sensitivity_output_file.exists():
    print(f"Removing existing file: {sensitivity_output_file}")
    sensitivity_output_file.unlink()

# Save the sensitivity matrix to a file for debugging
np.save(sensitivity_output_file, sensitivity_grid_np)


print("============================================")
print("          Sensitivity Analysis Done         ")
print("============================================")