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

### PARAMETERS ###

# Time steps for sensitivity analysis
initial_time = 0        # Starting time step
final_time =   2          # Ending time step 
# Choose channel index to study
initial_channel_indices = [154,155] # tauuo, tauvo
final_channel_indices = [76]  # zos

num_in_channels = len(initial_channel_indices)
num_out_channels = len(final_channel_indices)
# Define regions of interest in the ocean: latitude and longitude indices
lats = np.arange(0, 180)
lons = np.arange(0, 360)
# Define the final latitude and longitude for the output: the coords we want to study the sensitivity wrt to
final_lat = 90
final_lon = 180
# Model choice
hist = 1
N_test = 40 # Timesteps to use for testing
state_in_vars_config="3D_thermo_dynamic_all"
state_out_vars_config=state_in_vars_config
boundary_vars_config="3D_all_hfds_anom"


### SETUP STAGE ###

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
    
# Create indices for the full grid
initial_indices = []
for c in initial_channel_indices:
    for h in lats:
        for w in lons:
            initial_indices.append([0, c, h, w])

# For the output, we'll just look at a single point

final_indices = [[0, c, final_lat, final_lon] for c in final_channel_indices]  # Final indices for sensitivity computation
print(f"Final point for sensitivity: {final_indices}")  

print(f"Computing sensitivity matrix for {len(initial_indices)} initial points and {len(final_indices)} final points")


### COMPUTE SENSITIVITY AND SAVE RESULTS###

def convert_indices_to_time_indices(indices, t):
    out = []
    first_half = t % 2 == 0 # First half of state vector contains even timestep indices
    for b, c, h, w in indices:
        if c > 157: # Only 158 channels
            raise ValueError("Channel index out of bounds")
        # Channels 154-157 are boundary conditions shared across timesteps: no need to adjust them
        if c < 154: 
            if first_half and c > 76: # Need to adjust for the first half of the state vector
                c = c - 77
            if not first_half and c < 77: # Need to adjust for the second half of the state vector
                c = c + 77
        out.append((b, c, h, w, t))
    return out
                



def one_timestep(initial_time, final_time):
    in_indices = convert_indices_to_time_indices(initial_indices, initial_time)
    out_indices = convert_indices_to_time_indices(final_indices, final_time)

    sensitivity_matrix = adjoint_model.compute_state_sensitivity(
        test_data,
        in_indices=in_indices,
        out_indices=out_indices,
        device=device,
        use_checkpointing=True  # Set to True for larger computation
    )

    # Print the results
    print(f"Computed sensitivity matrix shape: {sensitivity_matrix.shape}")

    # Reshape the sensitivity matrix back to latitude-longitude grid for plotting
    lat_size = len(lats)
    lon_size = len(lons)
    sensitivity_grid = sensitivity_matrix.reshape(num_in_channels, lat_size, lon_size, num_out_channels)


    for in_ch_idx in range(num_in_channels):
        for out_ch_idx in range(num_out_channels):
            # Extract sensitivity for this channel pair
            sensitivity_slice = sensitivity_grid[in_ch_idx, :, :, out_ch_idx]
            
            # Convert to numpy
            sensitivity_np = sensitivity_slice.cpu().numpy()
            
            # Create filename that includes channel information
            filename = f'adjoint_sensitivity_matrix_in_ch{initial_channel_indices[in_ch_idx]}_out_ch{final_channel_indices[out_ch_idx]}_t={initial_time},{final_time}.npy'
            
            # Save to file
            np.save(filename, sensitivity_np)

one_timestep(initial_time, final_time)



def multi_timestep(initial_time, final_time):
    final_time = final_time 
    times = [i for i in range(initial_time, final_time+1)]

    in_indices = [convert_indices_to_time_indices(initial_indices, t) for t in times]
    in_indices = [item for sublist in in_indices for item in sublist]  # Flatten the list
    
    out_indices = convert_indices_to_time_indices(final_indices, final_time)

    sensitivity_matrix = adjoint_model.compute_state_sensitivity(
        test_data,
        in_indices=in_indices,
        out_indices=out_indices,
        device=device,
        use_checkpointing=True  # Set to True for larger computation
    )

    # Print the results
    print(f"Computed sensitivity matrix shape: {sensitivity_matrix.shape}")

    # Reshape the sensitivity matrix back to latitude-longitude grid for plotting
    lat_size = len(lats)
    lon_size = len(lons)
    
    # Reshape to (lat_size, lon_size, len(times))
    sensitivity_grid = sensitivity_matrix.reshape(len(times), lat_size, lon_size)

    # Write each to a numpy file
    for t in range(len(times)):
        sensitivity_grid_t = sensitivity_grid[t, :, :]

        # Convert sensitivity grid to numpy for masking
        sensitivity_grid_np = sensitivity_grid_t.cpu().numpy()

        # Write to file 
        sensitivity_output_file = Path(f'adjoint_sensitivity_matrix_t={t},{final_time}.npy')
        if sensitivity_output_file.exists():
            print(f"Removing existing file: {sensitivity_output_file}")
            sensitivity_output_file.unlink()

        # Save the sensitivity matrix to a file for debugging
        np.save(sensitivity_output_file, sensitivity_grid_np)


#multi_timestep(initial_time, final_time)

#final_time = 20
#for initial_time in range(0, 19):

print("============================================")
print("          Sensitivity Analysis Done         ")
print("============================================")
