#!/usr/bin/env python
# coding: utf-8

# # Samudra Loss Adjoint
# 
# This script demonstrates how to use the SamudraAdjoint model to compute loss function sensitivities
# using the adjoint method.

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

# Set the maximum time for the loss sensitivity calculation
t_start = 0 
t_end = 10

# Define regions of interest in the ocean: latitude and longitude indices
lat_slice = slice(0,180)
lon_slice = slice(0,360)

# Define the initial channels to compute sensitivity for
initial_channels = [76]  # Channels to compute sensitivity for

# Model choice
hist = 1
N_test = 40  # Timesteps to use for testing
state_in_vars_config = "3D_thermo_dynamic_all"
state_out_vars_config = state_in_vars_config
boundary_vars_config = "3D_all_hfds_anom"

# Batch index (usually 0 for single batch processing)
batch = 0
batch_slice = slice(batch, batch+1)

# Input times to compute sensitivity for
in_times = [0, 2, 4, 6, 8]

### SETUP STAGE ###

# Configure the environment for CUDA or CPU, and set random seeds for reproducibility
device = setup.torch_config_cuda_cpu_seed() 

# Based on our model, we get our list of variables
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

raw_data, raw_wet, raw_data_mean, raw_data_std = setup.load_data_raw(s_test, e_test, output_list_str)

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

### Choosing which sensitivities to take ###

# Create a weighted loss function
loss_fn = setup.gen_weighted_loss_fn(raw_data, state_in_vars_config)
print("Created weighted loss function")

# Define the chunks for which we want to compute sensitivity
in_chunks_dict = {
    t: [(batch_slice, slice(ch_num, ch_num+1), lat_slice, lon_slice) 
        for ch_num in initial_channels]
    for t in in_times
}

print(f"Computing loss sensitivity with batch {batch}, initial channels {initial_channels}, \
      lat slice {lat_slice}, lon slice {lon_slice}")

### COMPUTE LOSS SENSITIVITY AND SAVE RESULTS ###

def save_sensitivity_results(in_chunks_dict, sensitivity_results):
    """
    Save loss sensitivity results to disk
    """
    # Print the results
    print(f"Computed loss sensitivity results: {len(sensitivity_results)} entries")
    
    for in_time, in_slice_list in in_chunks_dict.items():
        for in_slice in in_slice_list:
            # Create index with string representation of slices
            index = (in_time, str(in_slice))
            
            if index in sensitivity_results:
                # Extract channel information from slices
                _, chin, _, _ = in_slice
                
                # Handle both integer and slice channel indices
                chin = chin.start if isinstance(chin, slice) else chin

                sensitivity_tensor = sensitivity_results[index]
                sensitivity_nparray = sensitivity_tensor.cpu().numpy()
                
                # Save the tensor to a file
                filename = f'loss_sensitivity_chin[{chin}]_t[{in_time},{t_end}].npy'

                np.save(filename, sensitivity_nparray)
                print(f"Saved loss sensitivity tensor for {filename} to {filename}")
            else:
                print(f"No sensitivity data for index {index}")

timer.checkpoint("Setup complete")

# Compute the loss sensitivity
sensitivity_results = adjoint_model.compute_loss_sensitivity(
    test_data,
    in_chunks_dict=in_chunks_dict,
    max_time=t_end,  # Set max_time to 10 as requested
    loss_fn=loss_fn,
    device=device,
    use_checkpointing=True,  # Set to True for larger computation
    timer=timer  # Pass the timer for profiling
)

timer.checkpoint("Finished computing loss sensitivity")

# Save the sensitivity results
save_sensitivity_results(
    in_chunks_dict=in_chunks_dict,
    sensitivity_results=sensitivity_results
)

timer.checkpoint("Finished saving sensitivity results")

print("============================================")
print("    Loss Sensitivity Analysis Complete      ")
print("============================================")