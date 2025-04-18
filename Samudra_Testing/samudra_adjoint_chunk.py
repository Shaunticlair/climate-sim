#!/usr/bin/env python
# coding: utf-8

# # Samudra Adjoint Chunk
# 
# This script demonstrates how to use the SamudraAdjoint model to compute state sensitivities
# using the chunked approach. This is similar to samudra_adjoint.py but uses chunk-based
# gradient tracking for more efficient computation of sensitivities.

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

# Time steps for sensitivity analysis
initial_time = 0        # Starting time step
final_time = 10         # Ending time step 

# Define regions of interest in the ocean: latitude and longitude indices
lat_slice = slice(0,180)    # A chunk around latitude 90
lon_slice = slice(0,360)  # A chunk around longitude 180

# Define the final latitude and longitude for the output: the coords we want to study the sensitivity wrt to
final_lat = 90
final_lon = 180

# Model choice
hist = 1
N_test = 40  # Timesteps to use for testing
state_in_vars_config = "3D_thermo_dynamic_all"
state_out_vars_config = state_in_vars_config
boundary_vars_config = "3D_all_hfds_anom"

# Channel to study (0 is the first channel which is temperature at 2.5m depth)
batch = 0
batch_slice = slice(batch, batch+1)
initial_channels = [76,153,154,155,156,157]
final_channel = 0


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

# Define chunks for gradient tracking
# For this example, we're tracking a region of the grid at initial_channel
# Using slices for all dimensions to maintain proper dimensionality
#in_chunks_dict = {
#    initial_time: [
#        (batch_slice, initial_channel_slice, lat_slice, lon_slice)
#        ],
#}

in_chunks_dict = {
    t: [(batch_slice, slice(ch_num, ch_num+1), lat_slice, lon_slice) for ch_num in initial_channels]
    for t in range(initial_time, final_time)
}

# Define output indices to compute sensitivity for
out_indices = [(0, final_channel, final_lat, final_lon, final_time)]
print(f"Final point for sensitivity: {out_indices}")

print(f"Computing sensitivity with batch {batch}, initial channels {initial_channels}, \
      lat slice {lat_slice}, lon slice {lon_slice}")


### COMPUTE CHUNKED SENSITIVITY AND SAVE RESULTS ###

def compute_chunked_sensitivity(initial_time, final_time):
    """
    Compute sensitivity between a chunk at initial_time and a point at final_time
    """
    sensitivity_results = adjoint_model.compute_state_sensitivity_chunked(
        test_data,
        in_chunks_dict=in_chunks_dict,
        out_indices=out_indices,
        device=device,
        use_checkpointing=True,  # Set to True for larger computation
        timer=timer,  # Pass the timer for profiling
    )

    timer.checkpoint("Finished computing chunked sensitivity")

    # Print the results
    print(f"Computed sensitivity results: {len(sensitivity_results)} output points")

    for i, sens_list in enumerate(sensitivity_results):
        print(f"  Output point {i}: {len(sens_list)} sensitivity tensors")
        for j, sens_tensor in enumerate(sens_list):
            # Check if it's a tensor before trying to access shape
            if isinstance(sens_tensor, torch.Tensor):
                print(f"    Sensitivity tensor {j} shape: {sens_tensor.shape}")
            else:
                print(f"    Sensitivity tensor {j} is not a tensor, but a {type(sens_tensor)}")
    

    # Save the sensitivity results to a file
    # We'll save the first sensitivity tensor for the first output point
    # Save all sensitivity tensors for the first output point
    # Save the sensitivity results to a file
    if len(sensitivity_results) > 0:
        # Access the first output point's sensitivity tensors
        sens_list = sensitivity_results[0]
        
        # Create a counter to track which tensor we're on
        tensor_idx = 0
        
        # For each time in our defined chunks
        for t in sorted(in_chunks_dict.keys()):
            # For each chunk at this time (which corresponds to different channels)
            for chunk_idx, chunk_spec in enumerate(in_chunks_dict[t]):
                if tensor_idx < len(sens_list):
                    # Extract channel from the chunk specification
                    _, channel_slice, _, _ = chunk_spec
                    # Get channel number from slice
                    if isinstance(channel_slice, slice):
                        ch_num = channel_slice.start
                    else:
                        ch_num = channel_slice
                    
                    # Get the sensitivity tensor for this time-channel combination
                    sens_tensor = sens_list[tensor_idx]
                    sensitivity_np = sens_tensor.cpu().numpy()
                    
                    # Create filename with channel information
                    filename = f'chunk_sensitivity_ch{ch_num}_t{t}-{final_time}.npy'
                    np.save(filename, sensitivity_np)
                    print(f"Saved sensitivity tensor for channel {ch_num}, timestep {t} to {filename}")
                    
                    tensor_idx += 1

    return sensitivity_results


timer.checkpoint("Setup complete")

# Run the sensitivity computation
sensitivity_results = compute_chunked_sensitivity(initial_time, final_time)

timer.checkpoint("Finished saving sensitivity results")

print("============================================")
print("    Chunked Sensitivity Analysis Done       ")
print("============================================")