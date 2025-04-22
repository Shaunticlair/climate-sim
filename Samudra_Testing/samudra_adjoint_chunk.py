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

#t=0 is the start of 2014
# We want t_end to be December 2015
t_start = 0 
# 699 days between January 2014 and December 2015: 700/5=140
t_end = 140

t_2year =   0 # A little less than 2 years from t_end
t_1year =   140 - 73 # 1 year back from t_end
t_6months = 140 - 36 # 6 months back from t_end
t_1month =  140 - 6 # 1 month back from t_end

#73 is one year

# Define regions of interest in the ocean: latitude and longitude indices
lat_slice = slice(0,180) #slice(106,156)  # A chunk around latitude 90
lon_slice = slice(0,360) #slice(264,314)  # A chunk around longitude 180

# Define the final latitude and longitude for the output: the coords we want to study the sensitivity wrt to
final_lat = 131
final_lon = 289

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
final_channel = 76


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

#print(list(test_data.inputs.variables))
# Also print all the axes of test_data.inputs xarray object
#print(f"Test data inputs axes: {test_data.inputs.dims}")

raise Exception

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

times = [t_2year, t_1year, t_6months, t_1month] # Times to compute sensitivity wrt to

in_chunks_dict = {
    t: [(batch_slice, slice(ch_num, ch_num+1), lat_slice, lon_slice) 
        for ch_num in initial_channels]
    for t in times
}

# Define output indices to compute sensitivity for
out_indices_dict = {t_end: [(0, final_channel, final_lat, final_lon)]}
print(f"Final point for sensitivity: {out_indices_dict}")

print(f"Computing sensitivity with batch {batch}, initial channels {initial_channels}, \
      lat slice {lat_slice}, lon slice {lon_slice}")


### COMPUTE CHUNKED SENSITIVITY AND SAVE RESULTS ###

def save_sensitivity_results(in_chunks_dict, out_indices_dict, sensitivity_results):
    """
    Compute sensitivity between a chunk at initial_time and a point at final_time
    """
    # Print the results
    print(f"Computed sensitivity results: {len(sensitivity_results)} output points")
    
    for out_time, out_idx_list in out_indices_dict.items():
        for out_idx in out_idx_list:
            for in_time, in_slice_list in in_chunks_dict.items():
                for in_slice in in_slice_list:

                    index = (out_time, out_idx, in_time, str(in_slice)) # Slice is not hashable
                    if index in sensitivity_results:
                        _, chin, _, _ = in_slice
                        _, chout, _, _ = out_idx
                        chin = chin.start if isinstance(chin, slice) else chin

                        sensitivity_tensor = sensitivity_results[index]
                        sensitivity_nparray = sensitivity_tensor.cpu().numpy()
                        # Save the tensor to a file
                        #filename = f'chunk_sensitivity_chout[{chout}]_chin[{chin}]_t[{in_time},{out_time}].npy'
                        filename = f'chunk_sensitivity_chin[{chin}]_chout[{chout}]_t[{in_time},{out_time}].npy'

                        np.save(filename, sensitivity_nparray)
                        print(f"Saved sensitivity tensor for {filename} to {filename}")
                    else:
                        print(f"No sensitivity data for index {index}")



timer.checkpoint("Setup complete")

sensitivity_results = adjoint_model.compute_state_sensitivity_chunked(
        test_data,
        in_chunks_dict=in_chunks_dict,
        out_indices_dict=out_indices_dict,
        device=device,
        use_checkpointing=True,  # Set to True for larger computation
        timer=timer,  # Pass the timer for profiling
    )

timer.checkpoint("Finished computing chunked sensitivity")

# Run the sensitivity computation
sensitivity_results = save_sensitivity_results(
    in_chunks_dict=in_chunks_dict,
    out_indices_dict=out_indices_dict,
    sensitivity_results=sensitivity_results
)

timer.checkpoint("Finished saving sensitivity results")

print("============================================")
print("    Chunked Sensitivity Analysis Done       ")
print("============================================")
