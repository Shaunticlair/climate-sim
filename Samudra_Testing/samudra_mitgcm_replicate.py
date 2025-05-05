#!/usr/bin/env python
# coding: utf-8

# # Samudra Adjoint using Average Sensitivity
#
# This script demonstrates how to use the SamudraAdjoint model to compute state sensitivities
# using the chunked approach with the new average sensitivity API.

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
# 699 days between January 2014 and December 2015: 700/5=140
t_start = 0 
#t_end = 72 # Approximately a year (73 might be more accurate, but doesn't divide into 12 months nicely)
t_end = 72

#t_months = [t_end - 6*i for i in range(1,13)]  # Months in a year
#in_times = t_months #[t_2year, t_1year, t_6months, t_1month] #[0] # Times to compute sensitivity wrt to
#in_times = [0, 2, 4, 6, 8]  # Times to compute sensitivity wrt to

in_times = [i*2 for i in range(0, 36)]  # Times to compute sensitivity wrt to: every 10 days

# (126, 324) is the point in the middle of the North Atlantic Ocean
# (90, 180) is the point at the Equatorial Pacific Ocean
#final_lat, final_lon = 90, 180
# Equatorial Pacific, Nantucket, North Atlantic Ocean
final_coords = [(90,180), (131,289), (126, 324)]
initial_channels = [76,154,155,156,157]  #[76]  # Channels to compute sensitivity for
#initial_channels = [76]
final_channel = 76

# Define regions of interest in the ocean: latitude and longitude indices
lat_slice = slice(0,180) #slice(106,156)  # A chunk around latitude 90
lon_slice = slice(0,360) #slice(264,314)  # A chunk around longitude 180

#final_lat_slice = slice(final_lat, final_lat+1)#slice(131, 133)
#final_lon_slice = slice(final_lon, final_lon+1)#slice(289, 291)  # Nantucket 2x2

# Model choice
hist = 1
N_test = 600  # Timesteps to use for testing
state_in_vars_config = "3D_thermo_dynamic_all"
state_out_vars_config = state_in_vars_config
boundary_vars_config = "3D_all_hfds_anom"

# Only 1 batch
batch = 0
batch_slice = slice(batch, batch+1)

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

### Prepare in_list_dict and out_list_dict for avg_state_sensitivity ###

# Modified approach: Create a separate dictionary for each channel-time combination
in_list_dict = []
channel_time_mapping = []  # To keep track of (channel, time) for each in_obj_idx

for channel in initial_channels:
    for t in in_times:
        # Two elements in dictionary: average over two time levels
        channeltime_dict = {t: [(batch_slice, slice(channel, channel+1), lat_slice, lon_slice)],
                            t+1: [(batch_slice, slice(channel + 77, channel + 78), lat_slice, lon_slice)]}
        
        # Add this dictionary to the list
        in_list_dict.append(channeltime_dict)
        
        # Keep track of which index corresponds to which channel-time pair
        channel_time_mapping.append((channel, t))

# Create just one output dictionary with two time slices
odd_final_channel = final_channel + 77
out_list_dict = []
# 6 time slices to average over

for final_coord in final_coords:
    # Each final coordinate creates a distinct output dictionary
    # This, they aren't averaged with each other: they're distinct points
    final_lat, final_lon = final_coord
    final_lat_slice = slice(final_lat, final_lat + 1)  # Slice for latitude
    final_lon_slice = slice(final_lon, final_lon + 1)  # Slice for longitude

    output_dict = {
        
        t_end + 0: [(slice(0,1), slice(final_channel, final_channel+1), final_lat_slice, final_lon_slice)],
        t_end + 1: [(slice(0,1), slice(odd_final_channel, odd_final_channel+1), final_lat_slice, final_lon_slice)],
        t_end + 2: [(slice(0,1), slice(final_channel, final_channel+1), final_lat_slice, final_lon_slice)],
        t_end + 3: [(slice(0,1), slice(odd_final_channel, odd_final_channel+1), final_lat_slice, final_lon_slice)],
        t_end + 4: [(slice(0,1), slice(final_channel, final_channel+1), final_lat_slice, final_lon_slice)],
        t_end + 5: [(slice(0,1), slice(odd_final_channel, odd_final_channel+1), final_lat_slice, final_lon_slice)],
    }

    # Add this output dictionary to the list
    out_list_dict.append(output_dict)



#print(f"Computing sensitivity with batch {batch}, initial channels {initial_channels}, \
#      lat slice {lat_slice}, lon slice {lon_slice}")

### COMPUTE AVERAGE SENSITIVITY AND SAVE RESULTS ###

def save_sensitivity_results(in_list_dict, out_list_dict, sensitivity_results, channel_time_mapping, final_coords):
    """
    Save sensitivity results between chunks and output points
    
    Parameters:
    -----------
    in_list_dict : list of dict
        List of dictionaries, each mapping a time to a list of slices
    out_list_dict : list of dict
        List of dictionaries, each mapping a time to a list of slices
    sensitivity_results : dict
        Dictionary mapping (out_obj_idx, in_obj_idx) to sensitivity tensors
    channel_time_mapping : list
        List of (channel, time) pairs, where the index corresponds to in_obj_idx
    final_coords : list
        List of (lat, lon) coordinates for output points
    """
    # Print the results
    print(f"Computed sensitivity results: {len(sensitivity_results)} entries")
    
    # Create output directory if it doesn't exist
    output_dir = Path("MITgcm_Replication")
    output_dir.mkdir(exist_ok=True)
    
    # For each (out_obj_idx, in_obj_idx) pair in sensitivity_results
    for (out_obj_idx, in_obj_idx), sensitivity_tensor in sensitivity_results.items():
        # Get the output dictionary
        out_dict = out_list_dict[out_obj_idx]
        
        # Get the channel and time for this input index
        chin, in_time = channel_time_mapping[in_obj_idx]
        
        # Get output time range (first and last time in the dictionary)
        out_times = sorted(out_dict.keys())
        out_time_start = out_times[0]
        out_time_end = out_times[-1]
        
        # Extract the output slice from any of the times (they all have the same spatial dimensions)
        # We just need this to get the channel information
        sample_time = out_times[0]
        sample_slice = out_dict[sample_time][0]
        
        # Extract channel information from output slice
        _, chout, _, _ = sample_slice
        chout = chout.start if isinstance(chout, slice) else chout
        
        # Get the output location (lat, lon) based on the out_obj_idx
        final_lat, final_lon = final_coords[out_obj_idx]
        
        # Convert the sensitivity tensor to a numpy array
        sensitivity_nparray = sensitivity_tensor.cpu().numpy()
        
        # Save the tensor to a file with location info in the filename
        filename = f'avg_sensitivity_chin[{chin}]_chout[{chout}]_loc[{final_lat},{final_lon}]_t[{in_time},{out_time_start}-{out_time_end}].npy'
        filepath = output_dir / filename
        np.save(filepath, sensitivity_nparray)
        print(f"Saved sensitivity tensor to {filepath}")

timer.checkpoint("Setup complete")

# Compute sensitivities using the new average state sensitivity method
sensitivity_results = adjoint_model.compute_avg_state_sensitivity(
    test_data,
    in_list_dict=in_list_dict,
    out_list_dict=out_list_dict,
    device=device,
    use_checkpointing=True,  # Set to True for larger computation
    timer=timer,  # Pass the timer for profiling
)

timer.checkpoint("Finished computing average state sensitivity")

# Save the sensitivity results with the channel-time mapping
save_sensitivity_results(
    in_list_dict=in_list_dict,
    out_list_dict=out_list_dict,
    sensitivity_results=sensitivity_results,
    channel_time_mapping=channel_time_mapping
)

timer.checkpoint("Finished saving sensitivity results")

print("============================================")
print("    Average Sensitivity Analysis Done       ")
print("============================================")