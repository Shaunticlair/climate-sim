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



# Add the Samudra package to the path
sys.path.append("/nobackup/sruiz5/SAMUDRATEST/Samudra/samudra/")

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

# ## Configure environment

if torch.cuda.is_available():
    print(f"Using device: {torch.cuda.get_device_name()}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("Using CPU")
    import os
    num_cores = os.cpu_count()
    torch.set_num_threads(num_cores)
    print(f"Using {num_cores} CPU cores for computation")

# Set seeds for reproducibility
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.manual_seed(42)
np.random.seed(42)

timer.checkpoint("Environment configured")

# ## Experiment Constants
# Define the experiment constants similar to samudra_rollout.py

# Depth levels
DEPTH_LEVELS = ['2_5', '10_0', '22_5', '40_0', '65_0', '105_0', '165_0', 
                '250_0', '375_0', '550_0', '775_0', '1050_0', '1400_0', 
                '1850_0', '2400_0', '3100_0', '4000_0', '5000_0', '6000_0']

# Define input and output variables
INPT_VARS = {
    "3D_thermo_dynamic_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ] + ["zos"],
    "3D_thermo_all": [
        k + str(j)
        for k in ["thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ] + ["zos"]
}
EXTRA_VARS = {
    "3D_all_hfds_anom": ["tauuo", "tauvo", "hfds", "hfds_anomalies"]
}
OUT_VARS = {
    "3D_thermo_dynamic_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ] + ["zos"],
    "3D_thermo_all": [
        k + str(j)
        for k in ["thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ] + ["zos"]
}
VARS = {
    "3D_thermo_dynamic_all": ['uo', 'vo', 'thetao', 'so', 'zos'],
    "3D_thermo_all": ['thetao', 'so', 'zos'],
}
MODEL_PATHS = {
    "3D_thermo_dynamic_all": "samudra_thermo_dynamic_seed1.pt",
    "3D_thermo_all": "samudra_thermo_seed1.pt",
}

# Choose the model type (thermo or thermo+dynamic)
exp_num_in = "3D_thermo_dynamic_all" # Options: "3D_thermo_all" or "3D_thermo_dynamic_all"
exp_num_extra = "3D_all_hfds_anom"
exp_num_out = exp_num_in
model_path = MODEL_PATHS[exp_num_out]

# Configuration for sensitivity analysis
hist = 1
N_samples = 2850  # Used for train
N_val = 50        # Used for validation
N_test = 20       # Number of time steps to use for testing

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

print("Number of inputs:", num_in)
print("Number of outputs:", num_out)

# Getting start and end indices of train and test
s_train = hist
e_train = s_train + N_samples
e_test = e_train + N_val

timer.checkpoint("Constants set")

# ## Load Data
# Load the necessary data for the model

from pathlib import Path

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

timer.checkpoint("Data loaded")

# Get the wetmask for the model
from utils import extract_wet
wet_zarr = data.wetmask
wet = extract_wet(wet_zarr, outputs_str, hist)
print("Wet mask shape:", wet.shape)

# Create the test data loader
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
    long_rollout=False,  # Setting to False for sensitivity analysis
    device=device,
)



# ## Data 
# The data is available from 1975 to the 2022, at 5-day temporal resolution. The variables in the data is arranged in the following format:
# 
# 
# ```
# thetao_lev_2_5
# thetao_lev_10_0
# thetao_lev_22_5
# ...
# thetao_lev_6000_0
# 
# so_lev_2_5
# so_lev_10_0
# so_lev_22_5
# ...
# so_lev_6000_0
# 
# uo_lev_2_5
# uo_lev_10_0
# uo_lev_22_5
# ...
# uo_lev_6000_0
# 
# vo_lev_2_5
# vo_lev_10_0
# vo_lev_22_5
# ...
# vo_lev_6000_0
# 
# zos
# 
# hfds
# hfds_anomalies
# tauuo
# tauvo
# ```

timer.checkpoint("Data loader created")

# ## Initialize the Adjoint Model
# Build and load the SamudraAdjoint model

from model_adjoint import SamudraAdjoint

# Initialize SamudraAdjoint with the same parameters as the original model
adjoint_model = SamudraAdjoint(
    wet=wet.to(device),
    hist=hist,
    ch_width=[num_in] + [200, 250, 300, 400],
    n_out=num_out
)

# Load the pre-trained weights
weights_path = Path(model_path)
if not weights_path.exists():
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="M2LInES/Samudra", 
        filename=weights_path.name,
        local_dir="."
    )

model_state = torch.load(weights_path, map_location=torch.device(device))["model"]
adjoint_model.load_state_dict(model_state)
adjoint_model = adjoint_model.to(device)

# Turn off gradient tracking for all model parameters to improve performance
for param in adjoint_model.parameters():
    param.requires_grad_(False)

print("SamudraAdjoint model initialized and weights loaded")
timer.checkpoint("Model loaded")

# ## Sensitivity Analysis Configuration
# Define the parameters for sensitivity analysis

# Time steps for sensitivity analysis
initial_time = 0        # Starting time step
final_time =   2          # Ending time step (adjust based on your needs)

# Example: Define regions of interest for sensitivity analysis
# For temperature sensitivity analysis at the ocean surface

channel_index = 0  # This is the index of the first variable in the output (e.g., thetao_lev_2_5)

# Example: Ocean surface temperature sensitivity (first depth level)
# Channel indices for temperature at the surface (first level)
surface_temp_channels = [channel_index]  

# Define regions of interest in the ocean
# Example: Small region in the Pacific
pacific_region = {
    'latitude_indices': np.arange(80, 100),    # Y indices for a region in the Pacific
    'longitude_indices': np.arange(150, 170),  # X indices for a region in the Pacific
}

# y: 180, x: 360, lev: 19, time: 602, y_b: 181, x_b: 361

full_map = {
    'latitude_indices': np.arange(0, 180),  # Full latitude range (0-179)
    'longitude_indices': np.arange(0, 360), # Full longitude range (0-359)
}

pacific_region = full_map


# Test compute_state_sensitivity method
testing_compute_state_sensitivity = True

if testing_compute_state_sensitivity:
    print("\nTesting compute_state_sensitivity method...")
    
    # Use the same point from the earlier test for consistency
    init_c = surface_temp_channels[0]  # First temperature channel
    
    # Create indices for the full grid
    initial_indices = []
    for h in pacific_region['latitude_indices']:
        for w in pacific_region['longitude_indices']:
            initial_indices.append([0, init_c, h, w])
    
    # For the output, we'll just look at a single point (can be modified as needed)
    final_lat = 90
    final_lon = 180
    print(f"Final point for sensitivity: ({init_c}, {final_lat}, {final_lon})")  # For plotting purposes
    final_indices = [[0, init_c, final_lat, final_lon]]  # Final indices for sensitivity computation
    
    print(f"Computing sensitivity matrix for {len(initial_indices)} initial points and {len(final_indices)} final points")
    
    print(test_data[0][0].shape)
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
    lat_size = len(pacific_region['latitude_indices'])
    lon_size = len(pacific_region['longitude_indices'])
    sensitivity_grid = sensitivity_matrix.reshape(lat_size, lon_size)

    # Get the wetmask (inverse of drymask) for the first level
    wet_mask = wet_zarr.isel(lev=0).values  # This gives you the 2D mask for the surface level

    

    # Reshape the sensitivity matrix back to latitude-longitude grid for plotting
    lat_size = len(pacific_region['latitude_indices'])
    lon_size = len(pacific_region['longitude_indices'])
    sensitivity_grid = sensitivity_matrix.reshape(lat_size, lon_size)

    # Get the corresponding wetmask for the region you're analyzing
    region_wet_mask = wet_mask[
        min(pacific_region['latitude_indices']):max(pacific_region['latitude_indices'])+1,
        min(pacific_region['longitude_indices']):max(pacific_region['longitude_indices'])+1
    ]

    

    # Convert sensitivity grid to numpy for masking
    sensitivity_grid_np = sensitivity_grid.cpu().numpy()

    # Write to file for debugging purposes (optional)
    sensitivity_output_file = Path("sensitivity_matrix.npy")
    if sensitivity_output_file.exists():
        print(f"Removing existing file: {sensitivity_output_file}")
        sensitivity_output_file.unlink()

    # Save the sensitivity matrix to a file for debugging
    np.save(sensitivity_output_file, sensitivity_grid_np)

    # Create a masked array where land (wet=0) is masked
    masked_sensitivity = np.ma.masked_array(
        sensitivity_grid_np,
        mask=~region_wet_mask.astype(bool)  # Invert wetmask to get drymask
    )

    # Plot the masked sensitivity map
    plt.figure(figsize=(10, 8))
    plt.imshow(masked_sensitivity, cmap='RdBu_r', interpolation='nearest', origin='lower')
    plt.colorbar(label='Sensitivity')
    plt.title(f'Sensitivity Map (t={initial_time} to t={final_time})')
    plt.xlabel('Longitude Index')
    plt.ylabel('Latitude Index')

    # Save the figure
    plt.savefig('sensitivity_map_with_land.png', dpi=300, bbox_inches='tight')
    plt.show()

    
    timer.checkpoint("Efficient state sensitivity computation and plotting completed")

timer.checkpoint("Process completed")

"""
# Wetmask and region-specific wetmask saving
# Save the full wetmask to a numpy file
wetmask_file = Path("full_wetmask.npy")
if wetmask_file.exists():
    print(f"Removing existing file: {wetmask_file}")
    wetmask_file.unlink()

# Get the full wetmask for the surface level and save it
full_wet_mask = wet_zarr.isel(lev=0).values
np.save(wetmask_file, full_wet_mask)
print(f"Full wetmask saved to {wetmask_file}")

# Also save the region-specific wetmask if needed
region_wetmask_file = Path("region_wetmask.npy")
if region_wetmask_file.exists():
    print(f"Removing existing file: {region_wetmask_file}")
    region_wetmask_file.unlink()
np.save(region_wetmask_file, region_wet_mask)
print(f"Region wetmask saved to {region_wetmask_file}")
"""

print("============================================")
print("          Sensitivity Analysis Done         ")
print("============================================")