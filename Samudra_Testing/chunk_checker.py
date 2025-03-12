#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
import cftime

# Load the chunked prediction files
chunk1 = xr.open_zarr("3D_thermo_dynamic_all_prediction_chunk1.zarr")
chunk2 = xr.open_zarr("3D_thermo_dynamic_all_prediction_chunk2.zarr")

# Load the full prediction file
full_prediction = xr.open_zarr("3D_thermo_dynamic_all_prediction.zarr")

print("Loaded all zarr files")
print(f"Chunk 1 time dimension: {len(chunk1.time)}")
print(f"Chunk 2 time dimension: {len(chunk2.time)}")
print(f"Full prediction time dimension: {len(full_prediction.time)}")

# Print the time coordinates to understand their structure
print("\nFirst few time values from each dataset:")
print(f"Chunk 1 start: {chunk1.time[0].values}, end: {chunk1.time[-1].values}")
print(f"Chunk 2 start: {chunk2.time[0].values}, end: {chunk2.time[-1].values}")
print(f"Full prediction start: {full_prediction.time[0].values}, end: {full_prediction.time[-1].values}")

# Combine chunks along the time dimension without modifying coordinates
# Just concatenate them directly assuming they represent sequential time periods
combined_chunks = xr.concat([chunk1, chunk2], dim='time')
print(f"Combined chunks time dimension: {len(combined_chunks.time)}")

# Compare the dimensions to ensure they match
print("\nComparing dimensions:")
print(f"Combined chunks: {combined_chunks.dims}")
print(f"Full prediction: {full_prediction.dims}")

# Compare combined chunks with full prediction
# First check if they have the same variables
print("\nComparing variables:")
for var in full_prediction.data_vars:
    if var in combined_chunks.data_vars:
        print(f"Variable {var} exists in both datasets")
    else:
        print(f"Variable {var} only exists in full prediction")

for var in combined_chunks.data_vars:
    if var not in full_prediction.data_vars:
        print(f"Variable {var} only exists in combined chunks")

# If time dimensions don't match, we need to select matching time periods for comparison
if len(combined_chunks.time) != len(full_prediction.time):
    print(f"\nWARNING: Time dimensions don't match. Combined: {len(combined_chunks.time)}, Full: {len(full_prediction.time)}")
    print("Will attempt to compare matching time periods.")
    
    # Find the common time period
    common_len = min(len(combined_chunks.time), len(full_prediction.time))
    combined_chunks = combined_chunks.isel(time=slice(0, common_len))
    full_prediction = full_prediction.isel(time=slice(0, common_len))
    print(f"Comparing first {common_len} time steps")

# Compare the data values
print("\nComparing data values:")
max_differences = {}
mean_differences = {}
are_identical = {}

for var in full_prediction.data_vars:
    if var in combined_chunks.data_vars:
        # Check if shapes match
        if combined_chunks[var].shape != full_prediction[var].shape:
            print(f"Shape mismatch for {var}: combined={combined_chunks[var].shape}, full={full_prediction[var].shape}")
            continue
        
        # Calculate difference
        diff = np.abs(combined_chunks[var].values - full_prediction[var].values)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        is_identical = np.allclose(combined_chunks[var].values, full_prediction[var].values, rtol=1e-5, atol=1e-8)
        
        max_differences[var] = max_diff
        mean_differences[var] = mean_diff
        are_identical[var] = is_identical
        
        print(f"{var}: Max difference = {max_diff:.6e}, Mean difference = {mean_diff:.6e}, Identical = {is_identical}")

# Create visualization of differences for a sample variable
if 'thetao' in full_prediction.data_vars and 'thetao' in combined_chunks.data_vars:
    # Plot surface temperature difference at the first time step
    var = 'thetao'
    level = 0  # Surface level
    time_idx = 0  # First time step
    
    if 'lev' in combined_chunks[var].dims and 'lev' in full_prediction[var].dims:
        diff = combined_chunks[var].isel(time=time_idx, lev=level) - full_prediction[var].isel(time=time_idx, lev=level)
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Difference in {var} at surface (time={time_idx})')
        plt.imshow(diff, cmap='coolwarm')
        plt.colorbar(label='Difference')
        plt.savefig(f'{var}_difference_time{time_idx}.png')
        print(f"\nSaved visualization of {var} difference to {var}_difference_time{time_idx}.png")

# Check if all variables are within acceptable tolerance
all_identical = all(are_identical.values())
print(f"\nAll variables identical within tolerance: {all_identical}")

if not all_identical:
    print("Some variables show differences between the chunked and full predictions.")
    print("This could be due to:")
    print("1. Initialization differences in the chunked approach")
    print("2. Floating-point precision issues")
    print("3. Different random seeds or other implementation details")
    
    # Plot the maximum difference over time for a key variable like temperature
    if 'thetao' in max_differences:
        times = np.arange(len(full_prediction.time))
        max_diffs_over_time = []
        
        for t in range(len(times)):
            diff = np.abs(combined_chunks.thetao.isel(time=t) - full_prediction.thetao.isel(time=t))
            max_diffs_over_time.append(np.max(diff))
        
        plt.figure(figsize=(10, 6))
        plt.title('Maximum temperature difference over time')
        plt.plot(times, max_diffs_over_time)
        plt.xlabel('Time step')
        plt.ylabel('Maximum difference')
        plt.grid(True)
        plt.savefig('max_diff_over_time.png')
        print("\nSaved visualization of maximum difference over time to max_diff_over_time.png")
