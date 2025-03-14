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

        # Get the data arrays
        combined_data = combined_chunks[var].values
        full_data = full_prediction[var].values
        
        # Create masks for valid (non-NaN) values in both datasets
        combined_mask = ~np.isnan(combined_data)
        full_mask = ~np.isnan(full_data)
        
        # Create a common mask where both datasets have valid values
        common_mask = combined_mask & full_mask
        
        # Skip if there are no valid common points
        if not np.any(common_mask):
            print(f"{var}: No valid common points for comparison")
            max_differences[var] = np.nan
            mean_differences[var] = np.nan
            are_identical[var] = False
            continue
        
        # Calculate difference only for valid points
        diff = np.abs(combined_data[common_mask] - full_data[common_mask])
        
        if len(diff) > 0:
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            # Use numpy.allclose only on valid data points
            is_identical = np.allclose(
                combined_data[common_mask], 
                full_data[common_mask], 
                rtol=1e-5, atol=1e-8
            )
        else:
            max_diff = np.nan
            mean_diff = np.nan
            is_identical = False

        max_differences[var] = max_diff
        mean_differences[var] = mean_diff
        are_identical[var] = is_identical

        print(f"{var}: Max difference = {max_diff:.6e}, Mean difference = {mean_diff:.6e}, Identical = {is_identical}")
        print(f"  Valid comparison points: {np.sum(common_mask):,} out of {common_mask.size:,} ({100*np.sum(common_mask)/common_mask.size:.2f}%)")

# Create visualization of differences for a sample variable
if 'thetao' in full_prediction.data_vars and 'thetao' in combined_chunks.data_vars:
    # Plot surface temperature difference at the first time step
    var = 'thetao'
    level = 0  # Surface level
    time_idx = 0  # First time step

    if 'lev' in combined_chunks[var].dims and 'lev' in full_prediction[var].dims:
        # Get the data
        combined_var = combined_chunks[var].isel(time=time_idx, lev=level)
        full_var = full_prediction[var].isel(time=time_idx, lev=level)
        
        # Calculate difference, handling NaN values
        diff = combined_var - full_var
        
        # Create a mask where both datasets have valid values
        valid_mask = ~np.isnan(combined_var) & ~np.isnan(full_var)
        
        # Calculate statistics for valid points
        valid_diff = diff.values[valid_mask.values]
        if len(valid_diff) > 0:
            vmin = np.percentile(valid_diff, 1)  # 1st percentile for lower bound
            vmax = np.percentile(valid_diff, 99)  # 99th percentile for upper bound
            
            plt.figure(figsize=(10, 6))
            plt.title(f'Difference in {var} at surface (time={time_idx})')
            
            # Only plot differences where both have valid data
            masked_diff = diff.copy()
            masked_diff = masked_diff.where(valid_mask)
            
            # Plot with a symmetric colormap centered at zero
            max_abs = max(abs(vmin), abs(vmax))
            im = plt.imshow(masked_diff, cmap='coolwarm', vmin=-max_abs, vmax=max_abs)
            plt.colorbar(im, label='Difference')
            plt.savefig(f'{var}_difference_time{time_idx}.png')
            print(f"\nSaved visualization of {var} difference to {var}_difference_time{time_idx}.png")
            print(f"  Valid comparison points in plot: {np.sum(valid_mask.values):,} out of {valid_mask.size:,} ({100*np.sum(valid_mask.values)/valid_mask.size:.2f}%)")
        else:
            print(f"\nCannot create visualization for {var}: No valid common points")

# Check if all variables are within acceptable tolerance
valid_comparisons = [v for v in are_identical.values() if v is not None and not np.isnan(v)]
if valid_comparisons:
    all_identical = all(valid_comparisons)
    print(f"\nAll variables identical within tolerance: {all_identical}")
else:
    print("\nNo valid comparisons could be made")

# Plot the maximum difference over time for a key variable like temperature
if 'thetao' in max_differences and not np.isnan(max_differences['thetao']):
    times = np.arange(len(full_prediction.time))
    max_diffs_over_time = []
    valid_point_percentages = []

    for t in range(len(times)):
        combined_data = combined_chunks.thetao.isel(time=t).values
        full_data = full_prediction.thetao.isel(time=t).values
        
        # Create common mask
        common_mask = ~np.isnan(combined_data) & ~np.isnan(full_data)
        valid_point_percentages.append(100 * np.sum(common_mask) / common_mask.size)
        
        if np.any(common_mask):
            diff = np.abs(combined_data[common_mask] - full_data[common_mask])
            max_diffs_over_time.append(np.max(diff))
        else:
            max_diffs_over_time.append(np.nan)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot maximum difference over time
    ax1.set_title('Maximum temperature difference over time')
    ax1.plot(times, max_diffs_over_time)
    ax1.set_ylabel('Maximum difference')
    ax1.grid(True)
    
    # Plot percentage of valid comparison points
    ax2.set_title('Percentage of valid comparison points over time')
    ax2.plot(times, valid_point_percentages)
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Valid points (%)')
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('max_diff_over_time.png')
    print("\nSaved visualization of maximum difference over time to max_diff_over_time.png")
