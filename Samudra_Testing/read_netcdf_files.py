#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert NetCDF gradient files to NumPy arrays with 360 longitude points (0-359°)
and rename variables according to standard naming conventions.
"""

import numpy as np
import xarray as xr
from pathlib import Path
import os

def nc_to_numpy_360lon(nc_file_path, output_dir=None, output_name=None):
    """
    Convert a NetCDF file to a NumPy array with exactly 360 longitude points (0-359°) and save it.
    
    Parameters:
    -----------
    nc_file_path : str or Path
        Path to the NetCDF file
    output_dir : str or Path, optional
        Directory to save the NumPy file, defaults to same directory as input
    output_name : str, optional
        Name to use for the output file, defaults to the same name as input
        
    Returns:
    --------
    numpy_file_path : Path
        Path to the saved NumPy file
    """
    # Convert to Path object
    nc_file_path = Path(nc_file_path)
    
    # Set output directory
    if output_dir is None:
        output_dir = nc_file_path.parent
    else:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # Get the base filename without extension
    base_name = nc_file_path.stem
    
    # Use the provided output name if specified
    output_base_name = output_name if output_name else base_name
    
    # Load the NetCDF file
    print(f"Loading {nc_file_path}...")
    ds = xr.open_dataset(nc_file_path)
    
    # Extract the variable (assumed to be named the same as the file)
    var_name = base_name
    if var_name not in ds:
        # Get the first data variable if the assumed name doesn't match
        var_name = list(ds.data_vars)[0]
    
    # Get the coordinates
    lats = ds.lat.values
    lons = ds.lon.values
    
    # Check if we need to adjust the longitude grid
    if len(lons) != 360:
        print(f"Current longitude grid has {len(lons)} points. Adjusting to 360 points...")
        
        # Create a new 360-point longitude grid (0 to 359)
        new_lons = np.arange(0, 360)
        
        # First, ensure the longitude is properly wrapped to 0-360 range
        wrapped_ds = ds.assign_coords(lon=(((ds.lon + 360) % 360)))
            
        # Interpolate to the new grid
        resampled_ds = wrapped_ds.interp(lon=new_lons)
        
        # Extract the resampled data
        data_array = resampled_ds[var_name].values
        
        # Save the new longitude values
        lons = new_lons
    else:
        # No adjustment needed, just extract the data
        data_array = ds[var_name].values
    
    # Create output path
    numpy_file_path = output_dir / f"{output_base_name}.npy"
    
    # Save as numpy array
    print(f"Saving to {numpy_file_path}...")
    np.save(numpy_file_path, data_array)
    
    # Save coordinates separately
    np.save(output_dir / f"{output_base_name}_lats.npy", lats)
    np.save(output_dir / f"{output_base_name}_lons.npy", lons)
    
    # Also save the dimensions in a readable format
    with open(output_dir / f"{output_base_name}_dimensions.txt", 'w') as f:
        f.write(f"Data shape: {data_array.shape}\n")
        f.write(f"Latitude points: {len(lats)}\n")
        f.write(f"Longitude points: {len(lons)}\n")
        f.write(f"Latitude range: {min(lats)} to {max(lats)}\n")
        f.write(f"Longitude range: {min(lons)} to {max(lons)}\n")
        f.write(f"Original variable name: {var_name}\n")
        f.write(f"New variable name: {output_base_name}\n")
    
    # Close the dataset
    ds.close()
    
    return numpy_file_path

def convert_all_gradient_files(base_dir, output_dir=None):
    """
    Convert all gradient NetCDF files in the directory to NumPy arrays with 360 longitude points.
    
    Parameters:
    -----------
    base_dir : str or Path
        Base directory containing gradient experiment directories
    output_dir : str or Path, optional
        Directory to save the NumPy files, defaults to input directories
    """
    base_dir = Path(base_dir)
    
    # Define the experiment directories
    gradient_dirs = [
        'emu_adj_48_48_1_15_743_1',
        'emu_adj_48_48_1_2_209_1',
        'emu_adj_48_48_1_56_1069_1'
    ]
    
    # Define the gradient file types and their new names
    gradient_types = ['adxx_qnet', 'adxx_tauu', 'adxx_tauv']
    new_names = ['hfds', 'tauuo', 'tauvo']
    
    # Create a mapping for easy lookup
    name_mapping = dict(zip(gradient_types, new_names))
    
    # Process each experiment directory
    for gdir in gradient_dirs:
        exp_dir = base_dir / gdir / 'output'
        
        # Check if directory exists
        if not exp_dir.exists():
            print(f"Directory {exp_dir} not found, skipping...")
            continue
            
        print(f"\nProcessing experiment directory: {exp_dir}")
        
        # Define experiment-specific output directory if needed
        exp_output_dir = output_dir / gdir / 'numpy_output' if output_dir else exp_dir / 'numpy_output'
        os.makedirs(exp_output_dir, exist_ok=True)
        
        # Process each gradient type
        for i, grad_type in enumerate(gradient_types):
            nc_file = exp_dir / f"{grad_type}.nc"
            
            # Check if file exists
            if not nc_file.exists():
                print(f"File {nc_file} not found, skipping...")
                continue
                
            # Get the new name for this gradient type
            new_name = name_mapping[grad_type]
            
            # Convert to numpy with 360 longitude points and the new name
            numpy_file = nc_to_numpy_360lon(nc_file, exp_output_dir, new_name)
            print(f"Converted {nc_file} to {numpy_file} with 360 longitude points and renamed to {new_name}")
            
            # Also save lag values separately (with the new name)
            ds = xr.open_dataset(nc_file)
            lag_values = ds.lag.values
            np.save(exp_output_dir / f"{new_name}_lags.npy", lag_values)
            ds.close()

if __name__ == "__main__":
    # Define the base directory containing the gradient experiments
    base_dir = Path("NETcdf_data")
    
    # Define an optional output directory
    output_dir = Path("Converted_NETcdf")
    
    # Convert all files
    convert_all_gradient_files(base_dir, output_dir)
    
    print("\nAll conversions complete!")