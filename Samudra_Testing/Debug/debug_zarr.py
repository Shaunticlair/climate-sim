#!/usr/bin/env python
# Data diagnosis script for Samudra

import sys
import xarray as xr
import numpy as np

def inspect_dataset(file_path, name):
    """Inspect the structure of a zarr dataset."""
    print(f"\n=== Inspecting {name}: {file_path} ===")
    try:
        ds = xr.open_zarr(file_path)
        print(f"Dataset dimensions: {ds.dims}")
        print(f"Variables: {list(ds.variables)}")
        
        # Check if 'thetao' exists
        if 'thetao' in ds:
            print(f"thetao shape: {ds.thetao.shape}")
            
            # Test the problematic operation
            print("\nTesting line that caused error:")
            try:
                if 'time' in ds.dims:
                    print(f"Time dimension size: {ds.dims['time']}")
                    if ds.dims['time'] > 0:
                        print("Trying to access time=0...")
                        section_mask = np.isnan(ds.thetao).all('x').isel(time=0)
                        print("Success! No error.")
                    else:
                        print("Warning: time dimension exists but has zero size")
                else:
                    print("Warning: No time dimension found")
            except Exception as e:
                print(f"Error during test: {str(e)}")
                print(f"Exception type: {type(e)}")
        else:
            print("Warning: 'thetao' variable not found")
            
        return ds
    except Exception as e:
        print(f"Error opening dataset: {str(e)}")
        return None

# File paths to check
path_to_thermo_pred = "3D_thermo_all_prediction.zarr"
path_to_thermo_dynamic_pred = "3D_thermo_dynamic_all_prediction.zarr"
data_file = "./data.zarr"

# Inspect each dataset
data_ds = inspect_dataset(data_file, "OM4 data")
thermo_ds = inspect_dataset(path_to_thermo_pred, "Thermo prediction")
thermo_dynamic_ds = inspect_dataset(path_to_thermo_dynamic_pred, "Thermo+Dynamic prediction")

# Try to identify differences in structure
if data_ds is not None and thermo_ds is not None:
    print("\n=== Comparing data structure ===")
    data_coords = set(data_ds.coords)
    thermo_coords = set(thermo_ds.coords)
    print(f"Coords in data but not in thermo: {data_coords - thermo_coords}")
    print(f"Coords in thermo but not in data: {thermo_coords - data_coords}")
    
    # Compare dimensions
    for dim in set(data_ds.dims).union(set(thermo_ds.dims)):
        if dim in data_ds.dims and dim in thermo_ds.dims:
            print(f"Dimension '{dim}': data={data_ds.dims[dim]}, thermo={thermo_ds.dims[dim]}")
        elif dim in data_ds.dims:
            print(f"Dimension '{dim}' only in data: {data_ds.dims[dim]}")
        else:
            print(f"Dimension '{dim}' only in thermo: {thermo_ds.dims[dim]}")

print("\nDiagnosis complete.")
