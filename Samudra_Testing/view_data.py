import sys
sys.path.append("../samudra/")
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

def safe_format(value):
    """Safely format a value to string, handling various types"""
    try:
        if np.isscalar(value):
            if isinstance(value, (int, float)):
                return f"{value:.6g}"
            return str(value)
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                return safe_format(value.item())
            else:
                return str(value).replace('\n', ' ')
        else:
            return str(value)
    except:
        return str(value)

def inspect_zarr_dataset(data_file):
    """
    Analyze a zarr dataset and print out its important features
    
    Parameters:
    -----------
    data_file : str
        Path to the zarr dataset
    """
    print(f"Loading dataset from: {data_file}")
    data = xr.open_zarr(data_file)
    
    # Basic dataset information
    print("\n=== DATASET OVERVIEW ===")
    print(f"Dataset type: {type(data)}")
    print(f"Dataset dimensions: {data.dims}")
    
    # Coordinates information
    print("\n=== COORDINATES ===")
    for coord_name, coord in data.coords.items():
        print(f"Coordinate: {coord_name}")
        print(f"  - Shape: {coord.shape}")
        print(f"  - Dtype: {coord.dtype}")
        
        # Show limited information about coordinate values
        if len(coord.shape) > 1:
            # For multi-dimensional coordinates, just show shape info
            print(f"  - Multi-dimensional coordinate with shape {coord.shape}")
        elif len(coord) <= 5:
            # For small 1D coordinates, show all values
            try:
                print(f"  - Values: {coord.values}")
            except:
                print("  - Values: [Error displaying values]")
        else:
            # For larger 1D coordinates, just show first and last values with length
            try:
                first_val = coord.values[0]
                last_val = coord.values[-1]
                
                # Handle different data types
                if np.issubdtype(coord.dtype, np.datetime64):
                    print(f"  - Range: {str(first_val)[:19]} to {str(last_val)[:19]} ({len(coord)} points)")
                else:
                    first_str = safe_format(first_val)
                    last_str = safe_format(last_val)
                    print(f"  - Range: {first_str} to {last_str} ({len(coord)} points)")
            except Exception as e:
                print(f"  - Range: [Error displaying range: {str(e)}]")
            
    # Data variables
    print("\n=== DATA VARIABLES ===")
    for var_name, var in data.data_vars.items():
        print(f"Variable: {var_name}")
        print(f"  - Dimensions: {var.dims}")
        print(f"  - Shape: {var.shape}")
        print(f"  - Dtype: {var.dtype}")
        
        # Check for missing values without loading the entire array
        try:
            # Use a small sample to check for NaN values
            sample = var.isel({dim: slice(0, min(10, size)) for dim, size in var.sizes.items()})
            has_missing = np.isnan(sample).any().values
            print(f"  - Contains missing values: {has_missing}")
        except:
            print("  - Could not determine missing values")
        
        # Print some basic statistics if numeric, using a sample to avoid loading the entire array
        if np.issubdtype(var.dtype, np.number):
            try:
                # Take a sample along each dimension to reduce memory usage
                sample_size = 1000  # Adjust as needed
                sample_indices = {}
                
                for dim, size in var.sizes.items():
                    if size > sample_size:
                        # Take evenly spaced indices
                        indices = np.linspace(0, size-1, sample_size, dtype=int)
                        sample_indices[dim] = indices
                
                if sample_indices:
                    sample = var.isel(sample_indices)
                else:
                    sample = var
                
                try:
                    min_val = safe_format(sample.min().values)
                    max_val = safe_format(sample.max().values)
                    mean_val = safe_format(sample.mean().values)
                    std_val = safe_format(sample.std().values)
                    
                    print(f"  - Approx. Min: {min_val}")
                    print(f"  - Approx. Max: {max_val}")
                    print(f"  - Approx. Mean: {mean_val}")
                    print(f"  - Approx. Std: {std_val}")
                    print("  - Note: Statistics based on a sample of the data")
                except Exception as e:
                    print(f"  - Could not format statistics: {str(e)}")
            except Exception as e:
                print(f"  - Could not compute statistics: {str(e)}")
    
    # Dataset attributes
    if data.attrs:
        print("\n=== DATASET ATTRIBUTES ===")
        for attr_name, attr_value in data.attrs.items():
            print(f"{attr_name}: {attr_value}")
    
    # Chunk information (for zarr optimization)
    print("\n=== CHUNKING INFORMATION ===")
    for var_name, var in data.data_vars.items():
        if hasattr(var, 'encoding') and 'chunks' in var.encoding:
            print(f"Variable: {var_name}")
            print(f"  - Chunks: {var.encoding['chunks']}")
    
    # Don't print the entire dataset as it's too large
    print("\n=== DATASET SUMMARY ===")
    print(f"Dataset contains {len(data.data_vars)} variables across {len(data.dims)} dimensions")
    
    return data

if __name__ == "__main__":
    data_file = "./data.zarr"
    data = inspect_zarr_dataset(data_file)