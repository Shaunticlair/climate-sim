# inspect_samudra_vars.py
import xarray as xr
import numpy as np
import sys
from pathlib import Path

# Add the Samudra package to the path
path = "/nobackup/sruiz5/SAMUDRATEST/Samudra/samudra/"
sys.path.append(path)

import setup
from utils import convert_train_data

def inspect_variables():
    """Inspect the variables in the Samudra dataset and their organization."""
    
    # Configure the environment
    device = setup.torch_config_cuda_cpu_seed()
    
    # Set up model configuration for testing
    hist = 1
    N_test = 5  # Just need a small sample
    
    # Get indices for loading data
    s_train, e_train, s_test, e_test = setup.compute_indices(
        hist=hist, N_samples=2850, N_val=50, N_test=N_test
    )
    
    # Load data
    print("Loading data...")
    data_file = "./data.zarr"
    if not Path(data_file).exists():
        raise FileNotFoundError(f"Data file {data_file} not found. Please download it first.")
    
    # Load the raw data
    data = xr.open_zarr(data_file)
    
    # Print original variables
    print("\n=== Original Variables ===")
    print("Variables:", list(data.data_vars))
    
    # Get the variable names in the order they appear
    var_names = list(data.data_vars)
    
    # Group variables by prefix
    prefixes = {}
    for var in var_names:
        if "_lev_" in var:
            prefix = var.split("_lev_")[0]
            if prefix not in prefixes:
                prefixes[prefix] = []
            prefixes[prefix].append(var)
    
    # Print the structure by prefix
    print("\n=== Variable Structure ===")
    for prefix in sorted(prefixes.keys()):
        print(f"\n{prefix} variables:")
        # Sort by level
        sorted_vars = sorted(prefixes[prefix], 
                            key=lambda x: float(x.split("_lev_")[1].replace("_", ".")))
        for var in sorted_vars:
            print(f"  {var}")
    
    # Print non-level variables
    print("\nOther variables:")
    for var in var_names:
        if "_lev_" not in var:
            print(f"  {var}")
    
    # Check what happens after convert_train_data
    print("\n=== After convert_train_data ===")
    converted_data = convert_train_data(data.copy())
    print("Variables:", list(converted_data.data_vars))
    
    # Print level values
    if 'lev' in data.coords:
        print("\n=== Level Values ===")
        print(data.lev.values)
    
    return data

if __name__ == "__main__":
    data = inspect_variables()
    
    # Additional inspection of the first variable shape and content
    if len(list(data.data_vars)) > 0:
        var_name = list(data.data_vars)[0]
        var_data = data[var_name]
        print(f"\n=== First Variable: {var_name} ===")
        print(f"Shape: {var_data.shape}")
        print(f"Dimensions: {var_data.dims}")