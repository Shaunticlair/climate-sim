import xarray as xr
import numpy as np
import sys
from pathlib import Path

# Add the Samudra package to the path if needed
path = "/path/to/samudra"  # Replace with your actual path
path = "/nobackup/sruiz5/SAMUDRATEST/Samudra/samudra/"
sys.path.append(path)

# Define the variables we want to analyze (all state variables, not forcings)
DEPTH_LEVELS = ['2_5', '10_0', '22_5', '40_0', '65_0', '105_0', '165_0', 
                '250_0', '375_0', '550_0', '775_0', '1050_0', '1400_0', 
                '1850_0', '2400_0', '3100_0', '4000_0', '5000_0', '6000_0']

# Define state variables (not forcings)
state_vars = []
# Temperature and salinity
for prefix in ["thetao_lev_", "so_lev_"]:
    for level in DEPTH_LEVELS:
        state_vars.append(f"{prefix}{level}")
# Velocity fields
for prefix in ["uo_lev_", "vo_lev_"]:
    for level in DEPTH_LEVELS:
        state_vars.append(f"{prefix}{level}")
# Surface height
state_vars.append("zos")

def analyze_normalized_state_vars():
    print("Loading data...")
    
    # Load data - replace these paths with your actual paths
    data_file = "./data.zarr"
    data_mean_file = "./data_mean.zarr"
    data_std_file = "./data_std.zarr"
    
    data = xr.open_zarr(data_file)
    data_mean = xr.open_zarr(data_mean_file)
    data_std = xr.open_zarr(data_std_file)
    
    # Initialize min/max values
    all_mins = []
    all_maxes = []
    
    print(f"Analyzing {len(state_vars)} state variables...")
    
    for var in state_vars:
        try:
            # Skip if variable doesn't exist
            if var not in data:
                print(f"Skipping {var} - not found in dataset")
                continue
                
            # Normalize: (data - mean) / std
            normalized = (data[var] - data_mean[var]) / data_std[var]
            
            # Get min and max, skip NaN values
            var_min = normalized.min().values.item()
            var_max = normalized.max().values.item()
            
            # Add to lists
            if not np.isnan(var_min) and not np.isnan(var_max):
                all_mins.append(var_min)
                all_maxes.append(var_max)
                
            print(f"{var}: min={var_min:.4f}, max={var_max:.4f}")
            
        except Exception as e:
            print(f"Error processing {var}: {e}")
    
    # Print overall min and max
    overall_min = min(all_mins)
    overall_max = max(all_maxes)
    
    print("\n=== SUMMARY ===")
    print(f"Overall min across all normalized state variables: {overall_min:.4f}")
    print(f"Overall max across all normalized state variables: {overall_max:.4f}")
    
    return overall_min, overall_max

if __name__ == "__main__":
    analyze_normalized_state_vars()