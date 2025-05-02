import xarray as xr
import numpy as np
import sys
from pathlib import Path

# Add the Samudra package to the path if needed
path = "/path/to/samudra"  # Replace with your actual path
sys.path.append(path)

# Define the forcings we want to analyze
FORCING_VARS = {
    "3D_all_hfds_anom": ["tauuo", "tauvo", "hfds", "hfds_anomalies"]
}

def analyze_forcings(config="3D_all_hfds_anom"):
    print("Loading data...")
    
    # Use the forcings from the specified configuration
    forcing_vars = FORCING_VARS[config]
    
    # Load data - replace these paths with your actual paths
    data_file = "./data.zarr"
    data_mean_file = "./data_mean.zarr"
    data_std_file = "./data_std.zarr"
    
    data = xr.open_zarr(data_file)
    data_mean = xr.open_zarr(data_mean_file)
    data_std = xr.open_zarr(data_std_file)
    
    # Initialize result containers
    results = []
    
    print(f"Analyzing {len(forcing_vars)} forcing variables using {config} configuration...")
    
    for var in forcing_vars:
        try:
            # Get raw (unnormalized) data
            var_data = data[var]
            
            # Get stored mean and std
            mean = float(data_mean[var].values.flatten()[0])
            std = float(data_std[var].values.flatten()[0])
            
            # Get unnormalized min and max
            raw_min = float(var_data.min().values)
            raw_max = float(var_data.max().values)
            
            # Normalize data
            normalized = ((var_data - data_mean[var]) / data_std[var]).fillna(0)
            
            # Get normalized min and max
            norm_min = float(normalized.min().values)
            norm_max = float(normalized.max().values)
            
            # Store results
            results.append({
                'variable': var,
                'raw_min': raw_min,
                'raw_max': raw_max,
                'norm_min': norm_min,
                'norm_max': norm_max,
                'mean': mean,
                'std': std
            })
            
        except Exception as e:
            print(f"Error processing {var}: {e}")
    
    # Print results
    print("\n=== RESULTS FOR FORCING VARIABLES ===")
    print(f"{'Variable':<20} {'Raw Min':>10} {'Raw Max':>10} {'Norm Min':>10} {'Norm Max':>10} {'Mean':>12} {'Std':>12}")
    print("="*84)
    
    all_norm_mins = []
    all_norm_maxes = []
    
    for result in results:
        print(f"{result['variable']:<20} {result['raw_min']:>10.4f} {result['raw_max']:>10.4f} "
              f"{result['norm_min']:>10.4f} {result['norm_max']:>10.4f} "
              f"{result['mean']:>12.4f} {result['std']:>12.4f}")
        
        all_norm_mins.append(result['norm_min'])
        all_norm_maxes.append(result['norm_max'])
    
    # Print overall min and max
    overall_norm_min = min(all_norm_mins)
    overall_norm_max = max(all_norm_maxes)
    
    print("\n=== SUMMARY ===")
    print(f"Overall normalized min across all forcing variables: {overall_norm_min:.4f}")
    print(f"Overall normalized max across all forcing variables: {overall_norm_max:.4f}")
    
    return results

if __name__ == "__main__":
    analyze_forcings()