import xarray as xr
import numpy as np

import sys
#sys.path.append("../samudra/")
sys.path.append("/nobackup/sruiz5/SAMUDRATEST/Samudra/samudra/")

from utils import convert_train_data

def profile_mean(ds):
    """Calculate weighted mean across spatial dimensions"""
    try:
        # Try with areacello if it exists
        return ds.weighted(ds.areacello).mean(["x", "y"])
    except AttributeError:
        # Fall back to unweighted mean if areacello doesn't exist
        print("Warning: areacello not found, using unweighted mean")
        return ds.mean(["x", "y"])

def compute_metrics(ground_truth, prediction, variable='thetao'):
    """
    Compute MAE and pattern correlation between ground truth and prediction
    
    Parameters:
    -----------
    ground_truth : xarray.Dataset
        Ground truth data
    prediction : xarray.Dataset
        Prediction data
    variable : str
        Variable name to compute metrics for
    
    Returns:
    --------
    mae : float
        Mean Absolute Error
    correlation : float
        Pattern correlation
    """
    print(f"Computing metrics for {variable}...")
    
    # Extract the variable of interest
    gt = ground_truth[variable]
    pred = prediction[variable]
    
    print(f"Ground truth shape: {gt.shape}, Prediction shape: {pred.shape}")
    
    # Print some data statistics
    print(f"Ground truth NaN count: {np.isnan(gt.isel(time=0)).sum().compute().item()}")
    print(f"Prediction NaN count: {np.isnan(pred.isel(time=0)).sum().compute().item()}")
    
    # First average over time to avoid alignment issues with timestamps
    gt = gt.mean('time')
    pred = pred.mean('time')
    
    # Compute profile mean (zonal mean), drop NaNs before computing
    gt_profile = profile_mean(gt.fillna(0))
    pred_profile = profile_mean(pred.fillna(0))
    
    print(f"Profile shapes - GT: {gt_profile.shape}, Pred: {pred_profile.shape}")
    
    # Check for complete NaN before computing
    if np.isnan(gt_profile).all().compute().item() or np.isnan(pred_profile).all().compute().item():
        print("WARNING: One or both profiles contain all NaNs")
        return np.nan, np.nan
    
    # Calculate MAE - compute for dask arrays, dropping NaNs
    diff = np.abs(pred_profile - gt_profile)
    print(f"Diff has {np.isnan(diff).sum().compute().item()} NaNs out of {diff.size}")
    
    mae_array = diff.mean(skipna=True)
    mae = float(mae_array.compute().values)  # Compute and convert to float
    
    # Compute the arrays for correlation
    gt_profile_computed = gt_profile.compute()
    pred_profile_computed = pred_profile.compute()
    
    # Flatten arrays for correlation calculation
    gt_flat = gt_profile_computed.values.flatten()
    pred_flat = pred_profile_computed.values.flatten()
    
    # Check if we have enough data for correlation
    print(f"Before filtering NaNs: {len(gt_flat)} values")
    
    # Remove NaN values
    mask = ~np.isnan(gt_flat) & ~np.isnan(pred_flat)
    gt_flat = gt_flat[mask]
    pred_flat = pred_flat[mask]
    
    print(f"After filtering NaNs: {len(gt_flat)} values")
    
    # Check if we have enough data for correlation
    if len(gt_flat) < 2:
        print("WARNING: Not enough valid data points for correlation")
        return mae, np.nan
    
    # Calculate correlation
    try:
        correlation = np.corrcoef(gt_flat, pred_flat)[0, 1]
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        correlation = np.nan
    
    return mae, correlation

# Example usage:
if __name__ == "__main__":
    # Paths
    path_to_thermo_pred = "3D_thermo_all_prediction.zarr"
    path_to_thermo_dynamic_pred = "3D_thermo_dynamic_all_prediction.zarr"
    data_file = "./data.zarr"
    
    # Start index and test length (match what's in the paper)
    start_index = 2903
    N_test = 600
    
    # Load data with chunking to avoid memory issues
    print("Loading ground truth data...")
    data = xr.open_zarr(data_file, chunks={'time': 100})
    ds_groundtruth = data.isel(time=slice(start_index, start_index+N_test))
    ds_groundtruth = convert_train_data(ds_groundtruth)
    
    # Load predictions with chunking
    print("Loading thermo prediction data...")
    thermo_pred = xr.open_zarr(path_to_thermo_pred, chunks={'time': 100})
    print("Loading thermo+dynamic prediction data...")
    thermo_dynamic_pred = xr.open_zarr(path_to_thermo_dynamic_pred, chunks={'time': 100})
    
    # Ensure coordinate alignment by matching coordinates
    print("Aligning datasets...")
    
    # For depth level matching
    common_levels = sorted(list(set(ds_groundtruth.lev.values) & 
                              set(thermo_pred.lev.values) & 
                              set(thermo_dynamic_pred.lev.values)))
    
    # Filter to common depths
    ds_groundtruth = ds_groundtruth.sel(lev=common_levels)
    thermo_pred = thermo_pred.sel(lev=common_levels)
    thermo_dynamic_pred = thermo_dynamic_pred.sel(lev=common_levels)
    
    # Compute metrics for temperature profiles
    print("Computing metrics for temperature (thetao):")
    try:
        mae_thermo, corr_thermo = compute_metrics(ds_groundtruth, thermo_pred, 'thetao')
        mae_thermo_dynamic, corr_thermo_dynamic = compute_metrics(ds_groundtruth, thermo_dynamic_pred, 'thetao')
        
        print(f"Thermo MAE: {mae_thermo:.6f} °C")
        print(f"Thermo Pattern Correlation: {corr_thermo:.6f}")
        print(f"Thermo+Dynamic MAE: {mae_thermo_dynamic:.6f} °C")
        print(f"Thermo+Dynamic Pattern Correlation: {corr_thermo_dynamic:.6f}")
    except Exception as e:
        print(f"Error computing temperature metrics: {e}")
    
    # Compute metrics for salinity profiles
    print("\nComputing metrics for salinity (so):")
    try:
        mae_thermo_s, corr_thermo_s = compute_metrics(ds_groundtruth, thermo_pred, 'so')
        mae_thermo_dynamic_s, corr_thermo_dynamic_s = compute_metrics(ds_groundtruth, thermo_dynamic_pred, 'so')
        
        print(f"Thermo MAE: {mae_thermo_s:.6f} psu")
        print(f"Thermo Pattern Correlation: {corr_thermo_s:.6f}")
        print(f"Thermo+Dynamic MAE: {mae_thermo_dynamic_s:.6f} psu")
        print(f"Thermo+Dynamic Pattern Correlation: {corr_thermo_dynamic_s:.6f}")
    except Exception as e:
        print(f"Error computing salinity metrics: {e}")
