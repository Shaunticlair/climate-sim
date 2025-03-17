import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from datetime import datetime

# Add necessary path
sys.path.append("/nobackup/sruiz5/SAMUDRATEST/Samudra/samudra/")

from utils import convert_train_data

def compute_zonal_mean_metrics(ds_truth, ds_pred, variable='thetao'):
    """
    Compute metrics for zonal mean of potential temperature.
    This computes metrics on the zonal mean (longitude-averaged) field
    as shown in the Samudra paper's Figure 1b.
    """
    # Get data
    truth = ds_truth[variable]
    pred = ds_pred[variable]
    
    # Print time dimensions
    print(f"Truth time dimension: {truth.time.size} steps")
    print(f"Prediction time dimension: {pred.time.size} steps")
    
    # Area weighting for longitude averaging
    area_weights = ds_truth['areacello']
    
    # Compute zonal mean (longitude-averaged) for both datasets
    # First compute weights for each longitude point
    x_weights = area_weights.sum(dim='y')
    x_weights = x_weights / x_weights.sum(dim='x')
    
    # Apply weights to compute zonal mean
    truth_zonal = (truth * x_weights).sum(dim='x')
    pred_zonal = (pred * x_weights).sum(dim='x')
    
    # Time averaging
    truth_zonal_time_mean = truth_zonal.mean(dim='time')
    pred_zonal_time_mean = pred_zonal.mean(dim='time')
    
    # Get depth layer thicknesses and latitudinal weights
    dz = ds_truth['dz']  # Has dimension 'lev'
    y_weights = area_weights.sum(dim='x')  # Has dimension 'y'

    # Normalize weights
    dz_norm = dz / dz.sum()
    y_weights_norm = y_weights / y_weights.sum()

    # Create weight arrays with both dimensions
    # This preserves the original coordinates
    combined_weights = xr.ones_like(truth_zonal_time_mean)

    # Multiply by each weight along its proper dimension
    combined_weights = combined_weights * dz_norm
    combined_weights = combined_weights * y_weights_norm

    # Renormalize
    combined_weights = combined_weights / combined_weights.sum()
    
    # Compute MAE between the time-averaged zonal means with proper weighting
    mae = (np.abs(pred_zonal_time_mean - truth_zonal_time_mean) * combined_weights).sum()
    
    # Compute pattern correlation between the time-averaged zonal means
    # First remove the mean across all depths and latitudes (weighted)
    truth_mean = (truth_zonal_time_mean * combined_weights).sum()
    pred_mean = (pred_zonal_time_mean * combined_weights).sum()
    
    truth_anom = truth_zonal_time_mean - truth_mean
    pred_anom = pred_zonal_time_mean - pred_mean
    
    # Compute correlation with proper weighting
    num = (truth_anom * pred_anom * combined_weights).sum()
    denom1 = (truth_anom**2 * combined_weights).sum()
    denom2 = (pred_anom**2 * combined_weights).sum()
    
    corr = num / np.sqrt(denom1 * denom2)
    
    return float(mae.values), float(corr.values), truth_zonal_time_mean, pred_zonal_time_mean

# Main execution
if __name__ == "__main__":
    # Load data
    data_file = "./data.zarr"
    path_to_pred = "3D_thermo_all_prediction.zarr"
    
    print("Loading ground truth data...")
    data = xr.open_zarr(data_file)
    ds_groundtruth = data.isel(time=slice(2, None))
    ds_groundtruth = convert_train_data(ds_groundtruth)
    
    print("Loading prediction data...")
    ds_prediction = xr.open_zarr(path_to_pred)
    # Adjust slice as needed to align with ground truth
    ds_prediction = ds_prediction.isel(time=slice(0, None))
    
    # Find overlapping time period
    truth_start = ds_groundtruth.time.values[0]
    truth_end = ds_groundtruth.time.values[-1]
    pred_start = ds_prediction.time.values[0]
    pred_end = ds_prediction.time.values[-1]
    
    print(f"Truth time range: {truth_start} to {truth_end}")
    print(f"Prediction time range: {pred_start} to {pred_end}")
    
    # Create output directory if needed
    output_dir = "zonal_mean_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute metrics
    print("Computing zonal mean metrics...")
    mae, corr, truth_zonal, pred_zonal = compute_zonal_mean_metrics(ds_groundtruth, ds_prediction, 'thetao')
    
    print(f"MAE for zonal mean temperature: {mae:.6f} °C")
    print(f"Pattern Correlation for zonal mean temperature: {corr:.6f}")