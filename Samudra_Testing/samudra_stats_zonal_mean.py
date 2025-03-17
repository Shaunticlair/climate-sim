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
    Compute metrics for zonal mean of ocean variables using the same approach
    as in the plotting code for Figure 1b in the Samudra paper.
    """
    # Get data
    truth = ds_truth[variable]
    pred = ds_pred[variable]
    
    # Print time dimensions
    print(f"Truth time dimension: {truth.time.size} steps")
    print(f"Prediction time dimension: {pred.time.size} steps")
    
    # Compute zonal mean exactly as in samudra_plotting.py
    # First, identify and apply land-ocean mask to avoid NaN issues
    section_mask = np.isnan(truth).all('x').isel(time=0)
    
    # Area-weighted zonal mean (longitude average) with time average
    truth_zonal = truth.weighted(ds_truth['areacello']).mean(['x', 'time'])
    pred_zonal = pred.weighted(ds_truth['areacello']).mean(['x', 'time'])
    
    # Apply mask to exclude land areas
    truth_zonal = truth_zonal.where(~section_mask)
    pred_zonal = pred_zonal.where(~section_mask)
    
    # Compute MAE without additional weighting, just on the masked fields
    # First, ensure both have the same NaN pattern for fair comparison
    combined_mask = np.isnan(truth_zonal) | np.isnan(pred_zonal)
    truth_masked = truth_zonal.where(~combined_mask)
    pred_masked = pred_zonal.where(~combined_mask)
    
    # Simple MAE calculation
    mae = np.abs(pred_masked - truth_masked).mean()
    
    # Pattern correlation
    # Remove mean for correlation
    truth_mean = truth_masked.mean()
    pred_mean = pred_masked.mean()
    
    truth_anom = truth_masked - truth_mean
    pred_anom = pred_masked - pred_mean
    
    # Compute correlation
    num = (truth_anom * pred_anom).sum()
    denom1 = (truth_anom**2).sum()
    denom2 = (pred_anom**2).sum()
    
    corr = num / np.sqrt(denom1 * denom2)
    
    return float(mae.values), float(corr.values), truth_zonal, pred_zonal

# Main execution
if __name__ == "__main__":
    # Load data
    data_file = "./data.zarr"
    path_to_thermo_pred = "3D_thermo_all_prediction.zarr"
    path_to_thermo_dynamic_pred = "3D_thermo_dynamic_all_prediction.zarr"
    
    print("Loading ground truth data...")
    data = xr.open_zarr(data_file)
    ds_groundtruth = convert_train_data(data)
    
    print("Loading prediction data...")
    ds_thermo = xr.open_zarr(path_to_thermo_pred)
    ds_thermo_dynamic = xr.open_zarr(path_to_thermo_dynamic_pred)
    
    
    # Compute metrics for temperature (thermo model)
    print("\nComputing zonal mean metrics for thermo model (temperature)...")
    mae_thermo, corr_thermo, truth_zonal, thermo_zonal = compute_zonal_mean_metrics(
        ds_groundtruth, ds_thermo, 'thetao')
    
    print(f"MAE for zonal mean temperature (thermo): {mae_thermo:.6f} °C")
    print(f"Pattern Correlation for zonal mean temperature (thermo): {corr_thermo:.6f}")
    
    # Compute metrics for temperature (thermo+dynamic model)
    print("\nComputing zonal mean metrics for thermo+dynamic model (temperature)...")
    mae_thermo_dynamic, corr_thermo_dynamic, _, thermo_dynamic_zonal = compute_zonal_mean_metrics(
        ds_groundtruth, ds_thermo_dynamic, 'thetao')
    
    print(f"MAE for zonal mean temperature (thermo+dynamic): {mae_thermo_dynamic:.6f} °C")
    print(f"Pattern Correlation for zonal mean temperature (thermo+dynamic): {corr_thermo_dynamic:.6f}")
    
    # Save results to CSV
    results = {
        'Model': ['Thermo', 'Thermo+Dynamic'],
        'Temperature_MAE': [mae_thermo, mae_thermo_dynamic],
        'Temperature_Correlation': [corr_thermo, corr_thermo_dynamic],
    }
    
    #df = pd.DataFrame(results)
    #df.to_csv(os.path.join(output_dir, 'zonal_mean_metrics.csv'), index=False)
    #print(f"\nResults saved to {os.path.join(output_dir, 'zonal_mean_metrics.csv')}")