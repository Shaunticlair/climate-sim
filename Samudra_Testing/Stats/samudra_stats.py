import xarray as xr
import numpy as np

from datetime import datetime
import sys

sys.path.append("/nobackup/sruiz5/SAMUDRATEST/Samudra/samudra/")

def compute_paper_metrics(ds_truth, ds_pred, variable='thetao'):
    # Get data
    truth = ds_truth[variable]
    pred = ds_pred[variable]
    
    # Print number of time steps in ds_groundtruth
    num_time_steps = ds_truth.time.size
    num_time_steps_pred = ds_pred.time.size
    print(f"Number of time steps: truth={num_time_steps}, prediction={num_time_steps_pred}")

    # Print the first 5 and last 5 time steps of ds_groundtruth
    print("First 5 time steps of ds_groundtruth:")
    print(ds_truth.time.values[:5])
    print("Last 5 time steps of ds_groundtruth:")
    print(ds_truth.time.values[-5:])
    print("First 5 time steps of ds_prediction:")
    print(ds_pred.time.values[:5])
    print("Last 5 time steps of ds_prediction:")
    print(ds_pred.time.values[-5:])
    
    # Volume weighting for spatial dimensions
    vol_weights = ds_truth['areacello'] * ds_truth['dz']
    vol_weights = vol_weights / vol_weights.sum()
    
    # MAE with volume weighting - compute over spatial dimensions first, then average over time
    diff = np.abs(pred - truth)
    
    # Compute weighted spatial MAE for each time step
    mae_time = (diff * vol_weights).sum(dim=['x', 'y', 'lev'])
    
    # Average over time
    mae = mae_time.mean(dim='time')
    
    # Pattern correlation - compute over spatial dimensions and then average correlation across time
    # Calculate spatial mean at each time step
    truth_spatial_mean = (truth * vol_weights).sum(dim=['x', 'y', 'lev'])
    pred_spatial_mean = (pred * vol_weights).sum(dim=['x', 'y', 'lev'])
    
    # Calculate anomalies at each time step
    truth_anom = truth.copy()
    pred_anom = pred.copy()
    
    # Loop through time to calculate anomalies
    correlations = []
    for t in range(len(truth.time)):
        truth_t = truth.isel(time=t)
        pred_t = pred.isel(time=t)
        
        truth_mean_t = truth_spatial_mean.isel(time=t)
        pred_mean_t = pred_spatial_mean.isel(time=t)
        
        truth_anom_t = truth_t - truth_mean_t
        pred_anom_t = pred_t - pred_mean_t
        
        num = (truth_anom_t * pred_anom_t * vol_weights).sum(dim=['x', 'y', 'lev'])
        denom1 = ((truth_anom_t**2) * vol_weights).sum(dim=['x', 'y', 'lev'])
        denom2 = ((pred_anom_t**2) * vol_weights).sum(dim=['x', 'y', 'lev'])
        
        # Avoid division by zero
        if denom1 == 0 or denom2 == 0:
            corr_t = np.nan
        else:
            corr_t = num / np.sqrt(denom1 * denom2)
        
        correlations.append(float(corr_t.values))
    
    # Average correlation over time
    corr = np.nanmean(correlations)
    
    return float(mae.values), corr

# Usage
data_file = "./data.zarr"
path_to_pred = "3D_thermo_all_prediction.zarr"
start_index = 2903
N_test = 600

# Load and process data
data = xr.open_zarr(data_file)
ds_groundtruth = data.isel(time=slice(2,None))
# Print time range
time_range = ds_groundtruth.time
print(f"Time range of ds_groundtruth: {time_range.values[0]} to {time_range.values[-1]}")

from utils import convert_train_data
ds_groundtruth = convert_train_data(ds_groundtruth)

ds_prediction = xr.open_zarr(path_to_pred)
ds_prediction = ds_prediction.isel(time=slice(1,None))

# Print time range for preds
time_range_pred = ds_prediction.time
print(f"Time range of ds_prediction: {time_range_pred.values[0]} to {time_range_pred.values[-1]}")

# Compute metrics
mae, corr = compute_paper_metrics(ds_groundtruth, ds_prediction, 'thetao')
print(f"MAE (temp): {mae:.6f} °C")
print(f"Pattern Correlation (temp): {corr:.6f}")