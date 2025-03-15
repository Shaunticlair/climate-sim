import xarray as xr
import numpy as np

from datetime import datetime
import sys

sys.path.append("/nobackup/sruiz5/SAMUDRATEST/Samudra/samudra/")

def compute_paper_metrics(ds_truth, ds_pred, variable='thetao'):
    # Get data
    truth = ds_truth[variable]
    pred = ds_pred[variable]
    
    # Make sure we have the same time points
    #common_times = sorted(list(set(truth.time.values) & set(pred.time.values)))
    #truth = truth.sel(time=common_times)
    #pred = pred.sel(time=common_times)

    # Print number of time steps in ds_groundtruth
    num_time_steps = ds_truth.time.size
    num_time_steps_pred = ds_pred.time.size
    print(num_time_steps,num_time_steps_pred)

    # Print the first 5 and last 5 time steps of ds_groundtruth
    print("First 5 time steps of ds_groundtruth:")
    print(ds_truth.time.values[:5])
    print("Last 5 time steps of ds_groundtruth:")
    print(ds_truth.time.values[-5:])
    print("First 5 time steps of ds_prediction:")
    print(ds_pred.time.values[:5])
    print("Last 5 time steps of ds_prediction:")
    print(ds_pred.time.values[-5:])
    


    # Volume weighting
    vol_weights = ds_truth['areacello'] * ds_truth['dz']
    vol_weights = vol_weights / vol_weights.sum()
    
    # MAE with volume weighting
    diff = np.abs(pred - truth)
    mae = (diff * vol_weights).sum(dim=['x', 'y', 'lev', 'time'])
    
    # Pattern correlation
    truth_mean = (truth * vol_weights).sum(dim=['x', 'y', 'lev', 'time'])
    pred_mean = (pred * vol_weights).sum(dim=['x', 'y', 'lev', 'time'])
    
    truth_anom = truth - truth_mean
    pred_anom = pred - pred_mean
    
    num = (truth_anom * pred_anom * vol_weights).sum(dim=['x', 'y', 'lev', 'time'])
    denom1 = ((truth_anom**2) * vol_weights).sum(dim=['x', 'y', 'lev', 'time'])
    denom2 = ((pred_anom**2) * vol_weights).sum(dim=['x', 'y', 'lev', 'time'])
    
    corr = num / np.sqrt(denom1 * denom2)
    
    return float(mae.values), float(corr.values)

# Usage
data_file = "./data.zarr"
path_to_pred = "3D_thermo_all_prediction.zarr"
start_index = 2903
N_test = 600

# Load and process data
data = xr.open_zarr(data_file)
ds_groundtruth = data.isel(time=slice(2,None))#.isel(time=slice(start_index, start_index+N_test))
#Print time range
time_range = ds_groundtruth.time
print(f"Time range of ds_groundtruth: {time_range.values[0]} to {time_range.values[-1]}")


from utils import convert_train_data
ds_groundtruth = convert_train_data(ds_groundtruth)

ds_prediction = xr.open_zarr(path_to_pred)

# Print time range for preds
time_range_pred = ds_prediction.time
print(f"Time range of ds_prediction: {time_range_pred.values[0]} to {time_range_pred.values[-1]}")

# Compute metrics
mae, corr = compute_paper_metrics(ds_groundtruth, ds_prediction, 'thetao')
print(f"MAE (temp): {mae:.6f} °C")
print(f"Pattern Correlation (temp): {corr:.6f}")
