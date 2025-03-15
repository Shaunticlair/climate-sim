import xarray as xr
import numpy as np
from datetime import datetime
import sys

sys.path.append("/nobackup/sruiz5/SAMUDRATEST/Samudra/samudra/")

from utils import convert_train_data

def profile_mean(ds):
    """Calculate weighted mean across spatial dimensions, ignoring NaNs"""
    return ds.weighted(ds.areacello).mean(["x", "y"], skipna=True)
        
def compute_metrics(ground_truth, prediction, variable='thetao'):
    """Compute MAE and pattern correlation between ground truth and prediction"""
    gt = ground_truth[variable]
    pred = prediction[variable]
    
    if 'lev' not in pred.dims and variable in ['thetao', 'so']:
        if 'var' in pred.dims:
            var_indices = {'thetao': slice(0, 19), 'so': slice(19, 38)}
            pred = pred.isel(var=var_indices[variable])
        
        if len(pred.dims) == 4:
            correct_dims = ['time', 'lev', 'y', 'x'] if 'lev' in pred.dims else ['time', 'y', 'x', 'lev']
            if pred.dims != ['lev', 'time', 'y', 'x']:
                pred = pred.transpose(*correct_dims)
    
    if 'time' in gt.dims and gt.time.size > 0:
        gt = gt.mean('time')
    
    if 'time' in pred.dims:
        pred = pred.mean('time')
    
    # Proper handling of NaNs in weighted mean
    gt_profile = profile_mean(gt)  # Remove fillna(0)
    pred_profile = profile_mean(pred)  # Remove fillna(0)
    
    if gt_profile.dims != pred_profile.dims:
        common_dims = set(gt_profile.dims) & set(pred_profile.dims)
        if common_dims:
            gt_profile = gt_profile.mean([d for d in gt_profile.dims if d not in common_dims]) 
            pred_profile = pred_profile.mean([d for d in pred_profile.dims if d not in common_dims])
    
    if np.isnan(gt_profile).all().compute().item() or np.isnan(pred_profile).all().compute().item():
        return np.nan, np.nan
    
    # Calculate MAE only on non-NaN values in both datasets
    diff = np.abs(pred_profile - gt_profile)
    mae_array = diff.mean(skipna=True)  # skipna=True ensures NaNs are ignored
    mae = float(mae_array.compute().values)
    
    gt_profile_computed = gt_profile.compute()
    pred_profile_computed = pred_profile.compute()
    
    gt_flat = gt_profile_computed.values.flatten()
    pred_flat = pred_profile_computed.values.flatten()
    
    mask = ~np.isnan(gt_flat) & ~np.isnan(pred_flat)
    gt_flat = gt_flat[mask]
    pred_flat = pred_flat[mask]
    
    if len(gt_flat) < 2:
        return mae, np.nan
    
    correlation = np.corrcoef(gt_flat, pred_flat)[0, 1]
    return mae, correlation

if __name__ == "__main__":
    path_to_thermo_pred = "3D_thermo_all_prediction.zarr"
    path_to_thermo_dynamic_pred = "3D_thermo_dynamic_all_prediction.zarr"
    data_file = "./data.zarr"
    
    start_index = 2903
    N_test = 600

    data = xr.open_zarr(data_file)
    ds_groundtruth = convert_train_data(data)
    
    thermo_pred = xr.open_zarr(path_to_thermo_pred)
    thermo_dynamic_pred = xr.open_zarr(path_to_thermo_dynamic_pred)
    
    if thermo_pred is not None:
        mae_thermo, corr_thermo = compute_metrics(ds_groundtruth, thermo_pred, 'thetao')
        print(f"Thermo MAE (temp): {mae_thermo:.6f} °C")
        print(f"Thermo Pattern Correlation (temp): {corr_thermo:.6f}")
    
    if thermo_dynamic_pred is not None:
        mae_thermo_dynamic, corr_thermo_dynamic = compute_metrics(ds_groundtruth, thermo_dynamic_pred, 'thetao')
        print(f"Thermo+Dynamic MAE (temp): {mae_thermo_dynamic:.6f} °C")
        print(f"Thermo+Dynamic Pattern Correlation (temp): {corr_thermo_dynamic:.6f}")
    
    if thermo_pred is not None:
        mae_thermo_s, corr_thermo_s = compute_metrics(ds_groundtruth, thermo_pred, 'so')
        print(f"Thermo MAE (salinity): {mae_thermo_s:.6f} psu")
        print(f"Thermo Pattern Correlation (salinity): {corr_thermo_s:.6f}")
    
    if thermo_dynamic_pred is not None:
        mae_thermo_dynamic_s, corr_thermo_dynamic_s = compute_metrics(ds_groundtruth, thermo_dynamic_pred, 'so')
        print(f"Thermo+Dynamic MAE (salinity): {mae_thermo_dynamic_s:.6f} psu")
        print(f"Thermo+Dynamic Pattern Correlation (salinity): {corr_thermo_dynamic_s:.6f}")