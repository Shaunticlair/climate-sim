import xarray as xr
import numpy as np
from datetime import datetime
import sys
import sys
#sys.path.append("../samudra/")
sys.path.append("/nobackup/sruiz5/SAMUDRATEST/Samudra/samudra/")


try:
    from utils import convert_train_data
except ImportError:
    print("Warning: Could not import convert_train_data from utils")
    # Fallback definition if needed
    def convert_train_data(ds):
        print("Using placeholder convert_train_data function")
        return ds

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
    
    # Extract the variable of interest and handle potential dimension differences
    gt = ground_truth[variable]
    pred = prediction[variable]
    
    print(f"Ground truth dims: {gt.dims}, shape: {gt.shape}")
    print(f"Prediction dims: {pred.dims}, shape: {pred.shape}")
    
    # For predictions, ensure 'lev' is a dimension, not a variable
    if 'lev' not in pred.dims and variable in ['thetao', 'so']:
        print("Restructuring prediction data to match expected format...")
        
        # Handle different dimension orderings
        if 'var' in pred.dims:
            # If data has 'var' dimension, select correct variables
            var_indices = {'thetao': slice(0, 19), 'so': slice(19, 38)}
            pred = pred.isel(var=var_indices[variable])
        
        # Transpose to match ground truth dimensions
        if len(pred.dims) == 3 and all(dim in pred.dims for dim in ['time', 'x', 'y']):
            # Missing lev dimension
            print("Prediction missing lev dimension. This isn't expected.")
        elif len(pred.dims) == 4:
            # Might need transpose
            correct_dims = ['time', 'lev', 'y', 'x'] if 'lev' in pred.dims else ['time', 'y', 'x', 'lev']
            target_dims = ['lev', 'time', 'y', 'x']  # Expected order
            
            if pred.dims != target_dims:
                try:
                    pred = pred.transpose(*correct_dims)
                    print(f"Transposed prediction to {pred.dims}")
                except ValueError:
                    print(f"Cannot transpose {pred.dims} to match {target_dims}")
    
    # Average over time for both datasets
    if 'time' in gt.dims and gt.time.size > 0:
        gt = gt.mean('time')
        print("Averaged ground truth over time")
    
    if 'time' in pred.dims:
        pred = pred.mean('time')
        print("Averaged prediction over time")
    
    # Compute profile mean (zonal mean), drop NaNs before computing
    print("Computing profiles...")
    gt_profile = profile_mean(gt.fillna(0))
    pred_profile = profile_mean(pred.fillna(0))
    
    print(f"Profile shapes - GT: {gt_profile.shape}, Pred: {pred_profile.shape}")
    
    # Make sure profiles have the same dimensions
    if gt_profile.dims != pred_profile.dims:
        print(f"Profile dimension mismatch: {gt_profile.dims} vs {pred_profile.dims}")
        # Try to align dimensions
        common_dims = set(gt_profile.dims) & set(pred_profile.dims)
        if common_dims:
            print(f"Using common dimensions: {common_dims}")
            # Create new arrays with only common dimensions
            gt_profile = gt_profile.mean([d for d in gt_profile.dims if d not in common_dims]) 
            pred_profile = pred_profile.mean([d for d in pred_profile.dims if d not in common_dims])
            print(f"New profile shapes - GT: {gt_profile.shape}, Pred: {pred_profile.shape}")
    
    # Check for complete NaN before computing
    if np.isnan(gt_profile).all().compute().item() or np.isnan(pred_profile).all().compute().item():
        print("WARNING: One or both profiles contain all NaNs")
        return np.nan, np.nan
    
    # Calculate MAE - compute for dask arrays, dropping NaNs
    diff = np.abs(pred_profile - gt_profile)
    print(f"Diff shape: {diff.shape}, has {np.isnan(diff).sum().compute().item()} NaNs out of {diff.size}")
    
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
    print(f"Starting Samudra model run at {datetime.now().strftime('%a %b %d %H:%M:%S %Y')}")
    
    # Paths
    path_to_thermo_pred = "3D_thermo_all_prediction.zarr"
    path_to_thermo_dynamic_pred = "3D_thermo_dynamic_all_prediction.zarr"
    data_file = "./data.zarr"
    
    # Start index and test length (match what's in the paper)
    start_index = 2903
    N_test = 600
    
    # Load data with chunking to avoid memory issues
    print("Loading ground truth data...")
    try:
        data = xr.open_zarr(data_file)
        ds_groundtruth = data
        ds_groundtruth = convert_train_data(ds_groundtruth)
    except Exception as e:
        print(f"Error loading ground truth data: {e}")
        ds_groundtruth = None
    
    # Load predictions with chunking
    print("Loading thermo prediction data...")
    try:
        thermo_pred = xr.open_zarr(path_to_thermo_pred, chunks={'time': 100})
    except Exception as e:
        print(f"Error loading thermo prediction data: {e}")
        thermo_pred = None
        
    print("Loading thermo+dynamic prediction data...")
    try:
        thermo_dynamic_pred = xr.open_zarr(path_to_thermo_dynamic_pred, chunks={'time': 100})
    except Exception as e:
        print(f"Error loading thermo+dynamic prediction data: {e}")
        thermo_dynamic_pred = None
    
    # Check if we have necessary data
    if ds_groundtruth is None:
        print("ERROR: Could not load ground truth data")
        sys.exit(1)
    
    if thermo_pred is None and thermo_dynamic_pred is None:
        print("ERROR: Could not load any prediction data")
        sys.exit(1)
    
    # Debug info about the datasets
    print("\nDataset information:")
    print(f"Ground truth shape: {ds_groundtruth.dims}")
    
    if thermo_pred is not None:
        print(f"Thermo prediction shape: {thermo_pred.dims}")
    
    if thermo_dynamic_pred is not None:
        print(f"Thermo+dynamic prediction shape: {thermo_dynamic_pred.dims}")
    
    # Compute metrics for temperature profiles
    print("\nComputing metrics for temperature (thetao):")
    if thermo_pred is not None:
        try:
            mae_thermo, corr_thermo = compute_metrics(ds_groundtruth, thermo_pred, 'thetao')
            print(f"Thermo MAE: {mae_thermo:.6f} °C")
            print(f"Thermo Pattern Correlation: {corr_thermo:.6f}")
        except Exception as e:
            print(f"Error computing thermo temperature metrics: {e}")
    
    if thermo_dynamic_pred is not None:
        try:
            mae_thermo_dynamic, corr_thermo_dynamic = compute_metrics(ds_groundtruth, thermo_dynamic_pred, 'thetao')
            print(f"Thermo+Dynamic MAE: {mae_thermo_dynamic:.6f} °C")
            print(f"Thermo+Dynamic Pattern Correlation: {corr_thermo_dynamic:.6f}")
        except Exception as e:
            print(f"Error computing thermo+dynamic temperature metrics: {e}")
    
    # Compute metrics for salinity profiles
    print("\nComputing metrics for salinity (so):")
    if thermo_pred is not None:
        try:
            mae_thermo_s, corr_thermo_s = compute_metrics(ds_groundtruth, thermo_pred, 'so')
            print(f"Thermo MAE: {mae_thermo_s:.6f} psu")
            print(f"Thermo Pattern Correlation: {corr_thermo_s:.6f}")
        except Exception as e:
            print(f"Error computing thermo salinity metrics: {e}")
    
    if thermo_dynamic_pred is not None:
        try:
            mae_thermo_dynamic_s, corr_thermo_dynamic_s = compute_metrics(ds_groundtruth, thermo_dynamic_pred, 'so')
            print(f"Thermo+Dynamic MAE: {mae_thermo_dynamic_s:.6f} psu")
            print(f"Thermo+Dynamic Pattern Correlation: {corr_thermo_dynamic_s:.6f}")
        except Exception as e:
            print(f"Error computing thermo+dynamic salinity metrics: {e}")
    
    print(f"\nFinished Samudra model run at {datetime.now().strftime('%a %b %d %H:%M:%S %Y')}")
