import xarray as xr
import sys
from utils import convert_train_data  # Import the conversion function

# Paths
path_to_thermo_pred = "3D_thermo_all_prediction.zarr"
path_to_thermo_dynamic_pred = "3D_thermo_dynamic_all_prediction.zarr"
data_file = "./data.zarr"

# Start index and test length
start_index = 2903
N_test = 600

# Load ground truth and convert
print("Loading and converting ground truth...")
try:
    data = xr.open_zarr(data_file)
    ds_groundtruth = data.isel(time=slice(start_index, start_index+N_test))
    ds_groundtruth = convert_train_data(ds_groundtruth)
    print("Ground truth converted successfully")
    print(f"Ground truth variables after conversion: {list(ds_groundtruth.variables)}")
except Exception as e:
    print(f"Error loading/converting ground truth: {e}")
    sys.exit(1)

# Load predictions
print("\nLoading predictions...")
try:
    thermo_pred = xr.open_zarr(path_to_thermo_pred)
    print(f"Thermo prediction variables: {list(thermo_pred.variables)}")
    
    # Check dimensions match
    print("\nDimension comparison:")
    for var in ['thetao', 'so', 'zos']:
        if var in thermo_pred and var in ds_groundtruth:
            gt_dims = ds_groundtruth[var].dims
            pred_dims = thermo_pred[var].dims
            print(f"{var} - Ground truth: {gt_dims}, Prediction: {pred_dims}")
            
            # Check dimensions
            if gt_dims != pred_dims:
                print(f"  WARNING: Dimension mismatch for {var}")
                
                # Check for lev in both
                if 'lev' in gt_dims and 'lev' in pred_dims:
                    gt_levs = ds_groundtruth.lev.values
                    pred_levs = thermo_pred.lev.values
                    if not (gt_levs == pred_levs).all():
                        print(f"  Level values differ: GT {gt_levs[:3]}..., Pred {pred_levs[:3]}...")
except Exception as e:
    print(f"Error comparing datasets: {e}")
