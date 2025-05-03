# This script sets up the environment for running Samudra.

import time
import random
import sys
import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

# Add the Samudra package to the path
path = "path/to/samudra"  # Replace with the actual path to the Samudra package
path = "/nobackup/sruiz5/SAMUDRATEST/Samudra/samudra/"
sys.path.append(path)

import utils  
import data_loaders
from einops import rearrange

### Constants and configurations

# Depth levels
DEPTH_LEVELS = ['2_5', '10_0', '22_5', '40_0', '65_0', '105_0', '165_0', 
                '250_0', '375_0', '550_0', '775_0', '1050_0', '1400_0', 
                '1850_0', '2400_0', '3100_0', '4000_0', '5000_0', '6000_0']

VARS = {
    "3D_thermo_dynamic_all": ['uo', 'vo', 'thetao', 'so', 'zos'],
    "3D_thermo_all": ['thetao', 'so', 'zos'],
}
# Define input and output variables
INPUT_VARS_LEV = {
    "3D_thermo_dynamic_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ] + ["zos"],
    "3D_thermo_all": [
        k + str(j)
        for k in ["thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ] + ["zos"]
}

BOUNDARY_VARS = {
    "3D_all_hfds_anom": ["tauuo", "tauvo", "hfds", "hfds_anomalies"]
}

OUTPUT_VARS_LEV = {
    "3D_thermo_dynamic_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ] + ["zos"],
    "3D_thermo_all": [
        k + str(j)
        for k in ["thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ] + ["zos"]
}

MODEL_PATHS = {
    "3D_thermo_dynamic_all": "samudra_thermo_dynamic_seed1.pt",
    "3D_thermo_all": "samudra_thermo_seed1.pt",
}

N_samples = 2850  # Used for train
N_val = 50        # Used for validation
N_test = 600       # Number of time steps to use for testing

### Setup functions

class Timer:
    """Used for time-tracking."""
    def __init__(self):
        self.start_time = time.time()
        self.last_checkpoint = self.start_time
        
    def checkpoint(self, section_name):
        current_time = time.time()
        section_duration = current_time - self.last_checkpoint
        total_duration = current_time - self.start_time
        print(f"Section '{section_name}' took {section_duration:.4f} seconds")
        print(f"Total elapsed time: {total_duration:.4f} seconds")
        print("-" * 40)
        self.last_checkpoint = current_time

class NullTimer(Timer):
    """Used for time-tracking."""
    def checkpoint(self, section_name):
        pass

def torch_config_cuda_cpu_seed():
    """
    Configure the environment for CUDA or CPU, and set random seeds for reproducibility.
    
    Returns:
    --------
    device : torch.device
        The device being used (CPU or CUDA)."""
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        import os
        num_cores = os.cpu_count()
        torch.set_num_threads(num_cores)
        print(f"Using {num_cores} CPU cores for computation")

    # Set seeds for reproducibility
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    return device

def choose_model(state_in_vars_config="3D_thermo_dynamic_all", boundary_vars_config="3D_all_hfds_anom",
                 hist=1):
    """
    Choose the model based on the experiment number.
    
    Parameters:
    --------
    state_in_vars_config : str
        Configuration for input state variables.
    boundary_vars_config : str
        Configuration for boundary variables.
    hist : int
        History length.

    Returns:
    --------
    list_list_str : list of list of str
        List of input, boundary, output, and variable lists.

        input_list_str : list of str
            List of input variables.
        boundary_list_str : list of str
            List of boundary variables.
        output_list_str : list of str
            List of output variables.
        vars_list_str : list of str
            List of variables.
    
    list_num_channels : list of int
        List of number of input and output channels.

        num_input_channels : int
            Number of input channels.
        num_output_channels : int
            Number of output channels.

    """

    state_out_vars_config = state_in_vars_config

    input_list_str =    INPUT_VARS_LEV[state_in_vars_config]
    boundary_list_str = BOUNDARY_VARS[boundary_vars_config]
    output_list_str =   OUTPUT_VARS_LEV [state_out_vars_config]
    vars_list_str =     VARS[state_out_vars_config]

    num_vars_in = len(input_list_str)
    num_vars_boundary = len(boundary_list_str)  # Number of atmosphere variables
    num_vars_out = len(output_list_str)

    num_input_channels = int((hist + 1) * num_vars_in + num_vars_boundary) 
    num_output_channels = int((hist + 1) * num_vars_out)

    print("Number of input channels:", num_input_channels)
    print("Number of output channels:", num_output_channels)

    list_list_str = [input_list_str, boundary_list_str, output_list_str, vars_list_str]
    list_num_channels = [num_input_channels, num_output_channels]

    return list_list_str, list_num_channels

def compute_indices(hist, N_samples=2850, N_val=50, N_test=600):
    s_train = hist
    e_train = s_train + N_samples
    s_test = e_train + N_val
    e_test = s_test + hist + 1 + N_test
    return s_train, e_train, s_test, e_test


# ## Load Data
# Load the necessary data for the model

def load_data_raw(start_idx, end_idx, output_list_str, suffix= '',
                     hist=1):
    """
    Load data, mean, and std from zarr files for testing.

    Parameters:
    --------
    - start_idx: start index for time samples
    - end_idx: end index for time samples
    - output_list_str: list of output variables for model
    - suffix: suffix for the zarr files (default: empty string)
    - hist: history length
    """
    from pathlib import Path

    # Replace these local zarr paths if required
    data_mean_file = "./data_mean" + suffix + ".zarr"
    data_std_file = "./data_std" + suffix + ".zarr"
    data_file = "./data" + suffix + ".zarr"

    if not Path(data_mean_file).exists():
        data_mean = xr.open_dataset("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4_means", engine='zarr', chunks={})
        data_mean.to_zarr(data_mean_file, mode="w")
        
    data_mean = xr.open_zarr("./data_mean.zarr")

    if not Path(data_std_file).exists():
        data_std = xr.open_dataset("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4_stds", engine='zarr', chunks={})
        data_std.to_zarr(data_std_file, mode="w")
    data_std = xr.open_zarr(data_std_file)

    if not Path(data_file).exists():
        data = xr.open_dataset("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4", engine='zarr', chunks={})
        data = data.isel(time=slice(start_idx, end_idx)) # We only require the data in the test period
        data.to_zarr(data_file, mode="w")
    data = xr.open_zarr(data_file)

    # Get the wetmask for the model
    
    wet_zarr = data.wetmask
    #print(output_list_str, 'zos' in output_list_str)
    
    # Remove duplicates from output_list_str
    output_list_str = list(set(output_list_str))  # Remove duplicates
    wet = utils.extract_wet(wet_zarr, output_list_str, hist)

    return data, wet, data_mean, data_std

def load_data(s_test, e_test, N_test,
                   input_list_str, boundary_list_str, output_list_str,
                   hist=1, device="cuda"):
    """
    Load data, mean, and std from zarr files for testing. Process data into Test object.

    Parameters:
    --------
    - s_test: start index for test samples
    - e_test: end index for test samples
    - outputs_str: list of output variables for model
    - hist: history length
    """
    data, wet, data_mean, data_std = load_data_raw(
        s_test, e_test, output_list_str,
        hist=hist
    )

    test_data = data_loaders.Test(
        data,
        input_list_str, 
        boundary_list_str, 
        output_list_str,
        wet,
        data_mean,
        data_std,
        N_test,
        hist,
        0,
        long_rollout=False,  # Setting to False for sensitivity analysis
        device=device,
    )


    return test_data, wet, data_mean, data_std

def load_weights(model, state_out_vars_config, 
                 device='cuda', param_grad_tracking = False):
    """
    Load the model weights from the specified path.
    
    Parameters:
    --------
    - model: the model to load weights into
    - exp_num_out: experiment number for the model
    - device: device to load the model on (default: 'cuda')
    - param_grad_tracking: whether to track gradients for model parameters (default: False)
    """
    
    model_path = MODEL_PATHS[state_out_vars_config]
    weights_path = Path(model_path)

    if not weights_path.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="M2LInES/Samudra", 
            filename=weights_path.name,
            local_dir="."
        )
    model_state = torch.load(weights_path, map_location=torch.device(device))["model"]
    model.load_state_dict(model_state)
    model = model.to(device)

    if param_grad_tracking:
        for param in model.parameters(): # Turn off gradient tracking for all model parameters to improve performance
            param.requires_grad_(False)

    return model


### This section is used to compute spatial correlations between a reference variable and field variables

def load_data_for_correlation_analysis(reference_point=(90, 180), 
                                       spatial_slice=(slice(50, 130), slice(110, 250)),
                                       reference_var='zos', 
                                       field_vars=['hfds', 'tauuo', 'tauvo', 'hfds_anomalies'],
                                       reference_depth='2_5',  # Default surface level
                                       time_window=600,  
                                       device='cuda'):
    """
    Load and preprocess the data for spatial correlation analysis.
    
    This function loads the dataset with specific time window for analyzing spatial 
    correlations between a reference variable at a fixed point and field variables
    across a spatial domain with various time lags.
    
    Parameters:
    -----------
    reference_point : tuple, optional
        (lat, lon) coordinates of the reference point, default (90, 180)
    spatial_slice : tuple, optional
        (lat_slice, lon_slice) defining the spatial region for correlation analysis
    reference_var : str, optional
        Name of the reference variable, default 'zos'
    field_vars : list, optional
        List of field variables to correlate with
    reference_depth : str, optional
        Depth level for the reference variable if applicable, default '2_5'
    time_window : int, optional
        Number of time steps to include (default: 73, approximately 1 year)
    device : str, optional
        Device to load data on ('cpu' or 'cuda')
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'reference_series': Time series of the reference variable at the reference point
        - 'field_series': Dictionary of time series for each field variable
        - 'wetmask': Binary mask for water/land points
        - 'data_info': Additional metadata (time coordinates, normalization factors, etc.)
    """
    # Calculate time indices
    
    # We have the last 600 timesteps of training to use for analysis
    analysis_start = 600 - time_window
    analysis_end = 600
    
    print(f"Loading data from time index {analysis_start} to {analysis_end}")
    
    # Construct a list of variables we need
    output_list_str = []
    if reference_var in ['thetao', 'so', 'uo', 'vo']:
        output_list_str.append(f"{reference_var}_lev_{reference_depth}")
    else:
        output_list_str.append(reference_var)
    
    for field_var in field_vars:
        if field_var in ['thetao', 'so', 'uo', 'vo']:
            output_list_str.append(f"{field_var}_lev_{reference_depth}")
        else:
            output_list_str.append(field_var)
    
    print("Set up")
    # Load data using existing function with "analysis" suffix to avoid overwriting
    data, wet, data_mean, data_std = load_data_raw( #If this is the first time, we load data from end of training
        2850-600, # 600 timesteps before the end of training data
        2850+2, # This is the last time step in the training data
        output_list_str,
        suffix="_analysis",
        hist=1
    )
    print("Finished loading")

    # Slice into the specific time window for analysis
    data = data.isel(time=slice(analysis_start, analysis_end))

    print("Finished slicing")
    
    # Extract reference variable data
    reference_data = {}
    ref_lat, ref_lon = reference_point
    
    # Handle reference variable with depth if needed
    if reference_var in ['thetao', 'so', 'uo', 'vo']:
        # Variable with depth levels
        ref_var_name = f"{reference_var}_lev_{reference_depth}"
    else:
        # Variable without depth (e.g., zos)
        ref_var_name = reference_var
    
    # Extract the raw and normalized reference data
    reference_raw = data[ref_var_name].sel(y=ref_lat, x=ref_lon, method='nearest')
    
    print(f"Available dimensions in data_mean: {list(data_mean.dims)}")
    print(f"Available coordinates in data_mean: {list(data_mean.coords)}")

    print("data")
    print(data)
    print("data_mean")
    print(data_mean)
    print("data_std")
    print(data_std)
    # The data is already normalized when loaded through load_data_raw
    # However, we need to use mean and std for potential future use
    reference_mean = data_mean[ref_var_name]
    reference_std = data_std[ref_var_name]
    
    # Store reference data
    reference_data['raw'] = reference_raw.compute()
    reference_data['mean'] = reference_mean.compute()
    reference_data['std'] = reference_std.compute()
    
    # Extract field variables data for the spatial slice
    field_data = {}
    lat_slice, lon_slice = spatial_slice
    
    for field_var in field_vars:
        field_data[field_var] = {}
        
        # Handle field variable with depth if needed
        if field_var in ['thetao', 'so', 'uo', 'vo']:
            # Variable with depth levels
            field_var_name = f"{field_var}_lev_{reference_depth}"
        else:
            # Variable without depth
            field_var_name = field_var
        
        # Extract the raw field data
        field_raw = data[field_var_name].sel(y=lat_slice, x=lon_slice)
        
        # Get mean and std for reference
        field_mean = data_mean[field_var_name]
        field_std = data_std[field_var_name]
        
        # Store field data
        field_data[field_var]['raw'] = field_raw.compute()
        field_data[field_var]['mean'] = field_mean.compute()
        field_data[field_var]['std'] = field_std.compute()
    
    # Get wetmask for the spatial region
    wetmask = data.wetmask.sel(y=lat_slice, x=lon_slice)
    
    # Get appropriate depth level for the wetmask
    if reference_var in ['thetao', 'so', 'uo', 'vo']:
        # Find the depth level index that corresponds to reference_depth
        depth_idx = list(data.lev.values).index(float(reference_depth.replace('_', '.')))
        wetmask = wetmask.isel(lev=depth_idx).compute()
    else:
        # For surface variables like zos, use the surface wetmask
        wetmask = wetmask.isel(lev=0).compute()
    
    # Organize data for return
    result = {
        'reference_series': reference_data,
        'field_series': field_data,
        'wetmask': wetmask,
        'data_info': {
            'time': data.time.values,
            'reference_point': reference_point,
            'spatial_slice': spatial_slice,
            'reference_var': reference_var,
            'field_vars': field_vars,
            'time_window': time_window
        }
    }
    
    return result

### This section is used to create a weighted loss function for the model

def compute_cell_thickness(depth_levels):
    """
    Compute the thickness of each ocean cell based on the center depths.
    
    Parameters:
    -----------
    depth_levels : list of str
        List of depth level strings like '2_5', '10_0', etc.
    
    Returns:
    --------
    dict
        Dictionary mapping each depth level to its thickness in meters
    """
    # Convert depth level strings to float values
    depths = [float(level.replace('_', '.')) for level in depth_levels]
    depths.sort()  # Ensure depths are in ascending order
    
    # Initialize thickness dictionary
    thickness = {}
    
    # Special case for the first (shallowest) layer
    # Assuming thickness extends from surface to midpoint between first and second depth
    if len(depths) > 1:
        thickness[str(depths[0]).replace('.', '_')] = (depths[0] + (depths[1] - depths[0]) / 2)
    else:
        thickness[str(depths[0]).replace('.', '_')] = depths[0] * 2  # Arbitrary if only one depth
    
    # Calculate thickness for middle layers
    # Each cell extends from midpoint between previous depth to midpoint between next depth
    for i in range(1, len(depths) - 1):
        half_dist_prev = (depths[i] - depths[i-1]) / 2
        half_dist_next = (depths[i+1] - depths[i]) / 2
        thickness[str(depths[i]).replace('.', '_')] = half_dist_prev + half_dist_next
    
    # Special case for the last (deepest) layer
    # Assuming thickness extends from midpoint between last two depths to twice the distance to the bottom
    if len(depths) > 1:
        last_idx = len(depths) - 1
        half_dist_prev = (depths[last_idx] - depths[last_idx-1]) / 2
        # For the deepest layer, we assume it extends as far below as the half-distance to the
        # previous level (this is a common approximation)
        thickness[str(depths[last_idx]).replace('.', '_')] = half_dist_prev * 2
    
    return thickness

def compute_zos_weights(areacello, wetmask_2d):
    """
    Compute the normalized weights for zos (surface height) variable.
    
    Parameters:
    -----------
    areacello : torch.Tensor or numpy array
        Area of each ocean cell (2D array with dimensions [lat, lon])
    wetmask_2d : torch.Tensor or numpy array
        Binary mask indicating ocean cells (1) vs land (0)
        For zos, this should be the surface level wetmask
    
    Returns:
    --------
    torch.Tensor
        Normalized weights for zos (2D array with dimensions [lat, lon])
    """
    import torch

    # Convert xarray arrays to numpy arrays
    if isinstance(areacello, xr.DataArray):
        areacello = areacello.values
    if isinstance(wetmask_2d, xr.DataArray):
        wetmask_2d = wetmask_2d.values
    
    # Convert to torch tensors if they are not already
    if not isinstance(areacello, torch.Tensor):
        areacello = torch.tensor(areacello, dtype=torch.float32)
    if not isinstance(wetmask, torch.Tensor):
        wetmask = torch.tensor(wetmask, dtype=torch.float32)
    
    # Apply wetmask to the cell area
    masked_area = areacello * wetmask_2d
    
    # Calculate the total area of wet cells
    total_area = masked_area.sum()
    
    # Normalize to get weights that sum to 1
    weights = masked_area / total_area
    
    return weights

def compute_3d_weights(depth_widths, areacello, wetmask_3d):
    """
    Compute the normalized weights for a 3D variable (with depth dimension).
    
    Parameters:
    -----------
    depth_widths : list or array
        Width/thickness of each depth level
    areacello : torch.Tensor or numpy array
        Area of each ocean cell (2D array with dimensions [lat, lon])
    wetmask_3d : torch.Tensor or numpy array
        3D binary mask indicating ocean cells (1) vs land (0)
        Shape should be [depth, lat, lon]
    
    Returns:
    --------
    torch.Tensor
        Normalized weights for the 3D variable with shape [depth, lat, lon]
    """
    import torch
    import numpy as np

    # Turn any xarray objects into numpy arrays
    if isinstance(areacello, xr.DataArray):
        areacello = areacello.values
    if isinstance(wetmask_3d, xr.DataArray):
        wetmask_3d = wetmask_3d.values
    if isinstance(depth_widths, xr.DataArray):
        depth_widths = depth_widths.values
    
    # Convert to torch tensors if they are not already
    if not isinstance(areacello, torch.Tensor):
        areacello = torch.tensor(areacello, dtype=torch.float32)
    if not isinstance(wetmask_3d, torch.Tensor):
        wetmask_3d = torch.tensor(wetmask_3d, dtype=torch.float32)
    if not isinstance(depth_widths, torch.Tensor):
        depth_widths = torch.tensor(depth_widths, dtype=torch.float32)
    
    # Ensure we have the right number of depth levels
    n_depths = len(depth_widths)
    wetmask_sliced = wetmask_3d[:n_depths, :, :]
    
    # Reshape areacello to allow broadcasting across depth dimension
    # Shape: [1, lat, lon]
    areacello_broadcast = areacello.unsqueeze(0)
    
    # Apply wetmask to the cell area
    # Shape: [depth, lat, lon]
    masked_area = wetmask_sliced * areacello_broadcast
    
    # Reshape depth_widths to allow broadcasting across lat and lon dimensions
    # Shape: [depth, 1, 1]
    depth_widths_broadcast = depth_widths.reshape(-1, 1, 1)
    
    # Multiply by depth width to get cell volumes
    # Shape: [depth, lat, lon]
    cell_volumes = masked_area * depth_widths_broadcast
    
    # Calculate total volume of all wet cells
    total_volume = cell_volumes.sum()
    
    # Normalize to get weights that sum to 1
    weights = cell_volumes / total_volume
    
    return weights

def create_full_weights_tensor(variables, depth_widths, areacello, wetmask_3d):
    """
    Create a full weights tensor for all variables in the state array.
    
    Parameters:
    -----------
    variables : list
        List of variable names in the state array (can include duplicates)
    depth_widths : dict
        Dictionary mapping depth level strings to their thickness values
    areacello : torch.Tensor
        Area of each ocean cell (2D array with dimensions [lat, lon])
    wetmask_3d : torch.Tensor
        3D binary mask indicating ocean cells (1) vs land (0)
        Shape should be [depth, lat, lon]
    
    Returns:
    --------
    torch.Tensor
        Full weights tensor with shape [num_channels, lat, lon]
    """
    import torch
    
    # Convert depth_widths dictionary to a list in the correct order
    depth_levels = sorted([k for k in depth_widths.keys()], 
                          key=lambda x: float(x.replace('_', '.')))
    depth_width_values = [depth_widths[level] for level in depth_levels]
    depth_width_tensor = torch.tensor(depth_width_values, dtype=torch.float32)
    
    # Compute the 2D weights for surface variables
    zos_weights = compute_zos_weights(areacello, wetmask_3d[0])  # Surface level wetmask
    
    # Compute the 3D weights for depth-dependent variables
    depth_weights = compute_3d_weights(depth_width_tensor, areacello, wetmask_3d)
    
    # List to store weights for each variable
    all_weights = []
    
    # Process each variable
    for var in variables:
        if var == 'zos':
            # For zos, use the 2D weights
            all_weights.append(zos_weights.unsqueeze(0))  # Add channel dimension
        elif var in ['uo', 'vo', 'thetao', 'so']:
            all_weights.append(depth_weights)
    
    # Stack all weights along the channel dimension
    full_weights = torch.cat(all_weights, dim=0)
    
    return full_weights

def gen_weighted_loss_fn(data, state_in_vars_config="3D_thermo_dynamic_all"):
    variables = VARS[state_in_vars_config]
    variables = variables + variables # Duplicate for two timesteps 

    weight_matrix = create_full_weights_tensor(
        variables,  # Example variables
        compute_cell_thickness(DEPTH_LEVELS),  # Thickness values
        data.areacello,  # Area of each ocean cell
        data.wetmask  # 3D wetmask
    )

    print("Weight matrix shape:", weight_matrix.shape)

    def weighted_loss_fn(prediction, targets):
        return torch.mean(
            (prediction - targets) ** 2 * weight_matrix.to(prediction.device)
        )
    
    return weighted_loss_fn



# ## Data 
# The data is available from 1975 to the 2022, at 5-day temporal resolution. The variables in the data is arranged in the following format:
# 
# THIS ORDERING IS WRONG :(
# ```
# thetao_lev_2_5
# thetao_lev_10_0
# thetao_lev_22_5
# ...
# thetao_lev_6000_0
# 
# so_lev_2_5
# so_lev_10_0
# so_lev_22_5
# ...
# so_lev_6000_0
# 
# uo_lev_2_5
# uo_lev_10_0
# uo_lev_22_5
# ...
# uo_lev_6000_0
# 
# vo_lev_2_5
# vo_lev_10_0
# vo_lev_22_5
# ...
# vo_lev_6000_0
# 
# zos
# 
# hfds
# hfds_anomalies
# tauuo
# tauvo
# ```