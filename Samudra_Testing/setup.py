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

N_samples = 2850  # Used for train
N_val = 50        # Used for validation
N_test = 20       # Number of time steps to use for testing
def compute_indices(hist, N_samples=2850, N_val=50, N_test=20):
    s_train = hist
    e_train = s_train + N_samples
    s_test = e_train + N_val
    e_test = s_test + hist + 1 + N_test
    return s_train, e_train, s_test, e_test


# ## Load Data
# Load the necessary data for the model

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
    from pathlib import Path

    # Replace these local zarr paths if required
    data_mean_file = "./data_mean.zarr"
    data_std_file = "./data_std.zarr"
    data_file = "./data.zarr"

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
        data = data.isel(time=slice(s_test, e_test)) # We only require the data in the test period
        data.to_zarr(data_file, mode="w")
    data = xr.open_zarr(data_file)

    # Get the wetmask for the model
    
    wet_zarr = data.wetmask
    wet = utils.extract_wet(wet_zarr, output_list_str, hist)

    class VisibleTest(data_loaders.Test):
        def __getitem__(self, idx):

            print("Input axes:", list(self.inputs.dims))
            print("Input vars:", list(self.inputs.data_vars))
            print("Inputs no extra", list(self.inputs_no_extra.data_vars))
                
            if type(idx) == slice:
                if idx.start == None and idx.stop == None:
                    idx = slice(0, self.size, idx.step)
                elif idx.start == None:
                    idx = slice(0, idx.stop, idx.step)
                elif idx.stop == None:
                    idx = slice(idx.start, self.size, idx.step)
            elif type(idx) == int:
                idx = slice(idx, idx + 1, 1)

            rolling_idx = self.rolling_indices.isel(window_dim=idx)
            x_index = xr.Variable(
                ["window_dim", "time"], rolling_idx
            )
            print("Out: ", (self.ind_start + x_index.isel(time=slice(self.hist + 1, None))).values, end=' ')
            data_in = self.inputs_no_extra.isel(time=x_index).isel(
                time=slice(None, self.hist + 1)
            )
            data_in = (
                (data_in - self.inputs_no_extra_mean) / self.inputs_no_extra_std
            ).fillna(0)
            print("data_in:", list(data_in.data_vars))
            print("data_in axes:", list(data_in.dims))
            shaunticlair_temp_array = data_in.to_array().transpose("window_dim", "time", "variable", "y", "x")
            print("data_in to_array:", list(shaunticlair_temp_array.coords["variable"].values))
            print("data_in shape:", shaunticlair_temp_array.shape)

            shaunticlair_transposed_array = rearrange(
                shaunticlair_temp_array, "window_dim time variable y x -> window_dim (time variable) y x"
            )
            print(shaunticlair_transposed_array)
            print("data_in time variable", shaunticlair_transposed_array.coords['time variable'])
            raise Exception
            data_in = (
                data_in.to_array()
                .transpose("window_dim", "time", "variable", "y", "x")
                .to_numpy()
            )
            data_in = rearrange(
                data_in, "window_dim time variable y x -> window_dim (time variable) y x"
            )
            print()
            if len(self.extras.variables) != 0:
                data_in_boundary = self.extras.isel(time=x_index).isel(time=self.hist)
                data_in_boundary = (
                    (data_in_boundary - self.extras_mean) / self.extras_std
                ).fillna(0)
                data_in_boundary = (
                    data_in_boundary.to_array()
                    .transpose("window_dim", "variable", "y", "x")
                    .to_numpy()
                )
                data_in = np.concatenate((data_in, data_in_boundary), axis=1)

            label = self.outputs.isel(time=x_index).isel(time=slice(self.hist + 1, None))
            label = ((label - self.out_mean) / self.out_std).fillna(0)
            label = (
                label.to_array()
                .transpose("window_dim", "time", "variable", "y", "x")
                .to_numpy()
            )
            label = rearrange(
                label, "window_dim time variable y x -> window_dim (time variable) y x"
            )

            items = (torch.from_numpy(data_in).float(), torch.from_numpy(label).float())

            return items

    test_data = VisibleTest(
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

    test_data[0] # run __getitem__ 


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


# ## Data 
# The data is available from 1975 to the 2022, at 5-day temporal resolution. The variables in the data is arranged in the following format:
# 
# 
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
