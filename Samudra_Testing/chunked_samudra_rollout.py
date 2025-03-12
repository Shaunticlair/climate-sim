#!/usr/bin/env python
# coding: utf-8

# # Samudra
# 
# This notebook will walk you through how to construct the model, load the weights, build the dataset, and use the model to generate a rollout.

# ## Imports

N_test = 20
chunk_size = 10

import time

class Timer:
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

# Start the clock
timer = Timer()

timer.checkpoint("Starting...")

# In[1]:


import sys
#sys.path.append("../samudra/")
sys.path.append("/nobackup/sruiz5/SAMUDRATEST/Samudra/samudra/")


# In[2]:


import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download

timer.checkpoint("Imports")

# ## Configs

# We now configure the backends and torch states, including setting the seeds for the RNGs.

# In[3]:


if torch.cuda.is_available():
    print(f"Using device: {torch.cuda.get_device_name()}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.manual_seed(42)
np.random.seed(42)



# In[4]:


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

timer.checkpoint("Torch config")

# #### Experiment Constants

# In[5]:


# Experiment inputs and outputs
DEPTH_LEVELS = ['2_5',
 '10_0',
 '22_5',
 '40_0',
 '65_0',
 '105_0',
 '165_0',
 '250_0',
 '375_0',
 '550_0',
 '775_0',
 '1050_0',
 '1400_0',
 '1850_0',
 '2400_0',
 '3100_0',
 '4000_0',
 '5000_0',
 '6000_0']

INPT_VARS = {
    "3D_thermo_dynamic_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ]
    + ["zos"],
    "3D_thermo_all": [
        k + str(j)
        for k in ["thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ]
    + ["zos"]
}
EXTRA_VARS = {
    "3D_all_hfds_anom": ["tauuo", "tauvo", "hfds", "hfds_anomalies"]
}
OUT_VARS = {
    "3D_thermo_dynamic_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ]
    + ["zos"],
    "3D_thermo_all": [
        k + str(j)
        for k in ["thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ]
    + ["zos"]
}
VARS = {
    "3D_thermo_dynamic_all": ['uo', 'vo', 'thetao', 'so', 'zos'],
    "3D_thermo_all": ['thetao', 'so', 'zos'],
}
MODEL_PATHS = {
    "3D_thermo_dynamic_all": "samudra_thermo_dynamic_seed1.pt",
    "3D_thermo_all": "samudra_thermo_seed1.pt",
}


# ### Choose the model!
# 
# Set exp_num_in to "3D_thermo_all" or "3D_thermo_dynamic_all" corresponding to thermo or the thermo+dynamic model referenced in the paper.

# In[6]:


exp_num_in = "3D_thermo_dynamic_all" # "3D_thermo_all" or "3D_thermo_dynamic_all"
exp_num_extra = "3D_all_hfds_anom"
exp_num_out = exp_num_in

model_path = MODEL_PATHS[exp_num_out]


# ### Data Configs
# 
# We will follow the paper's configuration to choose the data's configuration

# In[7]:


hist = 1
N_samples = 2850 # Used for train
N_val = 50 # Used for validation
#N_test = 4 # Used for testing
# Should take 1 min/2 N_test


# In[8]:


inputs_str = INPT_VARS[exp_num_in]
extra_in_str = EXTRA_VARS[exp_num_extra]
outputs_str = OUT_VARS[exp_num_out]
var_ls = VARS[exp_num_out]

levels = len(DEPTH_LEVELS)

N_atm = len(extra_in_str)  # Number of atmosphere variables
N_in = len(inputs_str)
N_extra = N_atm  # Number of atmosphere variables
N_out = len(outputs_str)

num_in = int((hist + 1) * N_in + N_extra)
num_out = int((hist + 1) * len(outputs_str))

print("Number of inputs: ", num_in)
print("Number of outputs: ", num_out) 

# Getting start and end indices of train and test
s_train = hist
e_train = s_train + N_samples
e_test = e_train + N_val  

timer.checkpoint("Set constants")
# 
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
# 
# 

# In[9]:


import xarray as xr


# Here, we must take a choice between downloading the data locally (generally the fastest method of generating rollout) or streaming the data from a remote source. 

# #### Choice 1: Downloading data locally (Preferred)
# Save the data locally! Storage required ~ 12GB

# In[10]:


from pathlib import Path

# Replace these local zarr paths if required
data_mean_file = "./data_mean.zarr"
data_std_file = "./data_std.zarr"
data_file = "./data.zarr"

if not Path(data_mean_file).exists():
    raise Exception("This file should exist already!")
    data_mean = xr.open_dataset("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4_means", engine='zarr', chunks={})
    data_mean.to_zarr(data_mean_file, mode="w")
    
data_mean = xr.open_zarr("./data_mean.zarr")

if not Path(data_std_file).exists():
    data_std = xr.open_dataset("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4_stds", engine='zarr', chunks={})
    data_std.to_zarr(data_std_file, mode="w")
data_std = xr.open_zarr(data_std_file)

if not Path(data_file).exists():
    data = xr.open_dataset("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4", engine='zarr', chunks={})
    data = data.isel(time=slice(e_test, e_test+hist+1+N_test)) # We only require the data in the test period
    data.to_zarr(data_file, mode="w")
data = xr.open_zarr(data_file)
data


# #### Choice 2: Streaming data
# Run this cell to stream the data from a remote source. Run this cell if storage is a constraint (slower rollout generation)

# In[11]:


# data_mean = xr.open_dataset("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4_stds", engine='zarr', chunks={})
# data_std = xr.open_dataset("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4_means", engine='zarr', chunks={})
# data = xr.open_dataset("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4", engine='zarr', chunks={})
# data = data.isel(time=slice(e_test, e_test+N_test)) # We only require the data in the test period
# data


# ### Dataloader
# The dataloader requires the actual data along with precomputed means and standard deviations for each variable. The dataloader will normalize the data using the means and standard deviations.

# In[12]:

timer.checkpoint("Data retrieved")

# We require the wet mask to pass to the data loader
from utils import extract_wet
wet_zarr = data.wetmask
wet = extract_wet(wet_zarr, outputs_str, hist)
print("Wet resolution:", wet.shape)


# In[13]:


from data_loaders import Test

"""
test_data = Test(
    data,
    inputs_str,
    extra_in_str,
    outputs_str,
    wet,
    data_mean,
    data_std,
    N_test,
    hist,
    0,
    long_rollout=True,
    device=device,
)
"""

timer.checkpoint("Dataloader run")

# ## Build Model
# We are now ready to build the mdoel.

# In[14]:


from model import Samudra

# num_in and num_out depends on the input/output variables [thermo / thermo+dynamic].
model = Samudra(n_out=num_out, ch_width=[num_in]+[200,250,300,400], wet=wet.to(device), hist=hist)

timer.checkpoint("Model create")
# ### Load weights

# In[15]:


weights_path = Path(model_path)
if not weights_path.exists(): #Modified to avoid duplication, hopefully
    hf_hub_download(
        repo_id="M2LInES/Samudra",
        filename=weights_path.name,
        local_dir=".",
    )    

timer.checkpoint("Weights retrieved")

model.load_state_dict(torch.load(weights_path, map_location=torch.device(device))["model"])
model = model.to(device)

timer.checkpoint("Weights loaded to model")

# ## Rollout

# In[16]:


from model import generate_model_rollout


# You will notice the rollout starts from index '2' as we are using the initial states at indices '0' and '1' as input.

# In[17]:


"""
#get_ipython().run_cell_magic('time', '', 'model_pred, _ = generate_model_rollout(\n            N_test,\n            test_data,\n            model,\n            hist,\n            N_out,\n            N_extra,\n            initial_input=None, \n        )')
# Used for timer nonsense. Gonna instead just run the code:
model_pred, _ = generate_model_rollout(
    N_test,
    test_data,
    model,
    hist,
    N_out,
    N_extra,
    initial_input=None, 
)

timer.checkpoint("Generate forward rollout")

# #### Convert the prediction and ground truth data to the correct format useful for plotting and comparison

# In[18]:


from utils import post_processor, convert_train_data
ds_prediction = xr.DataArray(
            data=model_pred,
            dims=["time", "x", "y", "var"]
        )

ds_prediction = ds_prediction.to_dataset(name="predictions")
ds_groundtruth = data.isel(time=slice(hist+1,hist+1+N_test))
ds_prediction = post_processor(ds_prediction, ds_groundtruth, var_ls)
ds_groundtruth = convert_train_data(ds_groundtruth)

timer.checkpoint("Data post-processing")
# ## Example Plot

# In[19]:


#import matplotlib.pyplot as plt
#import cmocean as cm
#surface_temp_bias = ds_prediction.thetao.isel(time=0, lev=0)
#surface_temp_bias.rename(r"Surface temperature at step 0 [$\degree C$]").plot(vmin=0, vmax=30, cmap=cm.cm.balance)
#plt.title("Surface Temperature")
#plt.show()


# ## Save rollout

# In[20]:


ds_prediction.to_zarr(exp_num_in + '_prediction.zarr', mode='w')
timer.checkpoint("Predictions made")
"""
# In[ ]:

# Replace the existing rollout code section with this

from utils import post_processor, convert_train_data

# Define chunk size

total_steps = N_test  # Original total steps (600)
num_chunks = (total_steps + chunk_size - 1) // chunk_size  # Ceiling division

print(f"Processing {total_steps} timesteps in {num_chunks} chunks of {chunk_size}")

# Variable to store the last output from previous chunk
last_chunk_output = None

for chunk_idx in range(num_chunks):
    # Calculate current chunk's start and size
    chunk_start = chunk_idx * chunk_size
    current_chunk_size = min(chunk_size, total_steps - chunk_idx * chunk_size)
    
    print(f"\nProcessing chunk {chunk_idx+1}/{num_chunks}: timesteps {chunk_idx*chunk_size} to {chunk_idx*chunk_size + current_chunk_size - 1}")
    
    """
    test_data = Test(
        data,
        inputs_str,
        extra_in_str,
        outputs_str,
        wet,
        data_mean,
        data_std,
        N_test,
        hist,
        0,
        long_rollout=True,
        device=device,
    )
    """
    # Create a new dataloader for this chunk
    chunk_test_data = Test(
        data,
        inputs_str,
        extra_in_str,
        outputs_str,
        wet,
        data_mean,
        data_std,
        current_chunk_size,
        hist,
        chunk_start,  # Adjust start index
        long_rollout=True,
        device=device,
    )
    
    timer.checkpoint(f"Dataloader created for chunk {chunk_idx+1}")
    
    # Generate rollout for this chunk
    chunk_model_pred, chunk_outputs = generate_model_rollout(
        current_chunk_size,
        chunk_test_data,
        model,
        hist,
        N_out,
        N_extra,
        initial_input=last_chunk_output,
    )

    # Store the last output for the next chunk
    last_chunk_output = chunk_outputs[-1]
    
    timer.checkpoint(f"Rollout generated for chunk {chunk_idx+1}")
    
    # Convert the prediction to the correct format for this chunk
    ds_chunk_prediction = xr.DataArray(
        data=chunk_model_pred,
        dims=["time", "x", "y", "var"]
    )
    ds_chunk_prediction = ds_chunk_prediction.to_dataset(name="predictions")
    
    # Get the corresponding ground truth for this chunk
    ds_chunk_groundtruth = data.isel(time=slice(chunk_start + hist + 1, 
                                                chunk_start + hist + 1 + current_chunk_size))
    
    # Post-process the chunk prediction
    ds_chunk_prediction = post_processor(ds_chunk_prediction, ds_chunk_groundtruth, var_ls)
    
    # Save this chunk's rollout
    chunk_filename = f"{exp_num_in}_prediction_chunk{chunk_idx+1}.zarr"
    print(f"Saving chunk to {chunk_filename}")
    ds_chunk_prediction.to_zarr(chunk_filename, mode='w')
    
    timer.checkpoint(f"Chunk {chunk_idx+1} saved")

print("\nAll chunks processed successfully!")



