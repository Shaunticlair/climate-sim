import torch
import numpy as np
import xarray as xr
import sys
import time
from pathlib import Path

# Add Samudra to path
sys.path.append("/nobackup/sruiz5/SAMUDRATEST/Samudra/samudra/")

# Import Samudra modules
from model import Samudra
from data_loaders import Test
from utils import extract_wet, post_processor, convert_train_data
from adjoint_method import SamudraAdjoint

# Timer class for performance monitoring
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

def main():
    # Start timer
    timer = Timer()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    timer.checkpoint("Setup complete")
    
    # Model configuration (same as in samudra_rollout.py)
    DEPTH_LEVELS = ['2_5', '10_0', '22_5', '40_0', '65_0', '105_0', '165_0', '250_0', 
                     '375_0', '550_0', '775_0', '1050_0', '1400_0', '1850_0', '2400_0', 
                     '3100_0', '4000_0', '5000_0', '6000_0']
    
    # Define input/output variable sets
    INPT_VARS = {
        "3D_thermo_dynamic_all": [
            k + str(j) for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
            for j in DEPTH_LEVELS
        ] + ["zos"],
        "3D_thermo_all": [
            k + str(j) for k in ["thetao_lev_", "so_lev_"]
            for j in DEPTH_LEVELS
        ] + ["zos"]
    }
    EXTRA_VARS = {
        "3D_all_hfds_anom": ["tauuo", "tauvo", "hfds", "hfds_anomalies"]
    }
    OUT_VARS = {
        "3D_thermo_dynamic_all": [
            k + str(j) for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
            for j in DEPTH_LEVELS
        ] + ["zos"],
        "3D_thermo_all": [
            k + str(j) for k in ["thetao_lev_", "so_lev_"]
            for j in DEPTH_LEVELS
        ] + ["zos"]
    }
    VARS = {
        "3D_thermo_dynamic_all": ['uo', 'vo', 'thetao', 'so', 'zos'],
        "3D_thermo_all": ['thetao', 'so', 'zos'],
    }
    MODEL_PATHS = {
        "3D_thermo_dynamic_all": "samudra_thermo_dynamic_seed1.pt",
        "3D_thermo_all": "samudra_thermo_seed1.pt",
    }
    
    # Choose model type
    exp_num_in = "3D_thermo_all"  # Choose between "3D_thermo_all" or "3D_thermo_dynamic_all"
    exp_num_extra = "3D_all_hfds_anom"
    exp_num_out = exp_num_in
    model_path = MODEL_PATHS[exp_num_out]
    
    # Data configuration
    hist = 1
    N_test = 4  # Number of timesteps to simulate
    
    # Variable configurations
    inputs_str = INPT_VARS[exp_num_in]
    extra_in_str = EXTRA_VARS[exp_num_extra]
    outputs_str = OUT_VARS[exp_num_out]
    var_ls = VARS[exp_num_out]
    
    N_atm = len(extra_in_str)
    N_in = len(inputs_str)
    N_extra = N_atm
    N_out = len(outputs_str)
    
    num_in = int((hist + 1) * N_in + N_extra)
    num_out = int((hist + 1) * len(outputs_str))
    
    print(f"Number of inputs: {num_in}")
    print(f"Number of outputs: {num_out}")
    
    timer.checkpoint("Configuration setup")
    
    # Load data
    data_mean_file = "./data_mean.zarr"
    data_std_file = "./data_std.zarr"
    data_file = "./data.zarr"
    
    # Load means and standard deviations
    data_mean = xr.open_zarr(data_mean_file)
    data_std = xr.open_zarr(data_std_file)
    
    # Load data
    data = xr.open_zarr(data_file)
    print("Data loaded successfully")
    
    timer.checkpoint("Data loaded")
    
    # Extract wet mask (used for masking ocean/land)
    wet_zarr = data.wetmask
    wet = extract_wet(wet_zarr, outputs_str, hist)
    print("Wet mask extracted, shape:", wet.shape)
    
    # Create data loader for observations
    test_data = Test(
        data,
        inputs_str,
        extra_in_str,
        outputs_str,
        wet,
        data_mean,
        data_std,
        N_test + 1,  # +1 for initial state
        hist,
        0,
        long_rollout=True,
        device=device,
    )
    
    timer.checkpoint("Test data loader created")
    
    # Build the model
    model = Samudra(n_out=num_out, ch_width=[num_in]+[200,250,300,400], wet=wet.to(device), hist=hist)
    
    # Load the model weights
    weights_path = Path(model_path)
    model.load_state_dict(torch.load(weights_path, map_location=device)["model"])
    model = model.to(device)
    print("Model loaded successfully")
    
    timer.checkpoint("Model built and loaded")
    
    # Create the adjoint method instance
    adjoint = SamudraAdjoint(model, device=device)
    
    # Get observations from the data loader
    observations = []
    for i in range(N_test + 1):  # +1 for initial state
        obs = test_data[i][1][0]  # Get the labels which are our observations
        observations.append(obs)
    
    # Get the initial state from the first observation
    initial_state = test_data[0][0][0]  # Get the first input state
    
    # Compute the gradient of the loss with respect to the initial state
    print("Computing adjoint...")
    initial_gradient, simulated_states, total_loss = adjoint.compute_initial_state_gradient(
        initial_state, observations
    )
    
    timer.checkpoint("Adjoint method completed")
    
    # Process and save the results
    print(f"Total loss: {total_loss.item()}")
    
    # Extract some statistics about the gradient
    gradient_norm = torch.norm(initial_gradient).item()
    gradient_mean = torch.mean(initial_gradient).item()
    gradient_std = torch.std(initial_gradient).item()
    
    print(f"Gradient norm: {gradient_norm}")
    print(f"Gradient mean: {gradient_mean}")
    print(f"Gradient standard deviation: {gradient_std}")
    
    # Save the gradient to a file
    gradient_numpy = initial_gradient.cpu().numpy()
    np.save("samudra_adjoint_gradient.npy", gradient_numpy)
    
    # Save the simulated states for comparison with observations
    simulated_numpy = [state.detach().cpu().numpy() for state in simulated_states]
    np.save("samudra_simulated_states.npy", simulated_numpy)
    
    print("Gradient and simulated states saved to disk")
    
    # Optionally, calculate sensitivity to particular regions of interest
    # For example, to find sensitivity to surface temperature in the tropics:
    if "thetao_lev_2_5" in inputs_str:
        # Find the index of surface temperature in the input state
        surface_temp_index = inputs_str.index("thetao_lev_2_5")
        
        # Extract gradient for surface temperature
        surface_temp_gradient = initial_gradient[:, surface_temp_index, :, :]
        
        # Calculate mean sensitivity in the tropical region (roughly -20 to 20 latitude)
        tropical_indices = (initial_state.shape[2] // 2 - initial_state.shape[2] // 9, 
                            initial_state.shape[2] // 2 + initial_state.shape[2] // 9)
        
        tropical_sensitivity = torch.mean(torch.abs(
            surface_temp_gradient[:, :, tropical_indices[0]:tropical_indices[1]]
        )).item()
        
        print(f"Mean sensitivity to tropical surface temperature: {tropical_sensitivity}")
    
    timer.checkpoint("Analysis completed")

if __name__ == "__main__":
    main()