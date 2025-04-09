#!/usr/bin/env python
# coding: utf-8

# Perturbation test for Samudra model using finite difference approximation
# Modified to compute sensitivities in a 5x5 grid centered around (180,90)

import torch
import numpy as np
import sys
import setup

path ="/path/to/samudra/" # Replace with the actual path to the Samudra package
path = "/nobackup/sruiz5/SAMUDRATEST/Samudra/samudra/"
sys.path.append(path)

from model import Samudra

def simple_sensitivity(model, data_input, source_coords, target_coords, perturbation_size=1e-4):
    """
    Computes sensitivity of target cell to a perturbation in source cell.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to evaluate sensitivity for
    data_input : torch.Tensor
        Input tensor to the model (b, c, h, w)
    source_coords : tuple
        (channel, y, x) coordinates of source cell to perturb
    target_coords : tuple
        (channel, y, x) coordinates of target cell to measure
    perturbation_size : float
        Size of perturbation to apply
        
    Returns:
    --------
    sensitivity : float
        Sensitivity value between source and target cells
    """
    model.eval()
    
    # Make sure we don't modify the original input
    data_input = data_input.clone().detach()
    
    # Unpack coordinates
    s_c, s_y, s_x = source_coords
    t_c, t_y, t_x = target_coords
    
    # Step 1: Run model on unperturbed input
    with torch.no_grad():
        baseline_output = model.forward_once(data_input)
        baseline_value = baseline_output[0, t_c, t_y, t_x].item()
    
    # Step 2: Perturb the input at source coordinates
    perturbed_input = data_input.clone()
    perturbed_input[0, s_c, s_y, s_x] += perturbation_size
    
    # Step 3: Run model on perturbed input
    with torch.no_grad():
        perturbed_output = model.forward_once(perturbed_input)
        perturbed_value = perturbed_output[0, t_c, t_y, t_x].item()
    
    # Step 4: Calculate sensitivity
    sensitivity = (perturbed_value - baseline_value) / perturbation_size
    
    return sensitivity

if __name__ == "__main__":
    # Create a timer for tracking performance
    timer = setup.Timer()
    
    # Configure the environment
    device = setup.torch_config_cuda_cpu_seed()
    timer.checkpoint("Environment configured")
    
    # Set up model configuration
    hist = 1
    N_test = 10
    state_in_vars_config = "3D_thermo_all"  # Could also be "3D_thermo_dynamic_all"
    state_out_vars_config = state_in_vars_config
    boundary_vars_config = "3D_all_hfds_anom"
    
    # Get variable lists and channel counts
    list_list_str, list_num_channels = setup.choose_model(
        state_in_vars_config, 
        boundary_vars_config, 
        hist
    )
    input_list_str, boundary_list_str, output_list_str, vars_list_str = list_list_str
    num_input_channels, num_output_channels = list_num_channels
    timer.checkpoint("Model configuration set up")
    
    # Get data indices
    s_train, e_train, s_test, e_test = setup.compute_indices(
        hist=hist, N_samples=2850, N_val=50, N_test=10
    )
    
    # Load the data
    test_data, wet, data_mean, data_std = setup.load_data(s_test, e_test, N_test,
        input_list_str, boundary_list_str, output_list_str,
        hist=hist, device=device
    )
    timer.checkpoint("Data loaded")
    
    # Load the model
    model = Samudra(
        n_out=num_output_channels, 
        ch_width=[num_input_channels]+[200,250,300,400], 
        wet=wet.to(device), 
        hist=hist
    )
    model = setup.load_weights(model, state_out_vars_config, device)
    timer.checkpoint("Model loaded")
    
    # Get input data for testing
    input_data = test_data[0][0].to(device)
    
    # Define the center point 
    center_y, center_x = 90, 180
    channel = 0  # Using the first channel for tests
    
    # Target coords - the center point
    target_coords = (channel, center_y, center_x)
    
    # Create a 5x5 grid of source points centered on the target point
    grid_size = 5
    half_grid = grid_size // 2
    
    # Initialize sensitivity array
    sensitivities = np.zeros((grid_size, grid_size))
    
    print(f"Computing sensitivities for 5x5 grid centered at {center_y, center_x}")
    
    # Compute sensitivity for each point in the 5x5 grid
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate the coordinates for this grid position
            source_y = center_y - half_grid + i
            source_x = center_x - half_grid + j
            
            # Skip if coordinates are outside the valid range
            if (source_y < 0 or source_y >= input_data.shape[2] or 
                source_x < 0 or source_x >= input_data.shape[3]):
                sensitivities[i, j] = np.nan
                continue
            
            # Compute sensitivity
            source_coords = (channel, source_y, source_x)
            sens = simple_sensitivity(model, input_data, source_coords, target_coords)
            sensitivities[i, j] = sens
            
            print(f"Sensitivity from ({source_y}, {source_x}) to {target_coords}: {sens}")
    
    # Print the full grid
    print("\nSensitivity Grid (5x5):")
    for row in sensitivities:
        print(" ".join([f"{x:.4f}" if not np.isnan(x) else "N/A" for x in row]))
    
    # Save results
    np.save("sensitivity_grid_5x5.npy", sensitivities)
    print("\nResults saved to sensitivity_grid_5x5.npy")
    
    timer.checkpoint("Sensitivity tests completed")