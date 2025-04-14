#!/usr/bin/env python
# coding: utf-8

# Perturbation test for Samudra model using the new compute_fd_sensitivity function
# Tests a 5x5 grid centered around (180,90) for sensitivity analysis

import torch
import numpy as np
import sys
import setup

path = "/nobackup/sruiz5/SAMUDRATEST/Samudra/samudra/"  # Path to the Samudra package
sys.path.append(path)

import model_adjoint

def run_perturbation_test():
    """
    Main function to run the perturbation test using the SamudraAdjoint's compute_fd_sensitivity function
    """
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
    
    # Load the adjoint model
    model = model_adjoint.SamudraAdjoint(
        n_out=num_output_channels, 
        ch_width=[num_input_channels]+[200,250,300,400], 
        wet=wet.to(device), 
        hist=hist
    )
    model = setup.load_weights(model, state_out_vars_config, device)
    timer.checkpoint("Model loaded")
    
    # Define the center point 
    center_y, center_x = 90, 180
    channel = 0  # Using the first channel for tests
    initial_time = 0  # Source time
    final_time = 20    # Target time
    perturbation_magn = -2
    perturbation_size = 10 ** perturbation_magn
    
    # Target coords - the center point
    target_coords = [(0, channel, center_y, center_x, final_time)]
    
    # Create a 5x5 grid of source points centered on the target point
    grid_size = 5
    half_grid = grid_size // 2
    
    # Initialize source coordinates list
    source_coords_list = []
    
    print(f"Setting up 5x5 grid centered at {center_y, center_x}")
    
    # Build the grid of source coordinates
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate the coordinates for this grid position
            source_y = center_y - half_grid + i
            source_x = center_x - half_grid + j
            
            # Skip if coordinates are outside the valid range
            if (source_y < 0 or source_y >= test_data[0][0].shape[2] or 
                source_x < 0 or source_x >= test_data[0][0].shape[3]):
                continue
            
            # Add valid source coordinates
            source_coords_list.append((0, channel, source_y, source_x, initial_time))
    
    print(f"Created {len(source_coords_list)} source points in the grid")
    
    # Compute sensitivities using the new function
    print(f"Computing sensitivities with perturbation size: {perturbation_size}")
    sensitivity_matrix = model.compute_fd_sensitivity(
        test_data,
        source_coords_list=source_coords_list,
        target_coords_list=target_coords,
        perturbation_size=perturbation_size,
        device=device
    )
    
    # Reshape the results into a grid
    sensitivities = np.zeros((grid_size, grid_size))
    sensitivities.fill(np.nan)  # Fill with NaN for positions outside bounds
    
    for idx, source_coord in enumerate(source_coords_list):
        # Get the grid position from the source coordinates
        _, _, y, x, _ = source_coord
        grid_i = y - (center_y - half_grid)
        grid_j = x - (center_x - half_grid)
        
        # Store the sensitivity value
        sensitivities[grid_i, grid_j] = sensitivity_matrix[idx, 0].item()
        
        print(f"Sensitivity from ({y}, {x}) to target: {sensitivities[grid_i, grid_j]:.4f}")
    
    # Print the full grid
    print(f"\nSensitivity Grid ({grid_size}x{grid_size}):")
    for row in sensitivities:
        print(" ".join([f"{x:.4f}" if not np.isnan(x) else "N/A" for x in row]))
    
    # Save results
    np.save(f'perturb_sensitivity_grid_{grid_size}x{grid_size}_t={initial_time},{final_time}_1e{perturbation_magn}.npy', sensitivities)
    print(f"\nResults saved to perturb_sensitivity_grid_{grid_size}x{grid_size}_t={initial_time},{final_time}_1e{perturbation_magn}.npy")
    
    timer.checkpoint("Sensitivity tests completed")

if __name__ == "__main__":
    run_perturbation_test()
