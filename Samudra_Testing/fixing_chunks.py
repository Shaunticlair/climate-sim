#!/usr/bin/env python
# coding: utf-8
import numpy as np

initial_channel_index = 0 
initial_time = 0        # Starting time step
final_time =   4          # Ending time step 

lats = np.arange(0, 2)
lons = np.arange(0, 3)

initial_indices = []
c = initial_channel_index
for h in lats:
    for w in lons:
        initial_indices.append([0, c, h, w])

def convert_indices_to_time_indices(indices, t):
    out = []
    first_half = t % 2 == 0 # First half of state vector contains even timestep indices
    for b, c, h, w in indices:
        if c > 157: # Only 158 channels
            raise ValueError("Channel index out of bounds")
        # Channels 154-157 are boundary conditions shared across timesteps: no need to adjust them
        if c < 154: 
            if first_half and c > 76: # Need to adjust for the first half of the state vector
                c = c - 77
            if not first_half and c < 77: # Need to adjust for the second half of the state vector
                c = c + 77
        out.append((b, c, h, w, t))
    return out

times = [i for i in range(initial_time, final_time+1)]

in_indices = [convert_indices_to_time_indices(initial_indices, t) for t in times]
in_indices = [item for sublist in in_indices for item in sublist]  # Flatten the list

in_indices = [str(tuple(int(j) for j in i)) for i in in_indices]  # Convert to integers if needed

print(in_indices)

in_indices = np.array(in_indices)

num_times = len(times)
# Reshape
reshaped_in_indices = in_indices.reshape( len(times),len(lats),len(lons) )  # Reshape to 2D array with 4 columns
print(reshaped_in_indices.shape)
print(reshaped_in_indices)

matrices = [ reshaped_in_indices[t,:,:] for t in range(0, 4) ]  # Reshape to 2D array with 4 columns

print(matrices)

raise Exception("STOP HERE")

print(reshaped_in_indices)

"""
Fix Sensitivity Matrices

This script loads individual sensitivity matrices that were incorrectly shaped,
recombines them, reshapes them correctly, and saves them as separate files again.
"""

import numpy as np
from pathlib import Path

# Parameters
final_time = 10  # The final time used in all matrices
initial_times = list(range(10))  # Initial times (0-9)
lat_size = 180  # Number of latitude points
lon_size = 360  # Number of longitude points

def main():
    # Step 1: Load all matrices
    matrices = []
    for t in initial_times:
        file_path = f'adjoint_sensitivity_matrix_t={t},{final_time}.npy'
        if not Path(file_path).exists():
            print(f"Warning: File {file_path} not found, skipping.")
            continue
        
        matrix = np.load(file_path)
        print(f"Loaded matrix for t={t} with shape {matrix.shape}")
        matrices.append(matrix)
    
    # Step 2: Combine all matrices into one big array
    combined = np.stack(matrices, axis=2)  # This will have shape (lat_size, lon_size, num_times)
    print(f"Combined matrix shape: {combined.shape}")
    
    # Step 3: If needed, reshape to flatten and then reshape correctly
    # This step might be unnecessary if the matrices were just individually stored with correct shapes
    # But including it for completeness
    flattened = combined.reshape(-1)  # Flatten completely
    correct_shape = (len(initial_times), lat_size, lon_size)
    reshaped = flattened.reshape(correct_shape)  # Reshape with time as first dimension
    print(f"Reshaped matrix shape: {reshaped.shape}")
    
    # Step 4: Save individual time slices with correct shape
    for t_idx, t in enumerate(initial_times):
        output_file = f'fixed_adjoint_sensitivity_matrix_t={t},{final_time}.npy'
        sensitivity_slice = reshaped[t_idx]  # Extract the time slice
        
        # Check if output file exists and remove if necessary
        output_path = Path(output_file)
        if output_path.exists():
            print(f"Removing existing file: {output_file}")
            output_path.unlink()
        
        # Save the correctly shaped time slice
        np.save(output_file, sensitivity_slice)
        print(f"Saved fixed matrix for t={t} to {output_file}")

if __name__ == "__main__":
    main()