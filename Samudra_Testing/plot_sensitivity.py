import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Path to the sensitivity matrix file
path = 'sensitivity_matrix.npy'

# Load the sensitivity matrix
sensitivity_matrix = np.load(path)
print(f"Original sensitivity matrix shape: {sensitivity_matrix.shape}")

sensitivity_latitude = 90  # Center latitude for the matrix (assuming 180x360 grid)
sensitivity_longitude = 180  # Center longitude for the matrix (assuming 180x360 grid)
t0,t1 = 0,2
"""
# Path to the wetmask file
wetmask_path = 'full_wetmask.npy'



# Load the wetmask (if it exists)
if Path(wetmask_path).exists():
    wetmask = np.load(wetmask_path)
    print(f"Loaded wetmask with shape: {wetmask.shape}")
    has_wetmask = True
else:
    print(f"Warning: Wetmask file {wetmask_path} not found. Proceeding without masking land areas.")
    has_wetmask = False

"""

drymask_path = 'drymask.npy'

# Turn into wetmask
if Path(drymask_path).exists():
    drymask = np.load(drymask_path)
    print(f"Loaded drymask with shape: {drymask.shape}")
    # Invert the drymask to get the wetmask (wet=0, dry=1)
    wetmask = np.where(drymask == 0, 1, 0)  # Invert the mask
    has_wetmask = True
else:
    print(f"Warning: Drymask file {drymask_path} not found. Proceeding without masking land areas.")
    has_wetmask = False
    
# Reshape to 180x360 if needed (assuming it represents lat/lon grid)
sensitivity_matrix = sensitivity_matrix.reshape(180, 360)
print(f"Reshaped sensitivity matrix: {sensitivity_matrix.shape}")

size = "local"

if size == "tiny":
    # Define the region of interest in the matrix for cropping
    xmin, xmax = 90-3, 90+4  # Matrix row indices
    ymin, ymax = 180-3, 180+4  # Matrix column indices

if size == "local":
    xmin, xmax = 90-20, 90+20  # Matrix row indices
    ymin, ymax = 180-20, 180+20  # Matrix column indices

if size == "global":
    xmin, xmax = 0, 180
    ymin, ymax = 0, 360


# Crop the sensitivity matrix to the region of interest
cropped_sensitivity = sensitivity_matrix[xmin:xmax, ymin:ymax]
print(f"Cropped sensitivity matrix: {cropped_sensitivity.shape}")

# Apply the wetmask if available
if has_wetmask:
    # Crop the wetmask to match the sensitivity matrix
    cropped_wetmask = wetmask[xmin:xmax, ymin:ymax]
    
    # Create a masked array where land (wet=0) is masked
    masked_sensitivity = np.ma.masked_array(
        cropped_sensitivity,
        mask=~cropped_wetmask.astype(bool)  # Invert wetmask to get drymask
    )
    print(f"Applied wetmask. Land areas will be masked in the plot.")
else:
    masked_sensitivity = cropped_sensitivity

# Plot the cropped and masked sensitivity matrix with indices as labels
plt.figure(figsize=(10, 8))
plt.imshow(masked_sensitivity, cmap='RdBu_r', aspect='auto', origin='lower')

# Get the row and column indices for the cropped region
row_indices = np.arange(xmin, xmax)
col_indices = np.arange(ymin, ymax)

# Set ticks at appropriate positions
y_positions = np.arange(cropped_sensitivity.shape[0])
x_positions = np.arange(cropped_sensitivity.shape[1])

# Display actual matrix indices on the axes
plt.xticks(x_positions[::36], col_indices[::36])  # Show every 36th index for readability
plt.yticks(y_positions[::18], row_indices[::18])  # Show every 18th index for readability

plt.colorbar(label='Sensitivity Value')
plt.title(f'$\\partial \\left( \\text{{2.5m }} \\theta_0 \\text{{ at }}({sensitivity_latitude},{sensitivity_longitude}), t={t0} \\right) / \\partial \\left( \\text{{2.5m }} \\theta_0 \\text{{ across map}}, t={t1} \\right)$')
plt.xlabel('Longitude Index')
plt.ylabel('Latitude Index')
plt.grid(False)
plt.tight_layout()

# Save the figure
plt.savefig('masked_sensitivity_map.png', dpi=300, bbox_inches='tight')
plt.show()