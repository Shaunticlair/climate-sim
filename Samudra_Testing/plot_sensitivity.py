import numpy as np
import matplotlib.pyplot as plt

# Path to the sensitivity matrix file
path = 'sensitivity_matrix.npy'

# Load the sensitivity matrix
sensitivity_matrix = np.load(path)
print(f"Original sensitivity matrix shape: {sensitivity_matrix.shape}")

# Reshape to 180x360 if needed (assuming it represents lat/lon grid)
sensitivity_matrix = sensitivity_matrix.reshape(180, 360)
print(f"Reshaped sensitivity matrix: {sensitivity_matrix.shape}")

# Define the region of interest in the matrix for cropping
xmin, xmax = 90-3, 90+4  # Matrix row indices
ymin, ymax = 180-3, 180+4  # Matrix column indices

xmin, xmax = 0, 180
ymin, ymax = 0, 360

# Crop the sensitivity matrix to the region of interest
cropped_sensitivity = sensitivity_matrix[xmin:xmax, ymin:ymax]
print(f"Cropped sensitivity matrix: {cropped_sensitivity.shape}")

# Plot the cropped sensitivity matrix with indices as labels
plt.figure(figsize=(10, 8))
plt.imshow(cropped_sensitivity, cmap='RdBu_r', aspect='auto', origin='lower')

# Get the row and column indices for the cropped region
row_indices = np.arange(xmin, xmax)
col_indices = np.arange(ymin, ymax)

# Set ticks at appropriate positions
y_positions = np.arange(cropped_sensitivity.shape[0])
x_positions = np.arange(cropped_sensitivity.shape[1])

# Display actual matrix indices on the axes
plt.xticks(x_positions[::2], col_indices[::2])  # Show every other index for readability
plt.yticks(y_positions[::1], row_indices[::1])

plt.colorbar(label='Sensitivity Value')
plt.title('Sensitivity Matrix Heatmap')
plt.xlabel('Matrix Column Index')
plt.ylabel('Matrix Row Index')
plt.grid(False)
plt.tight_layout()
plt.show()