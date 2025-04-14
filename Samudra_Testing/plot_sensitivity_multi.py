import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

### PARAMETERS ###

final_time = 10  # Fixed final time
size = "local"   # Region size

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

if size == "manual":
    deltax = 2
    deltay = 2
    # Define the region of interest in the matrix for cropping
    xmin, xmax = 90-deltax, 90+deltax+1  # Matrix row indices
    ymin, ymax = 180-deltay, 180+deltay+1  # Matrix column indices
    size = f'{2*deltax+1}x{2*deltay+1}'

# Center point for the sensitivity analysis
sensitivity_latitude = 90
sensitivity_longitude = 180

### LOAD DRY MASK (IF AVAILABLE) ###
drymask_path = 'drymask.npy'
has_wetmask = False

# Turn into wetmask
if Path(drymask_path).exists():
    drymask = np.load(drymask_path)
    print(f"Loaded drymask with shape: {drymask.shape}")
    # Invert the drymask to get the wetmask (wet=0, dry=1)
    wetmask = np.where(drymask == 0, 1, 0)  # Invert the mask
    has_wetmask = True
else:
    print(f"Warning: Drymask file {drymask_path} not found. Proceeding without masking land areas.")

### LOAD SENSITIVITY MATRICES AND PLOT GRID ###

# Number of time steps to plot
num_times = 8  # t from 0 to 9

# Create a grid of subplots (2 rows, 5 columns)
fig, axs = plt.subplots(2, 5, figsize=(20, 8), 
                         gridspec_kw={'wspace': 0.02, 'hspace': 0.15})
axs = axs.flatten()

# Track min and max values across all sensitivity matrices for consistent colormap
overall_min = float('inf')
overall_max = float('-inf')

# First pass to determine global min/max for consistent colormap
sensitivity_matrices = []
for t in range(num_times):
    # Path to the sensitivity matrix file
    path = f'adjoint_sensitivity_matrix_t={t},{final_time}.npy'
    
    if not Path(path).exists():
        print(f"Warning: Sensitivity matrix file {path} not found.")
        sensitivity_matrices.append(None)
        continue
    
    # Load the sensitivity matrix
    sensitivity_matrix = np.load(path)
    print(f"Loaded sensitivity matrix for t={t} with shape: {sensitivity_matrix.shape}")
    
    # Reshape to 180x360 if needed (assuming it represents lat/lon grid)
    sensitivity_matrix = sensitivity_matrix.reshape(180, 360)
    
    # Crop the sensitivity matrix to the region of interest
    cropped_sensitivity = sensitivity_matrix[xmin:xmax, ymin:ymax]
    
    # Apply the wetmask if available
    if has_wetmask:
        # Crop the wetmask to match the sensitivity matrix
        cropped_wetmask = wetmask[xmin:xmax, ymin:ymax]
        
        # Create a masked array where land (wet=0) is masked
        masked_sensitivity = np.ma.masked_array(
            cropped_sensitivity,
            mask=~cropped_wetmask.astype(bool)  # Invert wetmask to get drymask
        )
    else:
        masked_sensitivity = cropped_sensitivity
    
    # Update overall min/max
    valid_min = np.nanmin(masked_sensitivity) if np.ma.is_masked(masked_sensitivity) else np.min(masked_sensitivity)
    valid_max = np.nanmax(masked_sensitivity) if np.ma.is_masked(masked_sensitivity) else np.max(masked_sensitivity)
    
    overall_min = min(overall_min, valid_min)
    overall_max = max(overall_max, valid_max)
    
    sensitivity_matrices.append(masked_sensitivity)

# Make the colormap symmetric around zero if the data spans positive and negative
if overall_min < 0 and overall_max > 0:
    abs_max = max(abs(overall_min), abs(overall_max))
    vmin, vmax = -abs_max, abs_max
else:
    vmin, vmax = overall_min, overall_max

print(f"Global value range: [{vmin}, {vmax}]")

# Second pass to create plots with consistent colormap
for t, (ax, masked_sensitivity) in enumerate(zip(axs, sensitivity_matrices)):
    if masked_sensitivity is None:
        ax.text(0.5, 0.5, f"No data for t={t}", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        continue
    
    # Plot the masked sensitivity
    im = ax.imshow(masked_sensitivity, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                   aspect='auto', origin='lower')
    
    # Set proper title
    ax.set_title(f't={t} → t={final_time}', fontsize=12)
    
    # Get the row and column indices for the cropped region
    row_indices = np.arange(xmin, xmax)
    col_indices = np.arange(ymin, ymax)
    
    # Set ticks at appropriate positions
    y_positions = np.arange(masked_sensitivity.shape[0])
    x_positions = np.arange(masked_sensitivity.shape[1])
    
    # Only add y-axis labels for leftmost plots (first column)
    if t % 5 == 0:
        # Show every 20th index for latitude
        ax.set_yticks(y_positions[::20])
        ax.set_yticklabels(row_indices[::20])
        ax.set_ylabel('Latitude')
    else:
        ax.set_yticks([])
    
    # Only add x-axis labels for bottom plots (second row)
    if t >= 5:
        # Show every 60th index for longitude
        ax.set_xticks(x_positions[::60])
        ax.set_xticklabels(col_indices[::60])
        ax.set_xlabel('Longitude')
    else:
        ax.set_xticks([])

# Add a single colorbar for all subplots
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Sensitivity Value')

# Add an overall title
plt.suptitle(f'Sensitivity of θ₀ at ({sensitivity_latitude},{sensitivity_longitude}) '
             f'for varying initial times relative to t={final_time}', 
             fontsize=16, y=0.98)

# Save the figure
plt.savefig(f'adjoint_sensitivity_grid_t0-9_t10_{size}.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Plotted sensitivity grid for t=0-9 relative to t={final_time}")