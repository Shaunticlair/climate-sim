import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

### PARAMETERS ###

t_start,t_end = 0, 10

def plot(t0,t1):

    ### PARAMETERS ###
    size = "manual"
    magn= 4
    grid_size = 5

    fd_path = f'perturb_sensitivity_grid_{grid_size}x{grid_size}_t={t0},{t1}_1e-{magn}.npy' #

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
        deltax = 40
        deltay = 40
        # Define the region of interest in the matrix for cropping
        xmin, xmax = 90-deltax, 90+deltax+1  # Matrix row indices
        ymin, ymax = 180-deltay, 180+deltay+1  # Matrix column indices
        size = f'{2*deltax+1}x{2*deltay+1}'

    ### PLOT SENSITIVITY ###

    # Path to the sensitivity matrix file
    path = f'adjoint_sensitivity_matrix_t={t0},{t1}.npy'

    # Load the sensitivity matrix
    sensitivity_matrix = np.load(path)
    print(f"Original sensitivity matrix shape: {sensitivity_matrix.shape}")

    # Load the finite difference sensitivity grid
    
    if Path(fd_path).exists():
        fd_sensitivity = np.load(fd_path)
        print(f"Loaded finite difference sensitivity with shape: {fd_sensitivity.shape}")
        has_fd_sensitivity = True
    else:
        print(f"Warning: Finite difference sensitivity file {fd_path} not found.")
        has_fd_sensitivity = False

    sensitivity_latitude = 90  # Center latitude for the matrix (assuming 180x360 grid)
    sensitivity_longitude = 180  # Center longitude for the matrix (assuming 180x360 grid)
    x,y = sensitivity_latitude,sensitivity_longitude

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

    # Create symmetric limits around zero
    abs_max = np.max(np.abs(masked_sensitivity))
    vmin, vmax = -abs_max, abs_max
    print(f"Using symmetric color limits: [{vmin}, {vmax}]")

    im = plt.imshow(masked_sensitivity, cmap='RdBu_r', aspect='auto', origin='lower', 
                    vmin=vmin, vmax=vmax)

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
    name = f'adjoint_map_{x}-{y}_t={t0},{t1}_{size}.png'
    print(name)
    plt.savefig(name, dpi=300, bbox_inches='tight')

    ### PLOT CORRELATION ###

    # Compare sensitivity matrix with finite difference results if available
    if has_fd_sensitivity:
        # Extract sensitivity values from the adjoint method for the grid
        grid_size = fd_sensitivity.shape[0]
        half_grid = grid_size // 2
        
        # Extract the corresponding values from the sensitivity matrix
        adjoint_values = []
        fd_values = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate matrix indices
                lat_idx = sensitivity_latitude - half_grid + i
                lon_idx = sensitivity_longitude - half_grid + j
                
                # Skip if outside valid range or NaN in finite difference results
                if (lat_idx < 0 or lat_idx >= sensitivity_matrix.shape[0] or 
                    lon_idx < 0 or lon_idx >= sensitivity_matrix.shape[1] or
                    np.isnan(fd_sensitivity[i, j])):
                    continue
                
                adjoint_values.append(sensitivity_matrix[lat_idx, lon_idx])
                fd_values.append(fd_sensitivity[i, j])
        
        # Convert to numpy arrays
        adjoint_values = np.array(adjoint_values)
        fd_values = np.array(fd_values)
        
        # Calculate correlation
        slope, intercept, r_value, p_value, std_err = stats.linregress(adjoint_values, fd_values)
        
        # Create a scatter plot
        plt.figure(figsize=(8, 8))
        plt.scatter(adjoint_values, fd_values, alpha=0.7)
        
        # Add best fit line
        x_line = np.linspace(min(adjoint_values), max(adjoint_values), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, 'r-', linewidth=2)
        
        # Add correlation information
        plt.text(0.05, 0.95, f"Correlation: {r_value:.4f}\nSlope: {slope:.4f}\nIntercept: {intercept:.4f}", 
                transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.title('Sensitivity Comparison: Adjoint vs. Finite Difference')
        plt.xlabel('Adjoint Sensitivity')
        plt.ylabel('Finite Difference Sensitivity')
        plt.grid(True)
        plt.tight_layout()
        
        # Save the correlation plot
        plt.savefig('sensitivity_correlation.png', dpi=300, bbox_inches='tight')

    plt.close('all')
    print("Plots saved with symmetric color scale around zero.")

#for t0 in range(t_start, t_end):
#    print(t0)
#    plot(t0,t_end)

plot(0,2)