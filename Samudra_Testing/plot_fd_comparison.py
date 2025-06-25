import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable  # Import for better colorbar alignment

# Set global font sizes
x = 1.5
plt.rcParams['font.size'] = 24/2*x
plt.rcParams['axes.titlesize'] = 32/2*x
plt.rcParams['axes.labelsize'] = 28/2*x
plt.rcParams['xtick.labelsize'] = 24/2*x
plt.rcParams['ytick.labelsize'] = 24/2*x
plt.rcParams['legend.fontsize'] = 24/2*x
plt.rcParams['axes.titlepad'] = 24/2*x

# Adjust spacing parameters
plt.rcParams['figure.constrained_layout.use'] = True  # Use constrained layout
plt.rcParams['axes.titlepad'] = 40  # Increase padding between title and plot
plt.rcParams['figure.subplot.wspace'] = 10  # Increase width spacing between subplots
plt.rcParams['figure.subplot.hspace'] = 0  # Increase height spacing between subplots
# Add spacing for suptitle
#plt.rcParams['figure.subplot.top'] = 100  # Adjust top spacing for suptitle

def plot_sensitivity_comparison(t0, t1, chin, chout, magnitude=3, grid_size=5):
    """
    Create three side-by-side plots comparing adjoint and finite-difference sensitivity methods.
    
    Parameters:
    -----------
    t0 : int
        Start time point
    t1 : int
        End time point
    chin : int
        Input channel number
    chout : int
        Output channel number
    magnitude : int
        Magnitude for the finite-difference sensitivity (1e-magnitude)
    grid_size : int
        Size of the finite-difference grid (grid_size x grid_size)
    """
    
    # Define paths to sensitivity matrices
    adjoint_file = f'chunk_sensitivity_chin[{chin}]_chout[{chout}]_t[{t0},{t1}].npy'
    adjoint_path = f'adjoint_arrays/Equatorial_Pacific/' + adjoint_file
    fd_file = f'perturbation_grid_chin[{chin}]_chout[{chout}]_t[{t0},{t1}]_1e-{magnitude}.npy'
    fd_path = f'perturbation_arrays/Short_Time_1e-{magnitude}/' + fd_file
    
    # Check if files exist
    if not Path(adjoint_path).exists():
        print(f"Error: Adjoint sensitivity file {adjoint_path} not found.")
        return
    
    if not Path(fd_path).exists():
        print(f"Error: Finite difference sensitivity file {fd_path} not found.")
        return
    
    # Load sensitivity matrices
    adjoint_sensitivity = np.load(adjoint_path)
    fd_sensitivity = np.load(fd_path)
    
    # Reshape adjoint sensitivity if needed (assuming it represents lat/lon grid)
    if len(adjoint_sensitivity.shape) > 2:
        adjoint_sensitivity = adjoint_sensitivity.reshape(180, 360)
    
    # Define center of sensitivity (assumed to be equatorial Pacific)
    sensitivity_latitude = 90
    sensitivity_longitude = 180
    
    # Extract the corresponding grid from the adjoint sensitivity
    half_grid = grid_size // 2
    adjoint_grid = np.zeros((grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            lat_idx = sensitivity_latitude - half_grid + i
            lon_idx = sensitivity_longitude - half_grid + j
            
            if (0 <= lat_idx < adjoint_sensitivity.shape[0] and 
                0 <= lon_idx < adjoint_sensitivity.shape[1]):
                adjoint_grid[i, j] = adjoint_sensitivity[lat_idx, lon_idx]
    
    # Prepare for correlation plot
    adjoint_values = []
    fd_values = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Skip if NaN in finite difference results
            if np.isnan(fd_sensitivity[i, j]):
                continue
            
            adjoint_values.append(adjoint_grid[i, j])
            fd_values.append(fd_sensitivity[i, j])
    
    # Convert to numpy arrays
    adjoint_values = np.array(adjoint_values)
    fd_values = np.array(fd_values)
    
    # Calculate correlation
    slope, intercept, r_value, p_value, std_err = stats.linregress(adjoint_values, fd_values)
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    # Plot 1: Adjoint Sensitivity
    abs_max_adj = np.max(np.abs(adjoint_grid))
    divider0 = make_axes_locatable(axes[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    
    im1 = axes[0].imshow(adjoint_grid, cmap='RdBu_r', origin='lower', 
                         vmin=-abs_max_adj, vmax=abs_max_adj)
    axes[0].set_title(f'Adjoint Sensitivity (t={t0},{t1})')
    cbar1 = plt.colorbar(im1, cax=cax0, label='Sensitivity Value')  # Use the custom axes for colorbar
    cbar1.formatter.set_powerlimits((-2, 2))
    axes[0].set_xlabel('Longitude Index (centered at 180)')
    axes[0].set_ylabel('Latitude Index (centered at 90)')
    
    # Plot 2: Finite Difference Sensitivity
    abs_max_fd = np.max(np.abs(fd_sensitivity))
    divider1 = make_axes_locatable(axes[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    
    im2 = axes[1].imshow(fd_sensitivity, cmap='RdBu_r', origin='lower', 
                         vmin=-abs_max_fd, vmax=abs_max_fd)
    axes[1].set_title(f'Finite Difference Sensitivity (t={t0},{t1}, 1e-{magnitude})')
    cbar2 = plt.colorbar(im2, cax=cax1, label='Sensitivity Value')  # Use the custom axes for colorbar
    cbar2.formatter.set_powerlimits((-2, 2))
    axes[1].set_xlabel('Longitude Index (centered at 180)')
    axes[1].set_ylabel('Latitude Index (centered at 90)')
    
    # Plot 3: Correlation
    axes[2].scatter(adjoint_values, fd_values, alpha=0.7)
    
    # Add best fit line
    x_line = np.linspace(min(adjoint_values), max(adjoint_values), 100)
    y_line = slope * x_line + intercept
    axes[2].plot(x_line, y_line, 'r-', linewidth=2)
    axes[2].ticklabel_format(style='sci', scilimits=(-2, 2), axis='both')
    axes[2].xaxis.get_offset_text().set_position((1.1, 0))
    # Make axes equal
    axes[2].set_aspect('equal', adjustable='box')
    
    # Add correlation information
    axes[2].text(0.05, 0.75, f"Correlation: {r_value:.4f}\nSlope: {slope:.4f}\nIntercept: {intercept:.4f}", 
             transform=axes[2].transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    axes[2].set_title('Sensitivity Comparison')
    axes[2].set_xlabel('Adjoint Sensitivity')
    axes[2].set_ylabel('Finite Difference Sensitivity')
    axes[2].grid(True)
    
    #plt.tight_layout()
    
    # Create Plots directory if it doesn't exist
    Path("Plots").mkdir(exist_ok=True)
    
    # Save the figure
    fig.suptitle(f'Comparison of Sensitivity Methods (Channel zos → zos, Time {t0} → {t1})', 
             fontsize=24, y=1.1)
    
    output_path = f'Plots/comparison_{grid_size}x{grid_size}_t={t0},{t1}_1e-{magnitude}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved as {output_path}")

if __name__ == "__main__":
    # Example parameters
    t0 = 0
    t1 = 10
    magnitude = 2
    grid_size = 5
    chin, chout = 76, 76
    
    plot_sensitivity_comparison(t0, t1, chin, chout, magnitude, grid_size)