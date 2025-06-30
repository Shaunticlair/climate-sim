import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def compare_sensitivities(adjoint_path, perturb_path, t0, t1, chin, chout, 
                          grid_size=5, magnitude=3, output_pixel=(90, 180)):
    """
    Compare adjoint and perturbation sensitivities in a 3-panel plot format.
    
    Parameters:
    -----------
    adjoint_path : Path or str
        Path to the adjoint sensitivity numpy file
    perturb_path : Path or str
        Path to the perturbation sensitivity numpy file
    t0, t1 : int
        Initial and final time steps for the sensitivity
    chin : int
        Input channel number
    chout : int
        Output channel number
    grid_size : int
        Size of the finite-difference grid (grid_size x grid_size)
    magnitude : int
        Magnitude for the finite-difference sensitivity (1e-magnitude)
    output_pixel : tuple
        (lat, lon) of the output pixel
        
    Returns:
    --------
    dict
        Dictionary containing correlation statistics and metadata
    """
    
    # Check if files exist
    if not Path(adjoint_path).exists():
        print(f"Error: Adjoint sensitivity file {adjoint_path} not found.")
        return None
    
    if not Path(perturb_path).exists():
        print(f"Error: Finite difference sensitivity file {perturb_path} not found.")
        return None
    
    # Load sensitivity matrices
    adjoint_sensitivity = np.load(adjoint_path)
    fd_sensitivity = np.load(perturb_path)
    
    # Reshape adjoint sensitivity if needed (assuming it represents lat/lon grid)
    if len(adjoint_sensitivity.shape) > 2:
        adjoint_sensitivity = adjoint_sensitivity.reshape(180, 360)
    
    # Define center of sensitivity (assumed to be equatorial Pacific)
    sensitivity_latitude, sensitivity_longitude = output_pixel
    
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
    if len(adjoint_values) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(adjoint_values, fd_values)
        corr, p_val = stats.pearsonr(adjoint_values, fd_values)
    else:
        slope, intercept, r_value, p_value, std_err = np.nan, np.nan, np.nan, np.nan, np.nan
        corr, p_val = np.nan, np.nan
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    # Plot 1: Adjoint Sensitivity
    abs_max_adj = np.max(np.abs(adjoint_grid))
    divider0 = make_axes_locatable(axes[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    
    im1 = axes[0].imshow(adjoint_grid, cmap='RdBu_r', origin='lower', 
                         vmin=-abs_max_adj, vmax=abs_max_adj)
    axes[0].set_title(f'Adjoint Sensitivity (t={t0},{t1})')
    cbar1 = plt.colorbar(im1, cax=cax0, label='Sensitivity Value')
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
    cbar2 = plt.colorbar(im2, cax=cax1, label='Sensitivity Value')
    cbar2.formatter.set_powerlimits((-2, 2))
    axes[1].set_xlabel('Longitude Index (centered at 180)')
    axes[1].set_ylabel('Latitude Index (centered at 90)')
    
    # Plot 3: Correlation
    if len(adjoint_values) > 0:
        axes[2].scatter(adjoint_values, fd_values, alpha=0.7)
        
        # Add best fit line
        if not np.isnan(slope):
            x_line = np.linspace(min(adjoint_values), max(adjoint_values), 100)
            y_line = slope * x_line + intercept
            axes[2].plot(x_line, y_line, 'r-', linewidth=2)
        
        # Make axes equal
        axes[2].set_aspect('equal', adjustable='box')
        
        # Format axes with scientific notation
        axes[2].ticklabel_format(style='sci', scilimits=(-2, 2), axis='both')
        axes[2].xaxis.get_offset_text().set_position((1.1, 0))
        
        # Add correlation information
        if not np.isnan(corr):
            axes[2].text(0.05, 0.75, f"Correlation: {corr:.4f}\nSlope: {slope:.4f}\nIntercept: {intercept:.4f}", 
                     transform=axes[2].transAxes, bbox=dict(facecolor='white', alpha=0.8))
    else:
        axes[2].text(0.5, 0.5, "No valid data points for comparison", 
                     ha='center', va='center', transform=axes[2].transAxes)
    
    axes[2].set_title('Sensitivity Comparison')
    axes[2].set_xlabel('Adjoint Sensitivity')
    axes[2].set_ylabel('Finite Difference Sensitivity')
    axes[2].grid(True)
    
    # Set common title
    fig.suptitle(f'Comparison of Sensitivity Methods (Channel zos → zos, Time {t0} → {t1})', 
                 fontsize=24, y=1.1)
    
    # Create Plots directory if it doesn't exist
    Path("Plots").mkdir(exist_ok=True)
    
    # Save the figure
    output_path = f'Plots/comparison_{grid_size}x{grid_size}_t={t0},{t1}_1e-{magnitude}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved as {output_path}")
    
    # Return correlation info
    return {
        'correlation': corr,
        'p_value': p_val,
        'n_points': len(adjoint_values),
        'times': (t0, t1),
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2 if not np.isnan(r_value) else np.nan
    }

def plot_correlation_vs_lag(results):
    """
    Create a plot showing correlation vs time lag between adjoint and perturbation sensitivities.
    Focus only on the high correlation range (0.9-1.0).
    
    Parameters:
    -----------
    results : list of dict
        List of result dictionaries from compare_sensitivities function
    """
    # Extract time lags and correlations from results
    time_lags = []
    correlations = []
    
    for result in results:
        if result is None:  # Skip failed comparisons
            continue
        t0, t1 = result['times']
        time_lag = t1 - t0
        correlation = result['correlation']
        
        time_lags.append(time_lag)
        correlations.append(correlation)
    
    # Sort by time lag (in case they're not already in order)
    sorted_data = sorted(zip(time_lags, correlations))
    time_lags = [data[0] for data in sorted_data]
    correlations = [data[1] for data in sorted_data]
    
    # Create a single figure focused on the high correlation range
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define the y-axis range
    y_min = 0.9    # Lower limit (high correlation range)
    y_max = max(1.0, max(correlations) + 0.02) if correlations else 1.0  # Upper limit with a small margin
    
    # Plot data
    ax.plot(time_lags, correlations, 'o-', color='#1f77b4', linewidth=2, markersize=10)
    
    # Add horizontal reference line at perfect correlation
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    
    # Set y-axis limits - focus on 0.9-1.0 range
    ax.set_ylim(y_min, y_max)
    
    # Adjust tick spacing for better precision in high correlation range
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Labels and title
    ax.set_title('Correlation between Adjoint and Perturbation Sensitivities vs Time Lag')
    ax.set_xlabel('Time Lag (t_end - t_start)')
    ax.set_ylabel('Correlation Coefficient')
    
    # Annotate each point with its exact correlation value
    for x, y in zip(time_lags, correlations):
        ax.annotate(f'{y:.3f}', 
                    (x, y), 
                    textcoords="offset points",
                    xytext=(0, 10), 
                    ha='center',
                    fontsize=15)
    
    # Add summary information
    if correlations:
        avg_corr = sum(correlations) / len(correlations)
        plt.figtext(0.02, 0.02, f"Average correlation: {avg_corr:.3f}", fontsize=12)
    
    # Save the figure
    Path("Plots").mkdir(exist_ok=True)
    plt.savefig('Plots/correlation_vs_time_lag.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plot saved to Plots/correlation_vs_time_lag.png")
    return 'Plots/correlation_vs_time_lag.png'

if __name__ == "__main__":
    # Parameters from the original scripts
    t_end = 10
    
    # Time points to analyze
    initial_times = [0, 2, 4, 6, 8]
    
    # Variables
    ch_in, ch_out = 76, 76
    
    # Output pixel
    output_pixel = (90, 180)  # Equatorial Pacific
    
    # Perturbation settings
    magnitude = 2  # For 1e-2 perturbation
    grid_size = 5
    
    # Run comparisons for each time point
    results = []
    for initial_time in initial_times:
        in_time, out_time = initial_time, t_end
        
        # Construct file paths
        adjoint_folder = 'adjoint_arrays/Equatorial_Pacific/'
        adjoint_path = f'chunk_sensitivity_chin[{ch_in}]_chout[{ch_out}]_t[{in_time},{out_time}].npy'
        adjoint_path = Path(adjoint_folder) / adjoint_path
        
        perturb_folder = f'perturbation_arrays/Short_Time_1e-{magnitude}/'
        perturb_path = f'perturbation_grid_chin[{ch_in}]_chout[{ch_out}]_t[{in_time},{out_time}]_1e-{magnitude}.npy'
        perturb_path = Path(perturb_folder) / perturb_path
        
        # Check if both files exist
        if Path(adjoint_path).exists() and Path(perturb_path).exists():
            print(f"Comparing sensitivities for t=({in_time},{out_time})")
            
            result = compare_sensitivities(
                adjoint_path, perturb_path, in_time, out_time,
                ch_in, ch_out, grid_size, magnitude, output_pixel
            )
            
            results.append(result)
            
            if result is not None:
                print(f"Correlation: {result['correlation']:.4f} (p={result['p_value']:.4e})")
        else:
            missing_files = []
            if not Path(adjoint_path).exists():
                missing_files.append(f"adjoint file: {adjoint_path}")
            if not Path(perturb_path).exists():
                missing_files.append(f"perturbation file: {perturb_path}")
            print(f"Missing files for t=({in_time},{out_time}): {', '.join(missing_files)}")
    
    # Create summary table
    valid_results = [r for r in results if r is not None]
    if valid_results:
        print("\nCorrelation Summary:")
        print("-" * 70)
        print(f"{'Time':^10}{'Correlation':^15}{'p-value':^15}{'R-squared':^15}{'N Points':^10}")
        print("-" * 70)
        
        for result in valid_results:
            t0, t1 = result['times']
            print(f"({t0},{t1}){result['correlation']:^15.4f}{result['p_value']:^15.4e}{result['r_squared']:^15.4f}{result['n_points']:^10}")
        
        # Generate the correlation vs time lag plot
        plot_path = plot_correlation_vs_lag(results)
        print(f"\nCorrelation vs Time Lag plot saved to: {plot_path}")