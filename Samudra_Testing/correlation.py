import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats as stats
from matplotlib.ticker import ScalarFormatter

def compare_sensitivities(adjoint_path, perturb_path, map_dims, t0, t1, 
                          output_pixel, output_var, input_var,
                          scatter_kwargs=None, correlation_kwargs=None, step_size=''):
    """
    Compare adjoint and perturbation sensitivities for the same region.
    
    Parameters:
    -----------
    adjoint_path : Path or str
        Path to the adjoint sensitivity numpy file
    perturb_path : Path or str
        Path to the perturbation sensitivity numpy file
    map_dims : list
        [xmin, xmax, ymin, ymax] defining the region to plot
    t0, t1 : int
        Initial and final time steps for the sensitivity
    output_pixel : tuple
        (lat, lon) of the output pixel
    output_var, input_var : str
        Names of output and input variables
    scatter_kwargs : dict
        Optional kwargs for scatter plot
    correlation_kwargs : dict
        Optional kwargs for correlation calculation
    """
    # Default kwargs
    if scatter_kwargs is None:
        scatter_kwargs = {'s': 10, 'alpha': 0.7}
    if correlation_kwargs is None:
        correlation_kwargs = {'method': 'pearson'}
        
    # Process dimensions
    xmin, xmax, ymin, ymax = map_dims
    output_lat, output_lon = output_pixel
    view_name = f"({xmin},{xmax})x({ymin},{ymax})"
    
    # Load sensitivity matrices and handle different shapes
    adjoint_matrix = np.load(adjoint_path)
    perturb_matrix = np.load(perturb_path)
    
    print(f"Adjoint matrix shape: {adjoint_matrix.shape}")
    print(f"Perturbation matrix shape: {perturb_matrix.shape}")
    
    # Reshape adjoint if needed (typically 180x360)
    if len(adjoint_matrix.shape) > 2:
        adjoint_matrix = adjoint_matrix.reshape(180, 360)
    
    # Ensure perturb matrix matches our expected dimensions
    height, width = xmax - xmin, ymax - ymin
    if perturb_matrix.shape == (height, width):
        # Perturbation matrix is already cropped
        cropped_perturb = perturb_matrix
        cropped_adjoint = adjoint_matrix[xmin:xmax, ymin:ymax]
    else:
        # Reshape perturbation matrix if needed
        if len(perturb_matrix.shape) > 2:
            perturb_matrix = perturb_matrix.reshape(180, 360)
            
        # Crop both matrices to region of interest
        cropped_adjoint = adjoint_matrix[xmin:xmax, ymin:ymax]
        cropped_perturb = perturb_matrix[xmin:xmax, ymin:ymax]
    
    print(f"Cropped adjoint shape: {cropped_adjoint.shape}")
    print(f"Cropped perturb shape: {cropped_perturb.shape}")
    
    # Load wetmask if available
    drymask_path = 'drymask.npy'
    if Path(drymask_path).exists():
        drymask = np.load(drymask_path)
        wetmask = np.where(drymask == 0, 1, 0)  # Invert the mask
        cropped_wetmask = wetmask[xmin:xmax, ymin:ymax]
        
        # Apply mask only if shapes match
        if cropped_wetmask.shape == cropped_adjoint.shape:
            masked_adjoint = np.ma.masked_array(
                cropped_adjoint, 
                mask=~cropped_wetmask.astype(bool)
            )
        else:
            print("Wetmask shape doesn't match adjoint. Using unmasked adjoint.")
            masked_adjoint = cropped_adjoint
            
        if cropped_wetmask.shape == cropped_perturb.shape:
            masked_perturb = np.ma.masked_array(
                cropped_perturb, 
                mask=~cropped_wetmask.astype(bool)
            )
        else:
            print("Wetmask shape doesn't match perturbation. Using unmasked perturbation.")
            masked_perturb = cropped_perturb
    else:
        print("Wetmask not found. Using raw sensitivities.")
        masked_adjoint = cropped_adjoint
        masked_perturb = cropped_perturb
    
    # Check for NaN values
    if np.isnan(masked_adjoint).any():
        print("Warning: NaN values in adjoint matrix will be masked")
        adjoint_nan_mask = np.isnan(masked_adjoint)
        if isinstance(masked_adjoint, np.ma.MaskedArray):
            masked_adjoint.mask = masked_adjoint.mask | adjoint_nan_mask
        else:
            masked_adjoint = np.ma.masked_array(masked_adjoint, mask=adjoint_nan_mask)
    
    if np.isnan(masked_perturb).any():
        print("Warning: NaN values in perturbation matrix will be masked")
        perturb_nan_mask = np.isnan(masked_perturb)
        if isinstance(masked_perturb, np.ma.MaskedArray):
            masked_perturb.mask = masked_perturb.mask | perturb_nan_mask
        else:
            masked_perturb = np.ma.masked_array(masked_perturb, mask=perturb_nan_mask)
    
    # Ensure we have comparable data by making a common mask
    if isinstance(masked_adjoint, np.ma.MaskedArray) and isinstance(masked_perturb, np.ma.MaskedArray):
        common_mask = masked_adjoint.mask | masked_perturb.mask
        masked_adjoint = np.ma.masked_array(masked_adjoint.data, mask=common_mask)
        masked_perturb = np.ma.masked_array(masked_perturb.data, mask=common_mask)
    
    # Flatten the masked arrays for scatter plot, handling masks properly
    if isinstance(masked_adjoint, np.ma.MaskedArray) and isinstance(masked_perturb, np.ma.MaskedArray):
        # Only use points where both are not masked
        valid_indices = ~(masked_adjoint.mask | masked_perturb.mask)
        adjoint_flat = masked_adjoint.data[valid_indices].flatten()
        perturb_flat = masked_perturb.data[valid_indices].flatten()
    else:
        # If no masking, just flatten
        adjoint_flat = masked_adjoint.flatten()
        perturb_flat = masked_perturb.flatten()
    
    print(f"Number of valid comparison points: {len(adjoint_flat)}")
    
    # Calculate correlation coefficient if we have enough points
    if len(adjoint_flat) > 1:  # Need at least 2 points for correlation
        method = correlation_kwargs.get('method', 'pearson')
        if method == 'pearson':
            corr, p_value = stats.pearsonr(adjoint_flat, perturb_flat)
        elif method == 'spearman':
            corr, p_value = stats.spearmanr(adjoint_flat, perturb_flat)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
    else:
        corr, p_value = np.nan, np.nan
        print("Not enough points for correlation calculation")
    
    # Create plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Common colormap settings
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad('black', 0.5)
    
    # Calculate max absolute value for symmetric coloring
    abs_max_adjoint = np.max(np.abs(masked_adjoint.compressed())) if isinstance(masked_adjoint, np.ma.MaskedArray) else np.max(np.abs(masked_adjoint))
    abs_max_perturb = np.max(np.abs(masked_perturb.compressed())) if isinstance(masked_perturb, np.ma.MaskedArray) else np.max(np.abs(masked_perturb))
    abs_max = max(abs_max_adjoint, abs_max_perturb)
    
    # Plot adjoint sensitivity
    im1 = ax1.imshow(masked_adjoint, cmap=cmap, aspect='equal', origin='lower',
                     vmin=-abs_max, vmax=abs_max)
    ax1.set_title(f'Adjoint Sensitivity t=({t0},{t1})')
    
    # Use scientific notation for colorbar
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    
    # Plot perturbation sensitivity
    im2 = ax2.imshow(masked_perturb, cmap=cmap, aspect='equal', origin='lower',
                     vmin=-abs_max, vmax=abs_max)
    ax2.set_title(f'Perturbation Sensitivity t=({t0},{t1})')
    
    # Create scatter plot of adjoint vs perturbation
    if len(adjoint_flat) > 0:
        ax3.scatter(adjoint_flat, perturb_flat, **scatter_kwargs)
        
        # Add perfect correlation line
        min_val = min(adjoint_flat.min(), perturb_flat.min())
        max_val = max(adjoint_flat.max(), perturb_flat.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
        
        # Add regression line
        if len(adjoint_flat) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(adjoint_flat, perturb_flat)
            x_range = np.linspace(min(adjoint_flat), max(adjoint_flat), 100)
            ax3.plot(x_range, intercept + slope * x_range, 'r-', alpha=0.7)
            
        # Use scientific notation for scatter plot axes
        ax3.ticklabel_format(style='sci', scilimits=(-2, 2), axis='both')
    else:
        ax3.text(0.5, 0.5, "No valid data points for comparison", 
                 ha='center', va='center', transform=ax3.transAxes)
    
    ax3.set_xlabel('Adjoint Sensitivity')
    ax3.set_ylabel('Perturbation Sensitivity')
    
    if not np.isnan(corr):
        ax3.set_title(f'Correlation: {corr:.4f} (p={p_value:.4e})')
    else:
        ax3.set_title('No correlation (insufficient data)')
        
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar with scientific notation
    cbar1 = plt.colorbar(im1, ax=ax1, label='Sensitivity Value', format=formatter)
    cbar2 = plt.colorbar(im2, ax=ax2, label='Sensitivity Value', format=formatter)
    
    # Format variable names for title
    output_var = output_var.replace('(odd)', '').replace('(even)', '')
    input_var = input_var.replace('(odd)', '').replace('(even)', '')
    
    # Set common title
    fig.suptitle(f'Sensitivity Comparison: {input_var} -> {output_var} at ({output_lat},{output_lon}), t=({t0},{t1})', 
                 fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    
    # Create Plots directory if it doesn't exist
    Path("Plots").mkdir(exist_ok=True)
    
    name = f'Plots/comparison_{view_name}_chin[{input_var}]_chout[{output_var}]_t[{t0},{t1}]{step_size}.png'
    print(f"Saving comparison plot to: {name}")
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return correlation info
    return {
        'correlation': corr,
        'p_value': p_value,
        'n_points': len(adjoint_flat),
        'times': (t0, t1),
        'variables': (input_var, output_var)
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
    y_max = max(1.0, max(correlations) + 0.02)  # Upper limit with a small margin
    
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
    ax.set_title('Correlation between Adjoint and Perturbation Sensitivities vs Time Lag', fontsize=14)
    ax.set_xlabel('Time Lag (t_end - t_start)', fontsize=12)
    ax.set_ylabel('Correlation Coefficient', fontsize=12)
    
    # Annotate each point with its exact correlation value
    for x, y in zip(time_lags, correlations):
        ax.annotate(f'{y:.3f}', 
                    (x, y), 
                    textcoords="offset points",
                    xytext=(0, 10), 
                    ha='center',
                    fontsize=9)
    
    # Add summary information
    avg_corr = sum(correlations) / len(correlations) if correlations else 0
    plt.figtext(0.02, 0.02, f"Average correlation: {avg_corr:.3f}", fontsize=10)
    
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
    var_in, var_out = 'zos', 'zos'  # For testing purposes
    
    # Output pixel
    output_pixel = (90, 180)  # Equatorial Pacific
    
    # Map dimensions
    delta = 2
    map_dims = [90-delta, 90+delta+1, 180-delta, 180+delta+1]  # [xmin, xmax, ymin, ymax]
    
    # Run comparisons for each time point
    results = []
    for initial_time in initial_times:
        in_time, out_time = initial_time, t_end
        
        # Construct file paths
        adjoint_folder = 'adjoint_arrays/Equatorial_Pacific/'
        adjoint_path = f'chunk_sensitivity_chin[{ch_in}]_chout[{ch_out}]_t[{in_time},{out_time}].npy'
        adjoint_path = Path(adjoint_folder) / adjoint_path
        step_size = '_1e-2'  # Use specified step size
        perturb_folder = f'perturbation_arrays/Short_Time{step_size}/'
        perturb_path = f'perturbation_grid_chin[{ch_in}]_chout[{ch_out}]_t[{in_time},{out_time}]{step_size}.npy'
        perturb_path = Path(perturb_folder) / perturb_path
        
        # Check if both files exist
        if Path(adjoint_path).exists() and Path(perturb_path).exists():
            print(f"Comparing sensitivities for t=({in_time},{out_time})")
            
            result = compare_sensitivities(
                adjoint_path, perturb_path, map_dims, in_time, out_time,
                output_pixel, var_out, var_in,
                scatter_kwargs={'s': 15, 'alpha': 0.7, 'c': 'blue'},
                correlation_kwargs={'method': 'pearson'},
                step_size=step_size
            )
            
            results.append(result)
            print(f"Correlation: {result['correlation']:.4f} (p={result['p_value']:.4e})")
        else:
            missing_files = []
            if not Path(adjoint_path).exists():
                missing_files.append(f"adjoint file: {adjoint_path}")
            if not Path(perturb_path).exists():
                missing_files.append(f"perturbation file: {perturb_path}")
            print(f"Missing files for t=({in_time},{out_time}): {', '.join(missing_files)}")
    
    # Create summary table
    if results:
        print("\nCorrelation Summary:")
        print("-" * 70)
        print(f"{'Time':^10}{'Input Var':^15}{'Output Var':^15}{'Correlation':^15}{'p-value':^15}{'N Points':^10}")
        print("-" * 70)
        
        for result in results:
            t0, t1 = result['times']
            input_var, output_var = result['variables']
            print(f"({t0},{t1}){input_var:^15}{output_var:^15}{result['correlation']:^15.4f}{result['p_value']:^15.4e}{result['n_points']:^10}")
        
        # Generate the correlation vs time lag plot
        plot_path = plot_correlation_vs_lag(results)
        print(f"\nCorrelation vs Time Lag plot saved to: {plot_path}")