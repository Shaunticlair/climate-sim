import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def compute_effective_radius(sensitivity_map, center_lat, center_lon, max_radius=None, plot=False, save_prefix=None):
    """
    Compute the effective radius of sensitivity from a given center point.
    
    Parameters:
    -----------
    sensitivity_map : numpy.ndarray
        2D array of sensitivity values (lat, lon)
    center_lat : int
        Latitude index of the center point
    center_lon : int
        Longitude index of the center point
    max_radius : int, optional
        Maximum radius to consider. If None, will use the maximum possible radius.
    plot : bool, optional
        Whether to create a visualization of the result
    save_prefix : str, optional
        Prefix for saving plot files. If None, plots will be displayed instead of saved.
        
    Returns:
    --------
    effective_radius : float
        The radius at which cumulative sensitivity reaches 50% of global sensitivity
    """
    # Handle masked arrays
    if isinstance(sensitivity_map, np.ma.MaskedArray):
        # Replace masked values with zeros
        sensitivity = sensitivity_map.filled(0)
    else:
        sensitivity = sensitivity_map.copy()
    
    # Get dimensions
    lat_size, lon_size = sensitivity.shape
    
    # Calculate global sensitivity magnitude (sum of squared sensitivities)
    global_sensitivity_sq = np.sum(sensitivity**2)
    
    # If the global sensitivity is zero, return 0
    if global_sensitivity_sq == 0:
        print("Warning: Global sensitivity is zero")
        return 0
    
    # Set default max radius if none provided
    if max_radius is None:
        # Maximum possible radius is the distance to the farthest corner
        corners = [
            (0, 0),
            (0, lon_size-1),
            (lat_size-1, 0),
            (lat_size-1, lon_size-1)
        ]
        max_radius = max(
            np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
            for lat, lon in corners
        )
        max_radius = int(np.ceil(max_radius))
    
    # Create a distance map from the center point
    y_indices, x_indices = np.ogrid[:lat_size, :lon_size]
    distances = np.sqrt((y_indices - center_lat)**2 + (x_indices - center_lon)**2)
    
    # Calculate cumulative sensitivity for increasing radii
    radii = np.arange(1, max_radius + 1)
    cumulative_sensitivity = np.zeros(len(radii))
    num_points_per_radius = np.zeros(len(radii), dtype=int)
    
    for i, radius in enumerate(radii):
        # Create a mask for points within this radius
        mask = distances <= radius
        # Calculate sum of squared sensitivities within this radius
        cumulative_sensitivity[i] = np.sum(sensitivity[mask]**2)
        # Count number of points within this radius
        num_points_per_radius[i] = np.sum(mask)
    
    # Calculate the number of points added at each radius
    points_added = np.zeros(len(radii), dtype=int)
    points_added[0] = num_points_per_radius[0]  # First ring is just the center point
    points_added[1:] = num_points_per_radius[1:] - num_points_per_radius[:-1]  # Rest are differences
    
    # Calculate the sensitivity added at each radius
    ring_sensitivity = np.zeros(len(radii))
    ring_sensitivity[0] = cumulative_sensitivity[0]  # First ring is just the first cumulative value
    ring_sensitivity[1:] = cumulative_sensitivity[1:] - cumulative_sensitivity[:-1]  # Rest are differences
    
    # Calculate sensitivity density (sensitivity per point) in each ring
    # Avoid division by zero
    sensitivity_density = np.zeros(len(radii))
    for i in range(len(radii)):
        if points_added[i] > 0:
            sensitivity_density[i] = ring_sensitivity[i] / points_added[i]
    
    # Normalize cumulative sensitivity by global sensitivity
    normalized_cumulative = cumulative_sensitivity / global_sensitivity_sq
    
    # Normalize density for visualization (not by global, but to make the plot readable)
    max_density = np.max(sensitivity_density) if np.max(sensitivity_density) > 0 else 1
    normalized_density = sensitivity_density #/ max_density
    
    # Find the radius at which we reach 50% of global sensitivity
    if all(normalized_cumulative < 0.5):
        print("Warning: No radius reaches 50% of global sensitivity within the maximum radius")
        effective_radius = max_radius
    else:
        # Find the first radius that reaches or exceeds 50%
        effective_index = np.argmax(normalized_cumulative >= 0.5)
        effective_radius = radii[effective_index]
    
    # Plot if requested
    # Plot if requested
    if plot:
        # Create Plots directory if it doesn't exist
        Path("Plots").mkdir(exist_ok=True)
        
        # Plot 1: Sensitivity map with effective radius circle
        plt.figure(figsize=(10, 8))
        
        # Load the wetmask to show land/ocean boundaries
        drymask_path = 'drymask.npy'
        if Path(drymask_path).exists():
            drymask = np.load(drymask_path)
            wetmask = np.where(drymask == 0, 1, 0)  # Invert the mask
            
            # Create a masked sensitivity array for better visualization
            masked_sensitivity = np.ma.masked_array(
                sensitivity,
                mask=~wetmask.astype(bool)  # Invert wetmask to get drymask
            )
            
            # Create a custom colormap with black for masked values
            cmap = plt.cm.RdBu_r.copy()
            cmap.set_bad('black', 0.5)
            
            # Ensure symmetric color scale around zero
            abs_max = np.max(np.abs(masked_sensitivity.compressed()))
            
            im = plt.imshow(masked_sensitivity, cmap=cmap, origin='lower', vmin=-abs_max, vmax=abs_max)
        else:
            # Fallback if no wetmask is available
            # Ensure symmetric color scale around zero
            abs_max = np.max(np.abs(sensitivity))
            im = plt.imshow(sensitivity, cmap='RdBu_r', origin='lower', vmin=-abs_max, vmax=abs_max)
            
        circle = plt.Circle((center_lon, center_lat), effective_radius, fill=False, color='black', linestyle='--', linewidth=1)
        plt.gca().add_patch(circle)
        
        # Create colorbar with scientific notation for small values
        cbar = plt.colorbar(im, label='Sensitivity')
        cbar.formatter.set_powerlimits((-2, 2))  # Use scientific notation for values < 10^-2 or > 10^2
        cbar.update_ticks()
        
        plt.title(f'Sensitivity Map with Effective Radius ({effective_radius} pixels)')
        plt.xlabel('Longitude Index')
        plt.ylabel('Latitude Index')
        
        if save_prefix:
            plt.savefig(f'Plots/{save_prefix}_sensitivity_map.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        # Plot 2: Cumulative sensitivity vs radius (CDF)
        plt.figure(figsize=(10, 6))
        plt.plot(radii, normalized_cumulative, 'b-', linewidth=2)
        plt.axhline(0.5, color='r', linestyle='--', label='50% Threshold')
        plt.axvline(effective_radius, color='g', linestyle='--', label=f'Effective Radius = {effective_radius}')
        plt.xlabel('Radius (pixels)')
        plt.ylabel('Normalized Cumulative Sensitivity')
        plt.title('Cumulative Sensitivity vs Radius (CDF)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_prefix:
            plt.savefig(f'Plots/{save_prefix}_cumulative_sensitivity.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        # Plot 3: Sensitivity density at each radius (PDF)
        plt.figure(figsize=(10, 6))
        plt.plot(radii, normalized_density, 'r-', linewidth=2)
        plt.axvline(effective_radius, color='g', linestyle='--', label=f'Effective Radius = {effective_radius}')
        plt.xlabel('Radius (pixels)')
        plt.ylabel('Sensitivity Contribution per Pixel')
        plt.title('Sensitivity Density at Each Radius')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Apply scientific notation to y-axis if needed
        ax = plt.gca()
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        
        if save_prefix:
            plt.savefig(f'Plots/{save_prefix}_sensitivity_density.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    return effective_radius

def track_effective_radius_over_time(sensitivity_files, center_lat, center_lon, times=None, end_time=72, plot=True, save_file=None):
    """
    Track how the effective radius changes over time for a series of sensitivity maps.
    
    Parameters:
    -----------
    sensitivity_files : list
        List of file paths to sensitivity maps
    center_lat : int
        Latitude index of the center point
    center_lon : int
        Longitude index of the center point
    times : list, optional
        List of time values corresponding to each file. If None, will use indices.
    end_time : int, optional
        The end time value for calculating time lag
    plot : bool, optional
        Whether to create a visualization of the results
    save_file : str, optional
        File path to save the plot. If None, plot will be displayed instead of saved.
        
    Returns:
    --------
    effective_radii : list
        List of effective radii for each time point
    """
    effective_radii = []
    
    # Process each sensitivity file
    for file_path in sensitivity_files:
        # Load the sensitivity map
        sensitivity_map = np.load(file_path)
        if len(sensitivity_map.shape) > 2:
            sensitivity_map = sensitivity_map.reshape(180, 360)
        
        # Compute effective radius
        radius = compute_effective_radius(sensitivity_map, center_lat, center_lon)
        effective_radii.append(radius)
    
    # Create time points if not provided
    if times is None:
        times = list(range(len(sensitivity_files)))
    
    # Calculate time lags (time from output)
    time_lags = [end_time - t for t in times]
    
    # Plot if requested
    if plot:
        # Create Plots directory if it doesn't exist
        Path("Plots").mkdir(exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_lags, effective_radii, 'bo-', linewidth=2)
        plt.xlabel('Time Lag (steps from output)')
        plt.ylabel('Effective Radius (pixels)')
        plt.title('Effective Radius of Sensitivity vs Time Lag')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_file:
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    return effective_radii, time_lags

# Example usage:
if __name__ == "__main__":
    # Example: Process a single sensitivity map
    sample_file = "adjoint_arrays/Equatorial_Pacific/chunk_sensitivity_chin[76]_chout[76]_t[54,72].npy"
    
    if Path(sample_file).exists():
        print(f"Processing file: {sample_file}")
        sensitivity_map = np.load(sample_file)
        
        # Reshape if needed
        if len(sensitivity_map.shape) > 2:
            sensitivity_map = sensitivity_map.reshape(180, 360)
        
        # Compute effective radius with plot saving
        center_lat, center_lon = 90, 180  # Equatorial Pacific
        radius = compute_effective_radius(sensitivity_map, center_lat, center_lon, 
                                         plot=True, save_prefix="sample_file")
        print(f"Effective radius: {radius} pixels")
        
        # Track over time (example with multiple files)
        end_time = 72  # The end time for calculating time lag
        times = [6*i for i in range(0, 12)]
        
        files = [f"adjoint_arrays/Equatorial_Pacific/chunk_sensitivity_chin[76]_chout[76]_t[{t},72].npy" 
                 for t in times]
        
        # Check which files exist
        existing_files = [f for f in files if Path(f).exists()]
        existing_times = [times[i] for i, f in enumerate(files) if Path(f).exists()]

        print(existing_files)
        
        if len(existing_files) > 1:
            print(f"Tracking effective radius over {len(existing_files)} time points")
            save_path = "Plots/effective_radius_over_time.png"
            radii, lags = track_effective_radius_over_time(existing_files, center_lat, center_lon, 
                                                         existing_times, end_time=end_time,
                                                         save_file=save_path)
            print("Effective radii over time:", radii)
            print("Time lags:", lags)
            print(f"Plot saved to {save_path}")
    else:
        print(f"File not found: {sample_file}")
        print("This script requires sensitivity maps in the current directory to run.")
        print("Example files: chunk_sensitivity_chin[76]_chout[76]_t[0,10].npy")