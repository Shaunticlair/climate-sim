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
            
            fig = plt.gcf()
            ax = plt.gca()
            im = ax.imshow(masked_sensitivity, cmap=cmap, origin='lower', vmin=-abs_max, vmax=abs_max)
        else:
            # Fallback if no wetmask is available
            # Ensure symmetric color scale around zero
            abs_max = np.max(np.abs(sensitivity))
            fig = plt.gcf()
            ax = plt.gca()
            im = ax.imshow(sensitivity, cmap='RdBu_r', origin='lower', vmin=-abs_max, vmax=abs_max)
            
        circle = plt.Circle((center_lon, center_lat), effective_radius, fill=False, color='black', linestyle='--', linewidth=1)
        ax.add_patch(circle)

        # Create colorbar with scientific notation for small values and match plot height
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, label='Sensitivity')
        cbar.formatter.set_powerlimits((-2, 2))  # Use scientific notation for values < 10^-2 or > 10^2
        cbar.update_ticks()

        # Properly position labels on bottom and left sides
        ax.set_xlabel('Longitude Index')
        ax.set_ylabel('Latitude Index')
        ax.xaxis.set_label_position('bottom')
        ax.yaxis.set_label_position('left')
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()

        ax.set_title(f'Sensitivity Map with Effective Radius ({effective_radius} pixels)')

        # Add more room at the bottom for the x-axis labels and title
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Add padding at the bottom
        
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
    # Define center point coordinates (Equatorial Pacific)
    center_lat, center_lon = 90, 180
    
    # First: Process a single sensitivity map as an example
    in_time = 0
    sample_file = f"adjoint_arrays/Equatorial_Pacific/chunk_sensitivity_chin[76]_chout[76]_t[{in_time},12].npy"

    if Path(sample_file).exists():
        print(f"Processing file: {sample_file}")
        sensitivity_map = np.load(sample_file)
        
        # Reshape if needed
        if len(sensitivity_map.shape) > 2:
            sensitivity_map = sensitivity_map.reshape(180, 360)
        
        # Compute effective radius with plot saving
        radius = compute_effective_radius(sensitivity_map, center_lat, center_lon, 
                                         plot=True, save_prefix=f"chin[76]_chout[76]_t[{in_time},12]")
        print(f"Effective radius: {radius} pixels")
    else:
        print(f"File not found: {sample_file}")
        print("This script requires sensitivity maps in the current directory to run.")
        print("Example files: chunk_sensitivity_chin[76]_chout[76]_t[0,10].npy")
    
    # Second: Track effective radius over time with multiple files
    month_end_time = 72  # The end time for calculating time lag
    day_end_time = 12    # The end time for calculating time lag
    
    # Define time points
    day_times =  [6, 8, 10] #[0, 2, 4,
    month_times = [6*i for i in range(0, 12)]
    
    # Set channel numbers
    ch_out, ch_in = 76, 76
    
    # Define file paths
    day_files = [f"adjoint_arrays/Equatorial_Pacific/chunk_sensitivity_chin[{ch_in}]_chout[{ch_out}]_t[{t},{day_end_time}].npy"
                for t in day_times]
    month_files = [f"adjoint_arrays/Equatorial_Pacific/chunk_sensitivity_chin[{ch_in}]_chout[{ch_out}]_t[{t},{month_end_time}].npy" 
                for t in month_times]
    
    # Create two separate lists to handle different end times properly
    day_existing_files = [f for f in day_files if Path(f).exists()]
    day_existing_times = [t for i, t in enumerate(day_times) if Path(day_files[i]).exists()]
    
    month_existing_files = [f for f in month_files if Path(f).exists()]
    month_existing_times = [t for i, t in enumerate(month_times) if Path(month_files[i]).exists()]
    
    print(f"Found {len(day_existing_files)} day files and {len(month_existing_files)} month files")
    
    # Process day files
    if len(day_existing_files) > 1:
        print(f"Tracking effective radius over {len(day_existing_files)} day time points")
        save_path = f"Plots/effective_radius_day_[{center_lat},{center_lon}]_endtime[{day_end_time}]_ch[{ch_out}]_ch[{ch_in}].png"
        
        day_radii, day_lags = track_effective_radius_over_time(
            day_existing_files, center_lat, center_lon, 
            day_existing_times, end_time=day_end_time,
            save_file=save_path
        )
        
        print("Day effective radii:", day_radii)
        print("Day time lags:", day_lags)
    
    # Process month files
    if len(month_existing_files) > 1:
        print(f"Tracking effective radius over {len(month_existing_files)} month time points")
        save_path = f"Plots/effective_radius_month_[{center_lat},{center_lon}]_endtime[{month_end_time}]_ch[{ch_out}]_ch[{ch_in}].png"
        
        month_radii, month_lags = track_effective_radius_over_time(
            month_existing_files, center_lat, center_lon, 
            month_existing_times, end_time=month_end_time,
            save_file=save_path
        )
        
        print("Month effective radii:", month_radii)
        print("Month time lags:", month_lags)
    
    # If you want to combine them into a single plot, we need to handle the different end times
    if len(day_existing_files) > 0 and len(month_existing_files) > 0:
        print("Creating combined plot with both day and month data")
        
        # Combine files and times
        all_files = day_existing_files + month_existing_files
        
        # Convert to a common time scale - normalize to days from final output
        # For day files: lag = day_end_time - time
        # For month files: lag = month_end_time - time
        normalized_times = day_existing_times + month_existing_times
        
        # Use a special flag to indicate different end times
        # The track_effective_radius_over_time function needs to be modified to handle this
        save_path = f"Plots/effective_radius_combined_[{center_lat},{center_lon}]_ch[{ch_out}]_ch[{ch_in}].png"
        
        # Process files separately then combine for plotting
        day_radii, day_lags = [], []
        month_radii, month_lags = [], []
        
        # Process day files
        for i, file_path in enumerate(day_existing_files):
            sensitivity_map = np.load(file_path)
            if len(sensitivity_map.shape) > 2:
                sensitivity_map = sensitivity_map.reshape(180, 360)
            
            radius = compute_effective_radius(sensitivity_map, center_lat, center_lon)
            day_radii.append(radius)
            day_lags.append(day_end_time - day_existing_times[i])
        
        # Process month files
        for i, file_path in enumerate(month_existing_files):
            sensitivity_map = np.load(file_path)
            if len(sensitivity_map.shape) > 2:
                sensitivity_map = sensitivity_map.reshape(180, 360)
            
            radius = compute_effective_radius(sensitivity_map, center_lat, center_lon)
            month_radii.append(radius)
            month_lags.append(month_end_time - month_existing_times[i])
        
        # Create a custom combined plot
        plt.figure(figsize=(12, 7))
        
        # Plot day data with one color/marker
        plt.plot(day_lags, day_radii, 'bo-', linewidth=2, markersize=8)
        plt.plot(month_lags, month_radii, 'bo-', linewidth=2, markersize=8)
        # Plot month data with another color/marker
        #plt.plot(month_lags, month_radii, 'rs-', linewidth=2, markersize=8, label=f'Month scale (end time = {month_end_time})')
        
        plt.xlabel('Time Lag (steps from output)', fontsize=12)
        plt.ylabel('Effective Radius (pixels)', fontsize=12)
        plt.title('Effective Radius of Sensitivity vs Time Lag', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Combined plot saved to {save_path}")