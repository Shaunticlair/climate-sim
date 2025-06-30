import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import the MITgcm-specific function from the specialized file
try:
    import plot_effective_radii_over_time_mitgcm
except ImportError:
    plot_effective_radii_over_time_mitgcm = None
    print("Warning: Could not import plot_effective_radii_over_time_mitgcm")

# Global styling
x = 1.5
plt.rcParams['font.size'] = 24/2*x
plt.rcParams['axes.titlesize'] = 32/2*x
plt.rcParams['axes.labelsize'] = 28/2*x
plt.rcParams['xtick.labelsize'] = 24/2*x
plt.rcParams['ytick.labelsize'] = 24/2*x
plt.rcParams['legend.fontsize'] = 24/2*x
plt.rcParams['axes.titlepad'] = 22
plt.rcParams['figure.subplot.wspace'] = -0.5
plt.rcParams['figure.subplot.hspace'] = -0

def compute_effective_radius(sensitivity_map, center_lat, center_lon, max_radius=None, 
                           plot=False, save_prefix=None, map_dims=None):
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
    map_dims : list, optional
        [xmin, xmax, ymin, ymax] defining the plot dimensions. If None, will use the full sensitivity map.
        
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
    normalized_density = sensitivity_density
    
    # Find the radius at which we reach 50% of global sensitivity
    if all(normalized_cumulative < 0.5):
        print("Warning: No radius reaches 50% of global sensitivity within the maximum radius")
        effective_radius = max_radius
    else:
        # Find the first radius that reaches or exceeds 50%
        effective_index = np.argmax(normalized_cumulative >= 0.5)
        effective_radius = radii[effective_index]
    
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
            
            # Adjust the plot dimensions if map_dims is provided
            if map_dims is not None:
                xmin, xmax, ymin, ymax = map_dims
                plot_sensitivity = masked_sensitivity[xmin:xmax, ymin:ymax]
                # Adjust center coordinates for the cropped plot
                plot_center_lat = center_lat - xmin
                plot_center_lon = center_lon - ymin
                im = ax.imshow(plot_sensitivity, cmap=cmap, origin='lower', vmin=-abs_max, vmax=abs_max)
                circle = plt.Circle((plot_center_lon, plot_center_lat), effective_radius, fill=False, color='black', linestyle='--', linewidth=1)
            else:
                im = ax.imshow(masked_sensitivity, cmap=cmap, origin='lower', vmin=-abs_max, vmax=abs_max)
                circle = plt.Circle((center_lon, center_lat), effective_radius, fill=False, color='black', linestyle='--', linewidth=1)
            
            ax.add_patch(circle)
        else:
            # Fallback if no wetmask is available
            # Ensure symmetric color scale around zero
            abs_max = np.max(np.abs(sensitivity))
            fig = plt.gcf()
            ax = plt.gca()
            
            # Adjust the plot dimensions if map_dims is provided
            if map_dims is not None:
                xmin, xmax, ymin, ymax = map_dims
                plot_sensitivity = sensitivity[xmin:xmax, ymin:ymax]
                # Adjust center coordinates for the cropped plot
                plot_center_lat = center_lat - xmin
                plot_center_lon = center_lon - ymin
                im = ax.imshow(plot_sensitivity, cmap='RdBu_r', origin='lower', vmin=-abs_max, vmax=abs_max)
                circle = plt.Circle((plot_center_lon, plot_center_lat), effective_radius, fill=False, color='black', linestyle='--', linewidth=1)
            else:
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

def track_effective_radius_over_time(sensitivity_files, center_lat, center_lon, times=None, 
                                   end_time=72, plot=True, save_file=None, line_style='solid',
                                   colors=['blue'], alternating_colors=False):
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
    line_style : str, optional
        Style for the lines ('solid', 'alternating')
    colors : list, optional
        List of colors to use for plotting
    alternating_colors : bool, optional
        Whether to alternate colors between line segments
        
    Returns:
    --------
    effective_radii : list
        List of effective radii for each time point
    time_lags : list
        List of time lags
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
        
        if alternating_colors and len(colors) >= 2:
            # Draw lines with alternating colors
            for i in range(len(time_lags)-1):
                line_color = colors[1] if i % 2 == 1 else colors[0]
                plt.plot(time_lags[i:i+2], effective_radii[i:i+2], '-', color=line_color, linewidth=2)
            
            # Plot all points with first color
            plt.plot(time_lags, effective_radii, 'o', color=colors[0], markersize=6)
        else:
            # Standard single-color plotting
            color = colors[0] if colors else 'blue'
            plt.plot(time_lags, effective_radii, 'o-', color=color, linewidth=2, markersize=6)
        
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

def process_mitgcm_data_over_time(folder_path, var_name, center_lat, center_lon, 
                                days_per_level=7, max_days=None):
    """
    Process all time levels of MITgcm data for a specific variable and calculate effective radius.
    Filter to only include time lags up to max_days if specified.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing MITgcm data files
    var_name : str
        Variable name for the sensitivity files (e.g., 'hfds', 'tauuo', 'tauvo')
    center_lat : int
        Latitude index of the center point
    center_lon : int
        Longitude index of the center point
    days_per_level : int, optional
        Number of days per time level (default: 7)
    max_days : int, optional
        Maximum number of days to include (default: None for no limit)
        
    Returns:
    --------
    time_days : list
        List of time points in days
    effective_radii : list
        List of effective radii for each time point
    """
    if plot_effective_radii_over_time_mitgcm is None:
        print("Error: MITgcm processing module not available")
        return [], []
    
    # Use the specialized MITgcm function
    time_days, effective_radii = plot_effective_radii_over_time_mitgcm.process_mitgcm_data_over_time(
        folder_path, var_name, center_lat, center_lon, days_per_level
    )
    
    # Apply max_days filter if specified
    if max_days is not None:
        filtered_data = [(d, r) for d, r in zip(time_days, effective_radii) if d <= max_days]
        if filtered_data:
            time_days, effective_radii = zip(*filtered_data)
        else:
            time_days, effective_radii = [], []
    
    return list(time_days), list(effective_radii)

def track_samudra_effective_radius_over_time(sensitivity_files, center_lat, center_lon, 
                                           times=None, end_time=292, max_days=None):
    """
    Track how the effective radius changes over time for a series of Samudra sensitivity maps.
    Filter to only include time lags up to max_days if specified.
    
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
    max_days : int, optional
        Maximum number of days to include (default: None for no limit)
        
    Returns:
    --------
    time_lags : list
        List of time lags in days
    effective_radii : list
        List of effective radii for each time point
    """
    # Create time points if not provided
    if times is None:
        times = list(range(len(sensitivity_files)))
    
    # Calculate time lags (time from output) in days (5 days per time step)
    time_lags = [(end_time - t) * 5 for t in times]
    
    # Filter files and times to only include those within max_days
    filtered_files = []
    filtered_radii = []
    filtered_lags = []
    
    for i, (file_path, lag) in enumerate(zip(sensitivity_files, time_lags)):
        if max_days is None or lag <= max_days:
            if Path(file_path).exists():
                # Load the sensitivity map
                sensitivity_map = np.load(file_path)
                if len(sensitivity_map.shape) > 2:
                    sensitivity_map = sensitivity_map.reshape(180, 360)
                
                # Compute effective radius
                radius = compute_effective_radius(sensitivity_map, center_lat, center_lon, plot=False)
                
                filtered_files.append(file_path)
                filtered_radii.append(radius)
                filtered_lags.append(lag)
    
    return filtered_lags, filtered_radii

def compare_mitgcm_samudra(var_name, center_lat, center_lon, ch_in=154, ch_out=76, 
                          end_time=292, max_days=None, save_file=None):
    """
    Create a combined plot of effective radius vs time for both MITgcm and Samudra data.
    
    Parameters:
    -----------
    var_name : str
        Variable name for the sensitivity files (e.g., 'hfds', 'tauuo', 'tauvo')
    center_lat : int
        Latitude index of the center point
    center_lon : int
        Longitude index of the center point
    ch_in : int
        Input channel number for Samudra
    ch_out : int
        Output channel number for Samudra
    end_time : int
        End time for Samudra data
    max_days : int, optional
        Maximum number of days to include in the plot
    save_file : str, optional
        File path to save the plot
        
    Returns:
    --------
    tuple
        (mitgcm_time_days, mitgcm_radii, samudra_time_lags, samudra_radii)
    """
    # Create Plots directory if it doesn't exist
    Path("Plots").mkdir(exist_ok=True)
    
    # Part 1: Process MITgcm data
    mitgcm_folder = f'Converted_NETcdf/({center_lat},{center_lon})'
    mitgcm_time_days, mitgcm_radii = process_mitgcm_data_over_time(
        mitgcm_folder, var_name, center_lat, center_lon, 
        days_per_level=7, max_days=max_days
    )
    
    # Part 2: Process Samudra data
    samudra_folder = 'MITgcm_Replication/'
    samudra_times = [2*i for i in range(end_time//2)]  # Time points to load
    
    # Create list of file paths for Samudra data
    samudra_files = [
        f"{samudra_folder}avg_sensitivity_chin[{ch_in}]_chout[{ch_out}]_loc[{center_lat},{center_lon}]_t[{t},{end_time}-297].npy" 
        for t in samudra_times
    ]
    
    # Filter to only existing files
    existing_files = [f for f in samudra_files if Path(f).exists()]
    existing_times = [samudra_times[i] for i, f in enumerate(samudra_files) if Path(f).exists()]
    
    print(f"Found {len(existing_files)} Samudra files")
    
    # If we have Samudra data, process it
    if existing_files:
        samudra_time_lags, samudra_radii = track_samudra_effective_radius_over_time(
            existing_files, center_lat, center_lon, existing_times, 
            end_time, max_days=max_days
        )
    else:
        samudra_time_lags, samudra_radii = [], []
    
    # Part 3: Create combined plot
    plt.figure(figsize=(12, 8))
    
    # Plot MITgcm data
    if mitgcm_time_days:
        plt.plot(mitgcm_time_days, mitgcm_radii, 'b-', linewidth=2, marker='o', markersize=6, label='MITgcm')
    
    # Plot Samudra data
    if samudra_time_lags:
        plt.plot(samudra_time_lags, samudra_radii, 'r-', linewidth=2, marker='s', markersize=6, label='Samudra')
    
    # Add labels and legend
    plt.xlabel('Time Lag (days)')
    plt.ylabel('Effective Radius (pixels)')
    
    # Set title based on max_days
    if max_days:
        plt.title(f'Effective Radius vs Time Lag - {var_name} (up to {max_days} days)')
        # Set the x-axis limit to exactly max_days
        plt.xlim(0, max_days)
        # Set tick interval based on max_days
        if max_days > 50:
            tick_interval = 20
        elif max_days > 20:
            tick_interval = 10
        else:
            tick_interval = 5
        plt.xticks(np.arange(0, max_days + tick_interval, tick_interval))
    else:
        plt.title(f'Effective Radius vs Time Lag - {var_name}')
        # Set reasonable x-axis ticks for full range
        if mitgcm_time_days or samudra_time_lags:
            all_times = mitgcm_time_days + samudra_time_lags
            if all_times:
                max_time = max(all_times)
                if max_time > 400:
                    tick_interval = 200
                elif max_time > 200:
                    tick_interval = 50
                elif max_time > 100:
                    tick_interval = 20
                else:
                    tick_interval = 10
                plt.xticks(np.arange(0, max_time + tick_interval, tick_interval))
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=24)
    
    # Save the plot
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Combined plot saved to {save_file}")
    else:
        plt.show()
    
    return mitgcm_time_days, mitgcm_radii, samudra_time_lags, samudra_radii

if __name__ == "__main__":
    
    demo1, demo2, demo3,  demo5, demo6 = False, False, False,  False, False

    ## demo4 was removed; it was meant to replace the merge file, but it failed

    if demo1:

        # =================================================================
        # DEMO 1: Basic single sensitivity map with full visualization
        # =================================================================
        print("="*60)
        print("DEMO 1: Basic Single Sensitivity Map")
        print("="*60)
        
        in_time = 4
        end_time = 12
        center_lat, center_lon = 90, 180  # Equatorial Pacific
        sample_file = f"adjoint_arrays/Equatorial_Pacific/chunk_sensitivity_chin[76]_chout[76]_t[{in_time},{end_time}].npy"

        if Path(sample_file).exists():
            print(f"Processing file: {sample_file}")
            sensitivity_map = np.load(sample_file)
            
            # Reshape if needed
            if len(sensitivity_map.shape) > 2:
                sensitivity_map = sensitivity_map.reshape(180, 360)
            
            # Compute effective radius with plot saving
            delta = 50
            map_dims = [90 - delta, 90 + delta, 180 - delta, 180 + delta]
            
            radius = compute_effective_radius(sensitivity_map, center_lat, center_lon, 
                                            plot=True, save_prefix=f"demo1_chin[76]_chout[76]_t[{in_time},{end_time}]",
                                            map_dims=map_dims)
            print(f"Effective radius: {radius} pixels")
        else:
            print(f"File not found: {sample_file}")

    if demo2:
        # =================================================================
        # DEMO 2: Basic time series tracking (original functionality)
        # =================================================================
        print("\n" + "="*60)
        print("DEMO 2: Basic Time Series Tracking")
        print("="*60)
        
        end_time = 24
        times = [i for i in range(24)]
        ch_out, ch_in = 76, 76
        center_lat, center_lon = 90, 180
        
        folder = 'adjoint_arrays/Equatorial_Pacific/'
        files = [f"{folder}avg_sensitivity_chin[{ch_in}]_chout[{ch_out}]_t[{t},{end_time}].npy" 
                for t in times]
        
        # Check which files exist
        existing_files = [f for f in files if Path(f).exists()]
        existing_times = [times[i] for i, f in enumerate(files) if Path(f).exists()]
        
        if len(existing_files) > 1:
            print(f"Tracking effective radius over {len(existing_files)} time points")
            save_path = f"Plots/demo2_effective_radius_over_time_[{center_lat},{center_lon}]_endtime[{end_time}].png"
            radii, lags = track_effective_radius_over_time(
                existing_files, center_lat, center_lon, 
                existing_times, end_time=end_time,
                save_file=save_path
            )
            print("Effective radii over time:", radii[:5], "..." if len(radii) > 5 else "")
            print("Time lags:", lags[:5], "..." if len(lags) > 5 else "")
        else:
            print("Not enough files found for time series analysis")
        
    if demo3:
        # =================================================================
        # DEMO 3: Alternating color line style
        # =================================================================
        print("\n" + "="*60)
        print("DEMO 3: Alternating Color Line Style")
        print("="*60)
        
        if len(existing_files) > 1:
            print(f"Creating alternating color plot for {len(existing_files)} time points")
            save_path = f"Plots/demo3_alternating_colors_[{center_lat},{center_lon}]_endtime[{end_time}].png"
            radii, lags = track_effective_radius_over_time(
                existing_files, center_lat, center_lon, 
                existing_times, end_time=end_time,
                save_file=save_path,
                colors=['black', 'red'],
                alternating_colors=True
            )
            print(f"Alternating color plot saved to {save_path}")
        else:
            print("Not enough files found for alternating color demo")
        

    if demo5:
        # =================================================================
        # DEMO 5: MITgcm vs Samudra comparison (full range)
        # =================================================================
        print("\n" + "="*60)
        print("DEMO 5: MITgcm vs Samudra Comparison (Full Range)")
        print("="*60)
        
        # Define parameters
        variables = ['hfds', 'tauuo', 'tauvo']
        locations = {
            '(90,180)': (90, 180),  # Equatorial Pacific
            # Add more locations as needed
        }
        
        # Channel mappings
        channel_mapping = {
            'hfds': 156,
            'tauuo': 154,
            'tauvo': 155
        }
        
        # Process one variable for one location as demo
        for loc_name, coords in list(locations.items())[:1]:  # Just first location for demo
            center_lat, center_lon = coords
            
            for var_name in variables[:1]:  # Just first variable for demo
                print(f"\nProcessing {var_name} at location {loc_name}")
                
                # Get input channel for Samudra
                ch_in = channel_mapping.get(var_name, 156)
                ch_out = 76  # Output channel (zos)
                
                # Create comparison plot
                location_str = f"{center_lat}_{center_lon}".replace(",", "_")
                save_path = f"Plots/demo5_comparison_{var_name}_{location_str}.png"
                
                mitgcm_days, mitgcm_radii, samudra_lags, samudra_radii = compare_mitgcm_samudra(
                    var_name, center_lat, center_lon, ch_in, ch_out, save_file=save_path
                )
                
                if mitgcm_days or samudra_lags:
                    print(f"MITgcm data points: {len(mitgcm_days)}")
                    print(f"Samudra data points: {len(samudra_lags)}")
                else:
                    print("No data found for comparison")
                break  # Only process first variable for demo
            break  # Only process first location for demo
        
    if demo6:

        # =================================================================
        # DEMO 6: MITgcm vs Samudra comparison with day limit
        # =================================================================
        print("\n" + "="*60)
        print("DEMO 6: MITgcm vs Samudra Comparison (Limited to 100 days)")
        print("="*60)
        
        # Use same parameters but with max_days limit
        max_days = 100

        # Define parameters
        variables = ['hfds', 'tauuo', 'tauvo']
        locations = {
            '(90,180)': (90, 180),  # Equatorial Pacific
            # Add more locations as needed
        }
        
        # Channel mappings
        channel_mapping = {
            'hfds': 156,
            'tauuo': 154,
            'tauvo': 155
        }
        
        for loc_name, coords in list(locations.items())[:1]:  # Just first location for demo
            center_lat, center_lon = coords
            
            for var_name in variables[:1]:  # Just first variable for demo
                print(f"\nProcessing {var_name} at location {loc_name} (max {max_days} days)")
                
                # Get input channel for Samudra
                ch_in = channel_mapping.get(var_name, 156)
                ch_out = 76  # Output channel (zos)
                
                # Create comparison plot with day limit
                location_str = f"{center_lat}_{center_lon}".replace(",", "_")
                save_path = f"Plots/demo6_comparison_{var_name}_{location_str}_max{max_days}days.png"
                
                mitgcm_days, mitgcm_radii, samudra_lags, samudra_radii = compare_mitgcm_samudra(
                    var_name, center_lat, center_lon, ch_in, ch_out, 
                    max_days=max_days, save_file=save_path
                )
                
                if mitgcm_days or samudra_lags:
                    print(f"MITgcm data points (≤{max_days} days): {len(mitgcm_days)}")
                    print(f"Samudra data points (≤{max_days} days): {len(samudra_lags)}")
                else:
                    print("No data found for limited comparison")
                break  # Only process first variable for demo
            break  # Only process first location for demo
        
        print("\n" + "="*60)
        print("ALL DEMOS COMPLETED!")
        print("="*60)
        print("Saved plots can be found in the 'Plots/' directory")
        print("\nDemo summary:")
        print("- DEMO 1: Single sensitivity map with full visualization (Fig 5.4, 5.5a, 5.5b)")
        ## DEMO 1 works!
        print("- DEMO 2: Basic time series tracking")
        ## DEMO 2 works!
        print("- DEMO 3: Alternating color line style for DEMO 2 (Fig 5.11)")
        ## DEMO 3 works!
        print("- DEMO 4: Merged day and month timeframes")
        ## DEMO 4 doesn't really work
        print("- DEMO 5: MITgcm vs Samudra comparison (full range) (Fig 5.16b)")
        ## DEMO 5 works!
        print("- DEMO 6: MITgcm vs Samudra comparison (limited to 100 days) (Fig 5.17a)")
        ## DEMO 6 works!
        print("="*60)
