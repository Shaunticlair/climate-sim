import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

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
    global_sensitivity_sq = np.nansum(sensitivity**2)

    #print(global_sensitivity_sq)
    
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
        cumulative_sensitivity[i] = np.nansum(sensitivity[mask]**2)
        # Count number of points within this radius
        num_points_per_radius[i] = np.nansum(mask)
    
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
            abs_max = np.nanmax(np.abs(masked_sensitivity.compressed()))
            #print(f"Using color limits [{-abs_max}, {abs_max}]")
            
            fig = plt.gcf()
            ax = plt.gca()
            im = ax.imshow(masked_sensitivity, cmap=cmap, origin='lower', vmin=-abs_max, vmax=abs_max)
        else:
            # Fallback if no wetmask is available
            # Ensure symmetric color scale around zero
            abs_max = np.nanmax(np.abs(sensitivity))
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

def process_mitgcm_data_over_time(folder_path, var_name, center_lat, center_lon, days_per_level=7):
    """
    Process all time levels of MITgcm data for a specific variable and calculate effective radius.
    
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
        
    Returns:
    --------
    time_days : list
        List of time points in days
    effective_radii : list
        List of effective radii for each time point
    """
    # Create Plots directory if it doesn't exist
    Path("Plots").mkdir(exist_ok=True)
    
    # Construct the file path pattern for the variable
    file_path = Path(folder_path) / f"{var_name}.npy"
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return [], []
        
    # Load the data - it should have shape (time, lat, lon)
    sensitivity_data = np.load(file_path)
    
    # Check if we need to reshape
    if len(sensitivity_data.shape) != 3:
        print(f"Error: Expected 3D data (time, lat, lon), but got shape {sensitivity_data.shape}")
        return [], []
    
    num_time_levels = sensitivity_data.shape[0]
    print(f"Found {num_time_levels} time levels in the data")
    
    # Lists to store results
    time_days = []
    effective_radii = []
    
    # Process each time level
    for t in range(num_time_levels):
        # Extract sensitivity map for this time level
        sensitivity_map = sensitivity_data[t, :, :]
        
        # Calculate days as time lag (days from the end)
        days = (num_time_levels - 1 - t) * days_per_level
        
        # Compute effective radius
        print(f"Processing time level {t} ({days} days)")
        try:
            radius = compute_effective_radius(
                sensitivity_map, center_lat, center_lon, 
                plot=(t == 0),  # Only create plots for the first time level
                save_prefix=f"mitgcm_{var_name}_t{t}" if t == 0 else None
            )
            
            # Store results
            time_days.append(days)
            effective_radii.append(radius)
            
            print(f"Time level {t} ({days} days): Effective radius = {radius}")
            
        except Exception as e:
            print(f"Error processing time level {t}: {e}")
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.plot(time_days, effective_radii, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Time Lag (days)', fontsize=14)

    plt.ylabel('Effective Radius (pixels)', fontsize=14)
    plt.title(f'Effective Radius vs Time Lag for {var_name}', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # If we have enough data points, set x-axis ticks at sensible intervals
    if time_days:
        max_time = max(time_days)
        if max_time > 100:
            tick_interval = 100
        elif max_time > 50:
            tick_interval = 20
        else:
            tick_interval = 10
            
        plt.xticks(np.arange(0, max(time_days) + tick_interval, tick_interval))
    
    # Save the plot
    output_file = f"Plots/mitgcm_{var_name}_effective_radius_vs_time.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_file}")
    
    return time_days, effective_radii

def test_single_timestep(folder_path, var_name, center_lat, center_lon, time_level=0):
    """
    Test processing a single timestep from a MITgcm data file with full visualization.
    
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
    time_level : int, optional
        Time level to process (default: 0, the first timestep)
    
    Returns:
    --------
    effective_radius : float
        The effective radius for the selected time level
    """
    print(f"\n===== TESTING SINGLE TIMESTEP =====")
    print(f"Variable: {var_name}")
    print(f"Location: ({center_lat}, {center_lon})")
    print(f"Time level: {time_level}")
    
    # Construct the file path for the variable
    file_path = Path(folder_path) / f"{var_name}.npy"
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None
    
    # Load the data
    try:
        sensitivity_data = np.load(file_path)
        
        # Check dimensions
        if len(sensitivity_data.shape) != 3:
            print(f"Error: Expected 3D data (time, lat, lon), but got shape {sensitivity_data.shape}")
            return None
        
        num_time_levels = sensitivity_data.shape[0]
        if time_level >= num_time_levels:
            print(f"Error: Requested time level {time_level} is out of range (max: {num_time_levels-1})")
            return None
            
        # Extract the single timestep
        sensitivity_map = sensitivity_data[time_level, :, :]
        
        # Compute effective radius with all plots
        radius = compute_effective_radius(
            sensitivity_map, center_lat, center_lon, 
            plot=True,  # Create all visualizations
            save_prefix=f"test_mitgcm_{var_name}_t{time_level}"
        )
        
        print(f"Effective radius at time level {time_level}: {radius} pixels")
        return radius
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

if __name__ == "__main__":
    # Define parameters
    variables = ['hfds',]# 'tauuo', 'tauvo']  # MITgcm variables to process
    locations = {
        '(90,180)': (90, 180),  # Equatorial Pacific (lat, lon)
        #'(126,324)': (126, 324),  # North Atlantic Ocean
        #'(131,289)': (131, 289)  # Nantucket
    }
    
    # Global settings
    days_per_level = 7  # Each time level represents 7 days
    
    # First run the single timestep test for one location and variable
    test_location = '(90,180)'  # Equatorial Pacific
    test_var = 'hfds'
    center_lat, center_lon = locations[test_location]
    folder_path = f'Converted_NETcdf/{test_location}'
    
    # Run the test (processes first timestep with full visualization)
    test_single_timestep(folder_path, test_var, center_lat, center_lon)
    
    # Now process all data
    print("\n===== PROCESSING ALL DATA =====")
    
    # Process each variable for each location
    for loc_name, coords in locations.items():
        center_lat, center_lon = coords
        folder_path = f'Converted_NETcdf/{loc_name}'
        
        print(f"\nProcessing location: {loc_name} ({center_lat}, {center_lon})")
        
        for var_name in variables:
            print(f"\nProcessing variable: {var_name}")
            
            # Process this variable's data over time
            time_days, radii = process_mitgcm_data_over_time(
                folder_path, var_name, center_lat, center_lon, days_per_level
            )
            
            if time_days:  # If we got valid results
                # Also save the raw data for later use
                result_data = np.column_stack((time_days, radii))
                np.savetxt(
                    f"Plots/mitgcm_{var_name}_effective_radius_data_{loc_name.replace(',', '_')}.csv", 
                    result_data, 
                    delimiter=',', 
                    header='time_days,effective_radius',
                    comments=''
                )
                print(f"Raw data saved to CSV file")
                
    print("\nAll processing complete!")