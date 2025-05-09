import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import the compute_effective_radius function from the existing script
import plot_effective_radii_over_time
import plot_effective_radii_over_time_mitgcm

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
            radius = plot_effective_radii_over_time_mitgcm.compute_effective_radius(
                sensitivity_map, center_lat, center_lon, 
                plot=False  # Don't create plots for individual time levels
            )
            
            # Store results
            time_days.append(days)
            effective_radii.append(radius)
            
            print(f"Time level {t} ({days} days): Effective radius = {radius}")
            
        except Exception as e:
            print(f"Error processing time level {t}: {e}")
    
    return time_days, effective_radii

def track_samudra_effective_radius_over_time(sensitivity_files, center_lat, center_lon, times=None, end_time=292):
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
        
    Returns:
    --------
    time_lags : list
        List of time lags in days
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
        radius = plot_effective_radii_over_time.compute_effective_radius(sensitivity_map, center_lat, center_lon, plot=False)
        effective_radii.append(radius)
    
    # Create time points if not provided
    if times is None:
        times = list(range(len(sensitivity_files)))
    
    # Calculate time lags (time from output) in days (5 days per time step)
    time_lags = [(end_time - t) * 5 for t in times]
    
    return time_lags, effective_radii

def combined_plot(var_name, center_lat, center_lon, ch_in=154, ch_out=76, end_time=292):
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
    """
    # Create Plots directory if it doesn't exist
    Path("Plots").mkdir(exist_ok=True)
    
    # Part 1: Process MITgcm data
    mitgcm_folder = f'Converted_NETcdf/({center_lat},{center_lon})'
    mitgcm_time_days, mitgcm_radii = process_mitgcm_data_over_time(
        mitgcm_folder, var_name, center_lat, center_lon, days_per_level=7
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
            existing_files, center_lat, center_lon, existing_times, end_time
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
    plt.xlabel('Time Lag (days)', fontsize=14)
    plt.ylabel('Effective Radius (pixels)', fontsize=14)
    plt.title(f'Effective Radius vs Time Lag - {var_name}', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Set reasonable x-axis ticks
    if mitgcm_time_days or samudra_time_lags:
        all_times = mitgcm_time_days + samudra_time_lags
        if all_times:
            max_time = max(all_times)
            if max_time > 100:
                tick_interval = 100
            elif max_time > 50:
                tick_interval = 50
            else:
                tick_interval = 10
                
            plt.xticks(np.arange(0, max_time + tick_interval, tick_interval))
    
    # Save the plot
    location_str = f"{center_lat}_{center_lon}".replace(",", "_")
    output_file = f"Plots/combined_{var_name}_effective_radius_{location_str}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"Combined plot saved to {output_file}")
    
    # Also save the raw data for reference
    with open(f"Plots/combined_{var_name}_data_{location_str}.csv", 'w') as f:
        f.write("source,time_lag,effective_radius\n")
        for t, r in zip(mitgcm_time_days, mitgcm_radii):
            f.write(f"MITgcm,{t},{r}\n")
        for t, r in zip(samudra_time_lags, samudra_radii):
            f.write(f"Samudra,{t},{r}\n")
    
    return mitgcm_time_days, mitgcm_radii, samudra_time_lags, samudra_radii

if __name__ == "__main__":
    # Define parameters
    variables = ['hfds', 'tauuo', 'tauvo']  # Variables to process
    locations = {
        '(90,180)': (90, 180),  # Equatorial Pacific (lat, lon)
        #'(126,324)': (126, 324),  # North Atlantic Ocean
        #'(131,289)': (131, 289)  # Nantucket
    }
    
    # Channel mappings
    channel_mapping = {
        'hfds': 156,
        #'tauuo': 154,
        #'tauvo': 155
    }
    
    # Process each variable for each location
    for loc_name, coords in locations.items():
        center_lat, center_lon = coords
        
        print(f"\nProcessing location: {loc_name} ({center_lat}, {center_lon})")
        
        for var_name in variables:
            print(f"\nProcessing variable: {var_name}")
            
            # Get input channel for Samudra
            ch_in = channel_mapping[var_name]  # Default to hfds channel if not found
            ch_out = 76  # Output channel (zos)
            
            # Create combined plot
            combined_plot(var_name, center_lat, center_lon, ch_in, ch_out)
    
    print("\nAll processing complete!")