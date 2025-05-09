import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
import misc

# ===== CONFIGURATION (MODIFY THESE VALUES) =====
# Directory to save the output plot
output_dir = "Plots"
samudra_folder = 'MITgcm_Replication'
mitgcm_folder = 'Converted_NETcdf'
# ================================================

def calculate_magnitude(matrix):
    """Calculate the vector magnitude (L2 norm) of a matrix."""
    # Handle masked arrays if present
    if isinstance(matrix, np.ma.MaskedArray):
        # Only consider unmasked elements
        return np.sqrt(np.nansum(matrix.compressed() ** 2))
    else:
        # Regular array - use all elements
        return np.sqrt(np.nansum(matrix ** 2))

def load_mitgcm_sensitivities(var_name, location, days_per_level=7):
    """
    Load MITgcm sensitivity data and calculate magnitudes at each time point.
    
    Parameters:
    -----------
    var_name : str
        Name of the variable (e.g., 'hfds', 'tauuo', 'tauvo')
    location : tuple
        (lat, lon) coordinates for the output point
    days_per_level : int
        Number of days per time level in the data
        
    Returns:
    --------
    days : list
        List of days (from day 1460 backwards)
    magnitudes : list
        List of sensitivity magnitudes
    """
    # Construct folder path using location
    loc_str = f"({location[0]},{location[1]})"
    folder_path = Path(mitgcm_folder) / loc_str
    
    # Construct the file path for the variable
    file_path = folder_path / f"{var_name}.npy"
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return [], []
    
    # Load the data - it should have shape (time, lat, lon)
    sensitivity_data = np.load(file_path)
    
    # Check if we need to reshape
    if len(sensitivity_data.shape) != 3:
        print(f"Error: Expected 3D data (time, lat, lon), but got shape {sensitivity_data.shape}")
        return [], []
    
    # Load wetmask if available
    drymask_path = 'drymask.npy'
    if Path(drymask_path).exists():
        drymask = np.load(drymask_path)
        wetmask = np.where(drymask == 0, 1, 0)  # Invert the mask
    else:
        wetmask = None
    
    # Calculate magnitude at each time point
    num_time_levels = sensitivity_data.shape[0]
    magnitudes = []
    days = []
    
    # Process data in reverse order (oldest to newest)
    for t in range(num_time_levels):
        # Extract sensitivity map for this time level
        sensitivity_map = sensitivity_data[t, :, :]
        
        # Apply mask if available
        if wetmask is not None:
            masked_sensitivity = np.ma.masked_array(
                sensitivity_map,
                mask=~wetmask.astype(bool)
            )
        else:
            masked_sensitivity = sensitivity_map
        
        # Calculate magnitude
        magnitude = calculate_magnitude(masked_sensitivity)
        
        # Calculate days from end (days from day 1460)
        day = (num_time_levels - 1 - t) * days_per_level
        
        # Store results
        days.append(day)
        magnitudes.append(magnitude)
    
    # Sort by days (ascending)
    days_magnitudes = sorted(zip(days, magnitudes))
    days, magnitudes = zip(*days_magnitudes) if days_magnitudes else ([], [])
    
    return list(days), list(magnitudes)

def load_samudra_sensitivities(location, ch_in, ch_out, end_time=292, normalize=True):
    """
    Load Samudra sensitivity data and calculate magnitudes.
    
    Parameters:
    -----------
    location : tuple
        (lat, lon) coordinates for the output point
    ch_in : int
        Input channel number
    ch_out : int
        Output channel number
    end_time : int
        End time for sensitivity calculations
    normalize : bool
        Whether to normalize sensitivities
        
    Returns:
    --------
    days : list
        List of days (from day 1460 backwards)
    magnitudes : list
        List of sensitivity magnitudes
    """
    # Get variable names from channel numbers
    channel_to_var = misc.get_channel_to_var()
    var_out = channel_to_var[ch_out]
    var_in = channel_to_var[ch_in]
    
    # Load standard deviations for normalization if needed
    if normalize:
        data_std = xr.open_zarr("./data_std.zarr")
    
    # Define range of times to check (from 0 to end_time)
    times_to_check = range(0, end_time, 2)  # Every even timestep
    
    # Load wetmask if available
    drymask_path = 'drymask.npy'
    if Path(drymask_path).exists():
        drymask = np.load(drymask_path)
        wetmask = np.where(drymask == 0, 1, 0)  # Invert the mask
    else:
        wetmask = None
    
    # Calculate magnitudes for each time point
    magnitudes = []
    days = []
    
    for t in times_to_check:
        # Construct file path based on time, channels, and location
        file_path = Path(samudra_folder) / f"avg_sensitivity_chin[{ch_in}]_chout[{ch_out}]_loc[{location[0]},{location[1]}]_t[{t},{end_time}-297].npy"
        
        # Check alternate format if first format not found
        if not file_path.exists():
            file_path = Path(samudra_folder) / f"avg_sensitivity_chin[{ch_in}]_chout[{ch_out}]_loc[{location[0]},{location[1]}]_t[{t},{end_time}].npy"
        
        if file_path.exists():
            try:
                # Load the sensitivity matrix
                matrix = np.load(file_path)
                
                # Reshape if needed
                if len(matrix.shape) > 2:
                    matrix = matrix.reshape(180, 360)
                
                # Normalize if requested
                if normalize:
                    matrix = misc.denormalize_sensitivity(matrix, var_out, var_in, data_std)
                
                # Apply mask if available
                if wetmask is not None:
                    masked_matrix = np.ma.masked_array(
                        matrix,
                        mask=~wetmask.astype(bool)
                    )
                else:
                    masked_matrix = matrix
                
                # Calculate magnitude
                magnitude = calculate_magnitude(masked_matrix)
                
                # Calculate days from end (1460 days total, 5 days per step)
                day = (end_time - t) * 5
                
                # Store results
                days.append(day)
                magnitudes.append(magnitude)
                
                print(f"Samudra time {t}, Day {day}: Magnitude = {magnitude:.6e}")
                
            except Exception as e:
                print(f"Error processing Samudra file for time {t}: {e}")
    
    # Sort by days (ascending)
    days_magnitudes = sorted(zip(days, magnitudes))
    days, magnitudes = zip(*days_magnitudes) if days_magnitudes else ([], [])
    
    return list(days), list(magnitudes)

def plot_comparison(location, var_name, ch_in, ch_out, normalize=True):
    """
    Compare sensitivity magnitudes between MITgcm and Samudra.
    
    Parameters:
    -----------
    location : tuple
        (lat, lon) coordinates for the output point
    var_name : str
        Name of the variable (e.g., 'hfds', 'tauuo', 'tauvo')
    ch_in : int
        Input channel number for Samudra
    ch_out : int
        Output channel for Samudra (usually 76 for zos)
    normalize : bool
        Whether to normalize Samudra sensitivities
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get variable names from channel numbers
    channel_to_var = misc.get_channel_to_var()
    var_out = channel_to_var[ch_out]
    var_in = channel_to_var[ch_in]
    
    # Define units mapping for pretty printing
    channel_to_units = {
        76: "m",
        154: "Pa",
        155: "Pa",
        156: "W m^{-2}",
    }
    
    # Load MITgcm data
    mitgcm_days, mitgcm_magnitudes = load_mitgcm_sensitivities(var_name, location)

    print(mitgcm_days, mitgcm_magnitudes)

    #raise Exception
    
    # Load Samudra data
    samudra_days, samudra_magnitudes = load_samudra_sensitivities(location, ch_in, ch_out, normalize=normalize)
    
    # Create plot if we have data
    if mitgcm_days or samudra_days:
        plt.figure(figsize=(12, 8))
        
        # Plot MITgcm data if available
        if mitgcm_days:
            plt.plot(mitgcm_days, mitgcm_magnitudes, 'b-', linewidth=2, marker='o', markersize=6, label='MITgcm')
        
        # Plot Samudra data if available
        if samudra_days:
            plt.plot(samudra_days, samudra_magnitudes, 'r-', linewidth=2, marker='s', markersize=6, label='Samudra')
        
        plt.xlabel('Days before end (from day 1460)', fontsize=12)
        
        # Add units to y-axis label if available
        if ch_in in channel_to_units and ch_out in channel_to_units:
            units = f"{channel_to_units[ch_out]} / {channel_to_units[ch_in]}"
            plt.ylabel(f'Global Sensitivity Magnitude [{units}]', fontsize=12)
        else:
            plt.ylabel('Global Sensitivity Magnitude', fontsize=12)
        
        # Create clean variable names for display
        var_in_clean = var_in.replace('(odd)','').replace('(even)','')
        var_out_clean = var_out.replace('(odd)','').replace('(even)','')
        
        # Use global coordinates for location in title
        samudra_coords_to_global_coords = {
            (126, 324): ('36N', '36W'),  
            (90, 180): ('0', '0'),  
            (131, 289): ('41N', '71W')
        }
        
        if tuple(location) in samudra_coords_to_global_coords:
            loc_str = f"{samudra_coords_to_global_coords[tuple(location)]}"
        else:
            loc_str = f"({location[0]},{location[1]})"
        
        plt.title(f'Sensitivity Magnitude Comparison: {var_out_clean} at {loc_str} wrt {var_in_clean}', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.legend(fontsize=12)
        
        # Add exponential notation for y-axis
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        
        # Save the plot
        loc_str_file = f"{location[0]}_{location[1]}".replace(",", "_")
        plot_path = f"{output_dir}/sensitivity_magnitude_comparison_{var_name}_{loc_str_file}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved to {plot_path}")
        return True
    else:
        print("No valid data to plot for comparison")
        return False

if __name__ == "__main__":
    # Define parameters for comparison
    # Dictionary mapping variable names to channel numbers
    var_to_channel = {
        'hfds': 156,
        'tauuo': 154,
        'tauvo': 155
    }
    
    # List of locations to process
    locations = [
        (90, 180),    # Equatorial Pacific
        (126, 324),   # North Atlantic Ocean
        (131, 289)    # Nantucket
    ]
    
    # Output channel (always zos - sea surface height)
    ch_out = 76
    
    # Process each variable at each location
    for location in locations:
        for var_name, ch_in in var_to_channel.items():
            print(f"\nProcessing {var_name} at location {location}...")
            plot_comparison(location, var_name, ch_in, ch_out, normalize=True)