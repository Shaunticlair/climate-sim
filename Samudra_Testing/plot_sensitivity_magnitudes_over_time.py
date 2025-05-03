import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ===== CONFIGURATION (MODIFY THESE VALUES) =====
# List of starting times for which to load sensitivity matrices
times = [6*i for i in range(12)]  # Example: [0, 6, 12, ..., 66] for a year
# Input and output channel numbers
chin = 76
chout = 76
# End time value (same for all matrices)
tend = 72
# Directory to save the output plot
output_dir = "Plots"
input_folder = 'adjoint_arrays/Equatorial_Pacific'
import misc

denormalize = True
if denormalize:

    import xarray as xr
    data_std = xr.open_zarr("./data_std.zarr")

    

### Denormalization function

def denormalize_sensitivity(sensitivity_tensor, var_out, var_in, data_std):

    """
    Denormalize the sensitivity tensor based on the variable configurations.
    
    Parameters:
    -----------
    sensitivity_tensor : torch.Tensor or numpy.ndarray
        The sensitivity tensor in normalized space
    var_out : str
        The output variable name (e.g., 'zos', 'thetao_lev_2_5')
    var_in : str
        The input variable name (e.g., 'hfds', 'tauuo')
    data_std : xarray.Dataset
        Dataset containing standard deviations for variables
        
    Returns:
    --------
    torch.Tensor or numpy.ndarray
        The denormalized sensitivity tensor
    
    Notes:
    ------
    Sensitivity is defined as ∂y/∂x, where y is the output and x is the input.
    To denormalize, we need to multiply by std_y/std_x, since:
    ∂(y/std_y)/∂(x/std_x) = (∂y/∂x) * (std_x/std_y)
    """
    
    # Get standard deviations for output and input variables
    std_out = data_std[var_out].values.item()
    std_in = data_std[var_in].values.item()
    
    # Calculate the denormalization factor
    denorm_factor = std_out / std_in
    
    # Apply the denormalization
    return sensitivity_tensor * denorm_factor
    


# ================================================

def calculate_magnitude(matrix):
    """Calculate the vector magnitude (L2 norm) of a matrix."""
    # Handle masked arrays if present
    if isinstance(matrix, np.ma.MaskedArray):
        # Only consider unmasked elements
        return np.sqrt(np.sum(matrix.compressed() ** 2))
    else:
        # Regular array - use all elements
        return np.sqrt(np.sum(matrix ** 2))

def plot_sensitivity_magnitudes(chin, chout, denormalize=False):
    """
    Load sensitivity matrices for the given times, calculate their magnitudes,
    and plot delay vs magnitude.
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)

    channel_to_var = misc.get_channel_to_var()
    var_out = channel_to_var[chout]
    var_in = channel_to_var[chin]
    
    # Lists to store delay and corresponding magnitudes
    magnitudes = []
    valid_times = []
    delays = []
    
    # Process each time point
    for t in times:
        # Construct file path
        file_path = f"chunk_sensitivity_chin[{chin}]_chout[{chout}]_t[{t},{tend}].npy"
        file_path = Path(input_folder) / file_path
        
        if Path(file_path).exists():
            try:
                # Load the sensitivity matrix
                matrix = np.load(file_path)
                
                # Ensure matrix is properly shaped
                if len(matrix.shape) > 2:
                    matrix = matrix.reshape(180, 360)

                if denormalize:
                    
                    # Normalize the matrix using the standard deviation of the output variable
                    matrix = misc.denormalize_sensitivity(matrix, var_out, var_in, data_std)

                
                # Load wetmask if available
                drymask_path = 'drymask.npy'
                if Path(drymask_path).exists():
                    drymask = np.load(drymask_path)
                    wetmask = np.where(drymask == 0, 1, 0)  # Invert the mask
                    
                    # Apply mask if shapes match
                    if wetmask.shape == matrix.shape:
                        matrix = np.ma.masked_array(
                            matrix,
                            mask=~wetmask.astype(bool)  # Invert wetmask to get drymask
                        )
                
                # Calculate magnitude
                magnitude = calculate_magnitude(matrix)
                
                # Calculate delay
                delay = tend - t
                
                # Store results
                magnitudes.append(magnitude)
                valid_times.append(t)
                delays.append(delay)
                
                print(f"Time {t}, Delay {delay}: Magnitude = {magnitude:.6e}")
                
            except Exception as e:
                print(f"Error processing file for time {t}: {e}")
        else:
            print(f"File not found for time {t}")
    
    # Create plot if we have data
    if valid_times:
        plt.figure(figsize=(10, 6))
        
        # Sort by delay for proper plotting
        delay_magnitude = sorted(zip(delays, magnitudes))
        sorted_delays, sorted_magnitudes = zip(*delay_magnitude)
        
        plt.plot(sorted_delays, sorted_magnitudes, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Delay (t_end - t)', fontsize=12)
        plt.ylabel('Sensitivity Magnitude (L2 Norm)', fontsize=12)
        plt.title(f'Sensitivity Magnitude vs. Delay for varin[{var_in}]_varout[{var_out}]_t[*,{tend}]', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add exponential notation for y-axis
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        
        # Ensure integer ticks on x-axis
        plt.xticks(sorted_delays)
        
        # Save the plot
        plot_path = f"{output_dir}/sensitivity_magnitude_chin[{chin}]_chout[{chout}]_tend[{tend}].png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to {plot_path}")
        return valid_times, magnitudes, delays
    else:
        print("No valid data to plot")
        return [], [], []

if __name__ == "__main__":
    chins = [76,153,154,155,156,157]
    for chin in chins:
        plot_sensitivity_magnitudes(chin,chout,denormalize)