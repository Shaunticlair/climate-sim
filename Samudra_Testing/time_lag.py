import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys

# Import from setup.py
#sys.path.append("path/to/directory_containing_setup_py")
from setup import load_data_for_correlation_analysis, Timer, torch_config_cuda_cpu_seed

def compute_lagged_correlations(data_dict, time_window=60, max_lag=12, lag_step=1, device='cuda'):
    """
    Compute spatial lagged correlations between reference variable and field variables.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary returned by load_data_for_correlation_analysis
    time_window : int, optional
        Number of time steps to use for correlation calculation (default: 60)
        Should be less than the total time available minus max_lag
    max_lag : int, optional
        Maximum lag to consider (default: 12 time steps)
    lag_step : int, optional
        Step size for lags (default: 1)
    device : str, optional
        Device to compute on ('cpu' or 'cuda')
        
    Returns:
    --------
    dict
        Dictionary containing correlation maps for each field variable and lag
        Structure: {field_var: correlation_tensor, ...}
        where correlation_tensor has dimensions (lag, lat, lon)
    """
    # Get the reference series for the specified point
    reference_series = data_dict['reference_series']['raw'].values
    reference_series = torch.tensor(reference_series, dtype=torch.float32, device=device)
    
    # Number of time points in the data
    total_time_steps = len(reference_series)
    
    # Ensure we have enough data for the analysis
    if total_time_steps < time_window + max_lag:
        raise ValueError(f"Not enough time steps for analysis. Need at least {time_window + max_lag}, but got {total_time_steps}")
    
    # Use the most recent time_window points for the reference series
    ref_start_idx = total_time_steps - time_window
    ref_end_idx = total_time_steps
    reference = reference_series[ref_start_idx:ref_end_idx]
    
    # Standardize the reference series (mean=0, std=1)
    reference = (reference - reference.mean()) / (reference.std() + 1e-8)
    
    # Dictionary to store results
    correlation_maps = {}
    
    # Process each field variable
    for field_var, field_data in data_dict['field_series'].items():
        # Get the field values across all spatial points
        field_values = field_data['raw'].values
        
        # Get dimensions
        time_dim, lat_dim, lon_dim = field_values.shape
        
        # Convert to tensor
        field_tensor = torch.tensor(field_values, dtype=torch.float32, device=device)
        
        # Prepare tensor to store correlations for all lags
        num_lags = (max_lag // lag_step) + 1
        correlation_tensor = torch.zeros((num_lags, lat_dim, lon_dim), dtype=torch.float32, device=device)
        
        # Compute correlation for each lag
        for lag_idx, lag in enumerate(range(0, max_lag + 1, lag_step)):
            # For a given lag, we use field data from lag steps earlier than the reference
            # Field time range: [ref_start_idx - lag, ref_end_idx - lag)
            field_start_idx = ref_start_idx - lag
            field_end_idx = ref_end_idx - lag
            
            # Extract the lagged field tensor
            lagged_field = field_tensor[field_start_idx:field_end_idx]
            
            # Reshape to (time_window, lat_dim*lon_dim) for vectorized computation
            lagged_field_flat = lagged_field.reshape(time_window, -1)
            
            # Standardize each spatial point's time series
            lagged_field_mean = lagged_field_flat.mean(dim=0, keepdim=True)
            lagged_field_std = lagged_field_flat.std(dim=0, keepdim=True) + 1e-8  # Add epsilon to avoid division by zero
            lagged_field_standardized = (lagged_field_flat - lagged_field_mean) / lagged_field_std
            
            # Compute correlation coefficient (dot product of standardized series, divided by time_window)
            # reference shape: (time_window)
            # lagged_field_standardized shape: (time_window, lat_dim*lon_dim)
            correlation = torch.matmul(reference, lagged_field_standardized) / time_window
            
            # Reshape back to spatial dimensions
            correlation_map = correlation.reshape(lat_dim, lon_dim)
            
            # Store in the tensor
            correlation_tensor[lag_idx] = correlation_map
        
        # Apply wetmask to mask out land points
        wetmask = torch.tensor(data_dict['wetmask'].values, dtype=torch.float32, device=device)
        masked_correlation = correlation_tensor * wetmask
        
        # Store the result
        correlation_maps[field_var] = masked_correlation
    
    return correlation_maps

def save_correlation_maps(correlation_maps, reference_var, reference_point, output_dir='CorrelationMaps'):
    """
    Save correlation maps to disk.
    
    Parameters:
    -----------
    correlation_maps : dict
        Dictionary of correlation maps for each field variable
    reference_var : str
        Name of the reference variable
    reference_point : tuple
        (lat, lon) coordinates of the reference point
    output_dir : str, optional
        Directory to save correlation maps (default: 'CorrelationMaps')
        
    Returns:
    --------
    dict
        Dictionary with paths to saved files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    saved_paths = {}
    
    # Save each correlation map
    for field_var, correlation_tensor in correlation_maps.items():
        # Convert tensor to numpy for saving
        correlation_np = correlation_tensor.cpu().numpy()
        
        # Create filename
        filename = f"correlation_{reference_var}_at_{reference_point[0]}_{reference_point[1]}_with_{field_var}.npy"
        filepath = Path(output_dir) / filename
        
        # Save the numpy array
        np.save(filepath, correlation_np)
        saved_paths[field_var] = filepath
        
        print(f"Saved correlation map for {field_var} to {filepath}")
    
    return saved_paths

def plot_correlation_maps(correlation_maps, reference_var, reference_point, lags_to_plot=None, 
                          output_dir='CorrelationPlots', cmap='RdBu_r'):
    """
    Plot correlation maps for specified lags.
    
    Parameters:
    -----------
    correlation_maps : dict
        Dictionary of correlation maps for each field variable
    reference_var : str
        Name of the reference variable
    reference_point : tuple
        (lat, lon) coordinates of the reference point
    lags_to_plot : list, optional
        List of lag indices to plot. If None, plots all lags.
    output_dir : str, optional
        Directory to save plots (default: 'CorrelationPlots')
    cmap : str, optional
        Colormap to use for plotting (default: 'RdBu_r')
        
    Returns:
    --------
    dict
        Dictionary with paths to saved plots
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    saved_plots = {}
    
    # Process each field variable
    for field_var, correlation_tensor in correlation_maps.items():
        # Convert tensor to numpy for plotting
        correlation_np = correlation_tensor.cpu().numpy()
        
        # Get number of lags
        num_lags = correlation_np.shape[0]
        
        # Determine which lags to plot
        if lags_to_plot is None:
            lags_to_plot = list(range(num_lags))
        
        # Dictionaries to store plot paths
        saved_plots[field_var] = {}
        
        # Plot each specified lag
        for lag_idx in lags_to_plot:
            if lag_idx >= num_lags:
                print(f"Lag index {lag_idx} out of bounds for {field_var}. Skipping.")
                continue
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Get the correlation map for this lag
            lag_correlation = correlation_np[lag_idx]
            
            # Create symmetric color limits
            vmax = np.max(np.abs(lag_correlation))
            vmin = -vmax
            
            # Plot the correlation map
            im = plt.imshow(lag_correlation, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            plt.colorbar(im, label='Correlation')
            
            # Add title
            plt.title(f"Correlation of {reference_var} at {reference_point} with {field_var} (Lag {lag_idx})")
            
            # Add axis labels
            plt.xlabel('Longitude Index')
            plt.ylabel('Latitude Index')
            
            # Mark the reference point if it's within the mapped area
            plt.grid(False)
            
            # Save the plot
            plot_filename = f"correlation_plot_{reference_var}_at_{reference_point[0]}_{reference_point[1]}_with_{field_var}_lag_{lag_idx}.png"
            plot_filepath = Path(output_dir) / plot_filename
            plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            saved_plots[field_var][lag_idx] = plot_filepath
            print(f"Saved correlation plot for {field_var} at lag {lag_idx} to {plot_filepath}")
    
    return saved_plots

def main(reference_point=(90, 180), 
         spatial_slice=(slice(50, 130), slice(110, 250)),
         time_window=60,
         max_lag=12,
         lag_step=1,
         reference_var='zos',
         field_vars=['hfds', 'tauuo', 'tauvo', 'hfds_anomalies'],
         save_plots=True,
         lags_to_plot=None):
    """
    Main function to compute spatial lagged correlations and save results.
    
    Parameters:
    -----------
    reference_point : tuple, optional
        (lat, lon) coordinates of the reference point (default: (90, 180))
    spatial_slice : tuple, optional
        (lat_slice, lon_slice) defining the spatial region (default: (slice(50, 130), slice(110, 250)))
    time_window : int, optional
        Number of time steps to use for correlation (default: 60)
    max_lag : int, optional
        Maximum lag to consider (default: 12)
    lag_step : int, optional
        Step size for lags (default: 1)
    reference_var : str, optional
        Name of the reference variable (default: 'zos')
    field_vars : list, optional
        List of field variables to correlate with (default: ['hfds', 'tauuo', 'tauvo', 'hfds_anomalies'])
    save_plots : bool, optional
        Whether to save plots (default: True)
    lags_to_plot : list, optional
        List of lag indices to plot. If None, plots all lags.
        
    Returns:
    --------
    dict
        Dictionary with correlation maps and paths to saved files
    """
    # Initialize timer
    timer = Timer()
    
    # Set up device
    device = torch_config_cuda_cpu_seed()
    
    timer.checkpoint("Environment configured")
    
    # Load data
    print(f"Loading data for correlation analysis: {reference_var} at {reference_point} with {field_vars}")
    data_dict = load_data_for_correlation_analysis(
        reference_point=reference_point,
        spatial_slice=spatial_slice,
        reference_var=reference_var,
        field_vars=field_vars,
        time_window=time_window + max_lag,  # Need extra time for lagging
        device=device
    )
    
    timer.checkpoint("Data loaded")
    
    # Compute lagged correlations
    print(f"Computing lagged correlations with time_window={time_window}, max_lag={max_lag}, lag_step={lag_step}")
    correlation_maps = compute_lagged_correlations(
        data_dict,
        time_window=time_window,
        max_lag=max_lag,
        lag_step=lag_step,
        device=device
    )
    
    timer.checkpoint("Correlations computed")
    
    # Save correlation maps
    saved_maps = save_correlation_maps(
        correlation_maps,
        reference_var,
        reference_point
    )
    
    timer.checkpoint("Correlation maps saved")
    
    # Save plots if requested
    if save_plots:
        saved_plots = plot_correlation_maps(
            correlation_maps,
            reference_var,
            reference_point,
            lags_to_plot=lags_to_plot
        )
        timer.checkpoint("Plots saved")
    else:
        saved_plots = None
    
    # Return results
    results = {
        'correlation_maps': correlation_maps,
        'saved_maps': saved_maps,
        'saved_plots': saved_plots,
        'metadata': {
            'reference_point': reference_point,
            'spatial_slice': spatial_slice,
            'time_window': time_window,
            'max_lag': max_lag,
            'lag_step': lag_step,
            'reference_var': reference_var,
            'field_vars': field_vars
        }
    }
    
    print("Spatial lagged correlation analysis complete!")
    return results

if __name__ == "__main__":
    # Example usage
    print('None')
    results = main( 
        reference_point=(90, 180),
        spatial_slice=(slice(90-1,90+1+1), slice(180-1,180+1+1)),#(slice(50, 130), slice(110, 250)),
        time_window=72*2,  # Number of time steps for correlation
        max_lag=72,      # Maximum lag in time steps
        lag_step=6,      # Step size for lags
        reference_var='zos',
        field_vars=['zos'],
        save_plots=True,
        lags_to_plot=[6*i for i in range(13)],  # Plot specific lags (0, 3, 6, 9, 12)
    )

