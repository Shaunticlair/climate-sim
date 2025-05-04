import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_correlation_grid(correlation_file, map_dims, reference_point, var_in, var_out):
    """
    Create a 3x4 grid of lag correlation plots
    
    Parameters:
    -----------
    correlation_file : str
        Path to the correlation file
    map_dims : list
        [xmin, xmax, ymin, ymax] defining map boundaries
    reference_point : tuple
        (lat, lon) coordinates of the reference point
    """
    # Load the correlation data
    correlation_data = np.load(correlation_file)
    print(f"Loaded correlation data with shape: {correlation_data.shape}")
    
    # The first dimension should be the lag
    num_lags = correlation_data.shape[0]
    
    # We expect at least 12 lags
    if num_lags < 12:
        print(f"Warning: Only {num_lags} lags found in correlation data, expected at least 12")
    
    # Use first 12 lags or as many as available
    lags_to_plot = min(12, num_lags)
    
    # Extract map dimensions
    xmin, xmax, ymin, ymax = map_dims
    
    # Get reference point coordinates
    ref_lat, ref_lon = reference_point
    
    # Set up the figure with 3x4 grid
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 4, figure=fig, wspace=0.3, hspace=0.4)
    
    # Create a list to store all colorbars
    all_colorbars = []
    
    # Calculate tick positions and labels
    row_indices = np.arange(xmin, xmax)
    col_indices = np.arange(ymin, ymax)
    
    # Set ticks at appropriate positions
    sample_data = correlation_data[0, :, :]
    if xmin > 0 or ymin > 0 or xmax < sample_data.shape[0] or ymax < sample_data.shape[1]:
        # Crop sample data if map_dims don't match the full dimensions
        cropped_sample = sample_data[xmin:xmax, ymin:ymax]
    else:
        cropped_sample = sample_data
    
    x_pos = np.arange(cropped_sample.shape[1])
    y_pos = np.arange(cropped_sample.shape[0])
    
    # Calculate tick positions and labels
    xticks = 20
    yticks = 20
    
    x_tick_pos = x_pos[::xticks]
    x_tick_labs = col_indices[::xticks]
    
    y_tick_pos = y_pos[::yticks]
    y_tick_labs = row_indices[::yticks]
    
    # Create subplots for each lag
    for i in range(lags_to_plot):
        # Calculate grid position (row, col)
        row = i // 4
        col = i % 4
        
        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        
        # Get correlation data for this lag
        lag_data = correlation_data[i]
        
        # Crop if necessary
        if xmin > 0 or ymin > 0 or xmax < lag_data.shape[0] or ymax < lag_data.shape[1]:
            lag_data = lag_data[xmin:xmax, ymin:ymax]
            
        # Check for NaN values and create a masked array if needed
        if np.isnan(lag_data).any():
            lag_data = np.ma.masked_array(lag_data, mask=np.isnan(lag_data))
        
        # Create a custom colormap with black for masked values
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad('black', 0.5)
        
        # Calculate individual color scale (symmetric around zero)
        if isinstance(lag_data, np.ma.MaskedArray):
            abs_max = np.max(np.abs(lag_data.compressed()))
        else:
            abs_max = np.max(np.abs(lag_data))
            
        vmin, vmax = -abs_max, abs_max
        print(f"Lag {i}: Using color limits [{vmin}, {vmax}]")
        
        # Plot the correlation with its own color scale
        im = ax.imshow(lag_data, cmap=cmap, aspect='equal', origin='lower', 
                      vmin=vmin, vmax=vmax)
        
        # Add a small colorbar for each plot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        # Make colorbar values more compact (scientific notation for small values)
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.update_ticks()
        all_colorbars.append(cbar)
        
        # Add title with lag information
        ax.set_title(f'Lag {i}', fontsize=12)
        
        # Add reference point circle
        circle_y, circle_x = ref_lat - xmin, ref_lon - ymin
        
        # Only draw the circle if it's within the displayed region
        if (0 <= circle_y < lag_data.shape[0] and 
            0 <= circle_x < lag_data.shape[1]):
            # Create circle
            circle = plt.Circle((circle_x, circle_y), 2, 
                            color='black', fill=False, linewidth=2)
            ax.add_patch(circle)
        
        # Add ticks and labels for every other subplot
        if row == 2:  # Bottom row
            # Display actual matrix indices on the axes
            ax.set_xticks(x_tick_pos[:min(len(x_tick_pos), len(x_tick_labs))])
            ax.set_xticklabels(x_tick_labs[:min(len(x_tick_pos), len(x_tick_labs))], rotation=45)
            ax.set_xlabel('Longitude Index')
        else:
            ax.set_xticks([])
            
        if col == 0:  # Left column
            ax.set_yticks(y_tick_pos[:min(len(y_tick_pos), len(y_tick_labs))])
            ax.set_yticklabels(y_tick_labs[:min(len(y_tick_pos), len(y_tick_labs))])
            ax.set_ylabel('Latitude Index')
        else:
            ax.set_yticks([])
    
    # Add super title
    fig.suptitle(f'Lag Correlation of {var_out} at ({ref_lat},{ref_lon}) with {var_in} across map', 
                 fontsize=16, y=0.98)
    
    # Create Plots directory if it doesn't exist
    Path("Plots").mkdir(exist_ok=True)
    
    # Save the figure
    view_name = f"({map_dims[0]},{map_dims[1]})x({map_dims[2]},{map_dims[3]})"
    filename = f'Plots/lag_correlation_{var_out}_at_{ref_lat}_{ref_lon}_with_{var_in}_{view_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved lag correlation grid plot to {filename}")

if __name__ == "__main__":
    # Set parameters
    # Define reference point (center of interest)
    reference_point = (90, 180)  # Equatorial Pacific
    
    # Define map dimensions - adjust as needed
    delta = 40
    x_ref, y_ref = reference_point
    map_dims = [x_ref-delta, x_ref+delta+1, y_ref-delta, y_ref+delta+1]  # [xmin, xmax, ymin, ymax]
    map_dims= [0,180,0,360]
    #(50,130)x(110,250)
    map_dims = [50, 130, 110, 250]  # [xmin, xmax, ymin, ymax]
    # Path to correlation file
    var_out = 'zos'
    for var_in in ['zos', 'tauuo', 'tauvo', 'hfds', 'hfds_anomalies']:
        correlation_file = f"CorrelationMaps/correlation_zos_at_90_180_with_{var_in}.npy"
        
        # Create the correlation grid plot
        plot_correlation_grid(correlation_file, map_dims, reference_point, var_in, var_out)