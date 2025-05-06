import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_and_crop_sensitivity(path, map_dims):
    """Load a sensitivity matrix and crop it to the specified dimensions"""
    xmin, xmax, ymin, ymax = map_dims
    
    # Load the sensitivity matrix
    sensitivity_matrix = np.load(path)
    sensitivity_matrix = sensitivity_matrix.reshape(180, 360)

    # Crop the sensitivity matrix
    cropped_sensitivity = sensitivity_matrix[xmin:xmax, ymin:ymax]
    
    # Load and crop the wetmask
    drymask_path = 'drymask.npy'
    drymask = np.load(drymask_path)
    wetmask = np.where(drymask == 0, 1, 0)  # Invert the mask
    cropped_wetmask = wetmask[xmin:xmax, ymin:ymax]

    # Apply the wetmask to the sensitivity
    masked_sensitivity = np.ma.masked_array(
        cropped_sensitivity,
        mask=~cropped_wetmask.astype(bool)  # Invert wetmask to get drymask
    )
    
    return masked_sensitivity

def plot_grid_sensitivities(t_times, t_end, map_dims, output_pixel, var_in, var_out, ch_in, ch_out, folder='.'):
    """
    Create a 2x3 grid of sensitivity plots, each with its own color scale
    
    Parameters:
    -----------
    t_times : list
        List of 6 time points (t0 values)
    t_end : int
        End time point (t1 value)
    map_dims : list
        [xmin, xmax, ymin, ymax] defining map boundaries
    output_pixel : tuple
        (lat, lon) coordinates of the output pixel
    var_in : str
        Input variable name
    var_out : str
        Output variable name
    ch_in : int
        Input channel number
    ch_out : int
        Output channel number
    folder : str
        Folder to look for sensitivity files
    """
    # Reverse the order of t_times to show largest lags first
    t_times = sorted(t_times, reverse=True)
    
    # Set up the figure with 2x3 grid
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.0)
    
    # Get output pixel coordinates
    output_lat, output_lon = output_pixel
    
    # Create a list to store all masked sensitivities
    all_sensitivities = []
    
    # First, load all data
    for i, t0 in enumerate(t_times):
        in_time, out_time = t0, t_end
        plot_path = Path(f'{folder}/avg_sensitivity_chin[{ch_in}]_chout[{ch_out}]_t[{in_time},{out_time}].npy')
        
        if plot_path.exists():
            masked_sensitivity = load_and_crop_sensitivity(plot_path, map_dims)
            all_sensitivities.append(masked_sensitivity)
        else:
            print(f"File not found: {plot_path}")
            return
    
    # Calculate tick positions and labels ONCE before the loop
    xmin, xmax, ymin, ymax = map_dims
    row_indices = np.arange(xmin, xmax)
    col_indices = np.arange(ymin, ymax)
    
    # Set ticks at appropriate positions using the shape of the first sensitivity matrix
    sample_sensitivity = all_sensitivities[0]
    x_pos = np.arange(sample_sensitivity.shape[1])
    y_pos = np.arange(sample_sensitivity.shape[0])
    
    # Calculate tick positions and labels
    xticks = 20
    yticks = 20
    
    x_tick_pos = x_pos[::xticks]
    x_tick_labs = col_indices[::xticks]
    
    y_tick_pos = y_pos[::yticks]
    y_tick_labs = row_indices[::yticks]
    
    # Now create each subplot
    for i, t0 in enumerate(t_times):
        # Calculate grid position (row, col)
        row = i // 3
        col = i % 3
        
        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        
        masked_sensitivity = all_sensitivities[i]
        
        # Create a custom colormap with black for masked values
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad('black', 0.5)
        
        # Calculate individual color scale (symmetric around zero)
        abs_max = np.max(np.abs(masked_sensitivity.compressed()))
        vmin, vmax = -abs_max, abs_max
        print(f"Time {t0}: Using color limits [{vmin}, {vmax}]")
        
        # Plot the sensitivity with its own color scale
        im = ax.imshow(masked_sensitivity, cmap=cmap, aspect='equal', origin='lower', 
                      vmin=vmin, vmax=vmax)
        
        # Add a small colorbar for each plot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        # Make colorbar values more compact (scientific notation for small values)
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.update_ticks()
        
        # Calculate time lag from t0 to t_end
        time_lag = t_end - t0
        
        # Add title with time lag using LaTeX formatting
        ax.set_title(r'$t_0 = %d$ (lag = %d)' % (t0, time_lag), fontsize=12)
        
        # Add output pixel circle
        circle_y, circle_x = output_pixel
        # Adjust for cropping
        circle_y_adj = circle_y - xmin
        circle_x_adj = circle_x - ymin
        
        # Only draw the circle if it's within the cropped region
        if (0 <= circle_y_adj < masked_sensitivity.shape[0] and 
            0 <= circle_x_adj < masked_sensitivity.shape[1]):
            # Create circle
            circle = plt.Circle((circle_x_adj, circle_y_adj), 2, 
                            color='black', fill=False, linewidth=2)
            ax.add_patch(circle)
        
        # Add ticks and labels for every other subplot
        if row == 1:  # Bottom row
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
    
    # Clean up variable names for display
    var_in_clean = var_in.replace('(odd)','').replace('(even)','')
    var_out_clean = var_out.replace('(odd)','').replace('(even)','')
    
    # Format variable names for LaTeX
    var_in_latex = var_in_clean.replace('_', '\_')
    var_out_latex = var_out_clean.replace('_', '\_')
    
    # Add super title using LaTeX formatting
    fig.suptitle(r'Sensitivity of $%s$ at $(%d,%d)$ at $t_1=%d$' '\n'
                 r'to $%s$ across map at varying starting times $t_0$' % 
                 (var_out_latex, output_lat, output_lon, t_end, var_in_latex), 
                 fontsize=16, y=0.98)
    
    # Create Plots directory if it doesn't exist
    Path("Plots").mkdir(exist_ok=True)
    
    # Save the figure
    view_name = f"({map_dims[0]},{map_dims[1]})x({map_dims[2]},{map_dims[3]})"
    filename = f'Plots/grid_sensitivity_{view_name}_chin[{var_in_clean}]_chout[{var_out_clean}]_t1[{t_end}].png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved grid plot with individual scales to {filename}")

if __name__ == "__main__":
    # Set parameters
    t_end = 72#12  # End time
    t_times = [12*i for i in range(0,6)]#[0, 2, 4, 6, 8, 10]  # 6 starting time points
    t_end = 6
    t_times = [0,1,2,3,4,5]
    # Define output pixel (center of interest)
    output_pixel = (90, 180)  # Equatorial Pacific
    
    # Define map dimensions
    delta = 20
    map_dims = [90-delta, 90+delta+1, 180-delta, 180+delta+1]  # [xmin, xmax, ymin, ymax]
    #map_dims = [0, 180, 0, 360]  # [xmin, xmax, ymin, ymax]
    # Try to import variable dictionary from misc
    from misc import var_dict

    
    # Choose variables to plot
    var_in = 'zos(even)'  # Input variable
    var_out = 'zos(even)'  # Output variable
    
    # Get channel numbers from dictionary
    ch_in = var_dict[var_in]
    ch_out = var_dict[var_out]
    folder = 'adjoint_arrays/Equatorial_Pacific/'

    # Create the grid sensitivity plot
    plot_grid_sensitivities(t_times, t_end, map_dims, output_pixel, 
                        var_in, var_out, ch_in, ch_out, folder=folder)