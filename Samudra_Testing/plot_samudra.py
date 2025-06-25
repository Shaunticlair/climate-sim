import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray as xr
full_map = False
small_map = True

x = 1.5
plt.rcParams['font.size'] = 24/2*x
plt.rcParams['axes.titlesize'] = 32/2*x
plt.rcParams['axes.labelsize'] = 28/2*x
plt.rcParams['xtick.labelsize'] = 24/2*x
plt.rcParams['ytick.labelsize'] = 24/2*x
plt.rcParams['legend.fontsize'] = 24/2*x

#plt.rcParams['figure.constrained_layout.use'] = True  # Use constrained layout
plt.rcParams['axes.titlepad'] = 22  # Increase padding between title and plot
plt.rcParams['figure.subplot.wspace'] = 0.1 if full_map else -0.25 if small_map else -0.25#-0.5  # Increase width spacing between subplots
plt.rcParams['figure.subplot.hspace'] = -0.5 if full_map else 0.25 if small_map else 0.25#-0  # Increase height spacing between subplots

def denormalize_sensitivity(sensitivity_tensor, var_out, var_in, data_std):
    """
    Denormalize the sensitivity tensor based on the variable configurations.
    
    Parameters:
    -----------
    sensitivity_tensor : numpy.ndarray
        The sensitivity tensor in normalized space
    var_out : str
        The output variable name (e.g., 'zos', 'thetao_lev_2_5')
    var_in : str
        The input variable name (e.g., 'hfds', 'tauuo')
    data_std : xarray.Dataset
        Dataset containing standard deviations for variables
        
    Returns:
    --------
    numpy.ndarray
        The denormalized sensitivity tensor
    """
    # Strip any (even) or (odd) suffixes for matching with data_std variable names
    var_out_clean = var_out.replace('(odd)','').replace('(even)','')
    var_in_clean = var_in.replace('(odd)','').replace('(even)','')
    
    # Get standard deviations for output and input variables
    std_out = data_std[var_out_clean].values.item()
    std_in = data_std[var_in_clean].values.item()
    
    # Calculate the denormalization factor
    denorm_factor = std_out / std_in
    
    # Apply the denormalization
    return sensitivity_tensor * denorm_factor

def load_and_crop_sensitivity(path, map_dims, var_out, var_in, denormalize=True):
    """Load a sensitivity matrix and crop it to the specified dimensions"""
    xmin, xmax, ymin, ymax = map_dims
    
    # Load the sensitivity matrix
    sensitivity_matrix = np.load(path)
    sensitivity_matrix = sensitivity_matrix.reshape(180, 360)

    # Denormalize if requested
    if denormalize:
        # Load standard deviation data
        data_std = xr.open_zarr("./data_std.zarr")
        sensitivity_matrix = denormalize_sensitivity(sensitivity_matrix, var_out, var_in, data_std)

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

def plot_grid_sensitivities(t_days, t_end, map_dims, output_pixel, var_in, var_out, ch_in, ch_out, 
                            folder='.', circle=True, denormalize=True):
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
    denormalize : bool
        Whether to denormalize sensitivity values
    """
    # Reverse the order of t_times to show largest lags first
    #t_days = sorted(t_days, reverse=True)
    
    # Set up the figure with 2x3 grid
    fig = plt.figure(figsize=(22, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig) ############## OVERHERE!!!!!!!!!!!!!!!!###################### -0.7
    
    # Get output pixel coordinates
    output_lat, output_lon = output_pixel
    
    # Create a list to store all masked sensitivities
    all_sensitivities = []

    # First, load all data
    for i, t0 in enumerate(t_days):
        if t_end == '292-297':
            in_time = (1460 - t0) // 5
            out_time = t_end # String
        else:
            in_time = (t_end - t0) // 5
            out_time = (t_end) // 5

        
        avg_or_chunk = ['chunk','avg']
        loc_or_no_loc = [f'_loc[{output_lat},{output_lon}]', '']
        plot_paths = []
        for avg_chunk in avg_or_chunk:
            for loc in loc_or_no_loc:
                plot_paths.append(Path(f'{folder}/{avg_chunk}_sensitivity_chin[{ch_in}]_chout[{ch_out}]{loc}_t[{in_time},{out_time}].npy'))
                print(plot_paths)
        
        found_file = False
        for plot_path in plot_paths:
            if plot_path.exists():
                masked_sensitivity = load_and_crop_sensitivity(plot_path, map_dims, var_out, var_in, denormalize)
                all_sensitivities.append(masked_sensitivity)
                found_file = True
                break
        if not found_file:
            print(f"File not found: {plot_path}. Failed")
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
    xticks = 40 if sample_sensitivity.shape[1] > 120 else 10
    yticks = 40 if sample_sensitivity.shape[0] > 120 else 10
    
    x_tick_pos = x_pos[::xticks]
    x_tick_labs = col_indices[::xticks]
    
    y_tick_pos = y_pos[::yticks]
    y_tick_labs = row_indices[::yticks]
    
    # Now create each subplot
    for i, t0 in enumerate(t_days):
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
        
        time_lag = t0
        
        # Add title with time lag using LaTeX formatting
        ax.set_title(f'{time_lag} days back ($t_{1}-t_0 = {(t0)//5}$)')
        
        if circle: 
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

    # Define unit mapping for variables
    var_units = {
        'zos': 'm',
        'tauuo': 'Pa',
        'tauvo': 'Pa',
        'hfds': 'W m$^{-2}$',
        'hfds_anomalies': 'W m$^{-2}$',
    }
    
    # Get units for the variables
    out_unit = var_units.get(var_out_clean, '')
    in_unit = var_units.get(var_in_clean, '')
    
    # Create unit string if both units are available
    unit_str = f" [{out_unit}/{in_unit}]" if (out_unit and in_unit) else ""

    samudra_coords_to_global_coords = {
        (126, 324): ('36N', '36W'),  
        (90, 180): ('0N', '0W'),  
        (131, 289): ('41N', '71W')
    }
    output_pixel = samudra_coords_to_global_coords[output_pixel]
    output_lat, output_lon = output_pixel

    final_day = 1460 if t_end == '292-297' else t_end
    
    y=0.85 if full_map else 1.0 if small_map else 0.95
    # Add super title using LaTeX formatting
    fig.suptitle(f'Samudra: Sensitivity of {var_out_clean} at ({output_lat},{output_lon}) at day {final_day}\n'
                 f'wrt {var_in_clean} across map over various time scales{unit_str}', 
                 y=y) ################################ OVERHERE!!!!!!!!!!!!!!!!###################### 0.85 
                # 1.1
    # Create Plots directory if it doesn't exist
    Path("Plots").mkdir(exist_ok=True)
    
    # Save the figure
    view_name = f"({map_dims[0]},{map_dims[1]})x({map_dims[2]},{map_dims[3]})"
    filename = f'Plots/grid_samudra_sensitivity_[{output_lat},{output_lon}]_{view_name}_chin[{var_in_clean}]_chout[{var_out_clean}].png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved grid plot with individual scales to {filename}")

if __name__ == "__main__":
    t_end = '292-297'
    #t_days = [70, 140, 210, 350, 490, 700] #[70, 140, 210, 350, 490, 700] #[10*i for i in range(1,7)]
    t_days = [10*i for i in range(1,7)]
    
    
    #t_days = [2*i*5 for i in range(1,7)]
    #t_end = 12*5

    #t_days = [12*i*5 for i in range(1,7)]
    #t_end = 72*5
    # Define output pixel (center of interest)
    
    #t_days = [5*i for i in range(1,7)]
    #t_end = 24*5
    
    # Try to import variable dictionary from misc
    from misc import var_dict

    
    # Choose variables to plot
    vars_in = ['hfds',]# 'tauuo', 'tauvo']#['zos(even)']#['hfds', 'tauuo', 'tauvo']#, 'tauuo', 'tauvo']  # Input variable
    var_out = 'zos(even)'  # Output variable
    
    for loc in ['(126,324)']:#'(131,289)',]:#['(90,180)',]:#:# []: 
        #loc = '(126,324)'  # North Atlantic Ocean
        xout, yout = eval(loc)
        output_pixel = (xout, yout)  # Coordinates of the output pixel
    
        # Define map dimensions
        delta = 20
        map_dims = [xout-delta, xout+delta+1, yout-delta, yout+delta+1]  # [xmin, xmax, ymin, ymax]
        #map_dims = [0, 180, 180, 360] # [0, 180, 0, 360]  # [xmin, xmax, ymin, ymax]
        #map_dims = [0, 180, 0, 360]  # [xmin, xmax, ymin, ymax]

        for var_in in vars_in:
            # Get channel numbers from dictionary
            ch_in = var_dict[var_in]
            ch_out = var_dict[var_out]
            folder = 'MITgcm_Replication/'#'adjoint_arrays/Equatorial_Pacific/' #

            # Create both normalized and denormalized plots
            plot_grid_sensitivities(t_days, t_end, map_dims, output_pixel, 
                                var_in, var_out, ch_in, ch_out, folder=folder, circle=False, denormalize=True)
            
            # Optionally also create normalized version for comparison
            #plot_grid_sensitivities(t_days, t_end, map_dims, output_pixel, 
            #                    var_in, var_out, ch_in, ch_out, folder=folder, circle=False, denormalize=False)