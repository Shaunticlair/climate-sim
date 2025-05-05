import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def center_bounds(view_size, centerx, centery):
    if view_size == "tiny":
        # Define the region of interest in the matrix for cropping
        xmin, xmax = centerx-3, centerx+4  # Matrix row indices
        ymin, ymax = centery-3, centery+4  # Matrix column indices

    if view_size == "local":
        xmin, xmax = centerx-20, centerx+20  # Matrix row indices
        ymin, ymax = centery-20, centery+20  # Matrix column indices

    if view_size == "global":
        xmin, xmax = 0, 180
        ymin, ymax = 0, 360

    if isinstance(view_size, list): #Manual mode
        deltax, deltay = view_size
        # Define the region of interest in the matrix for cropping
        xmin, xmax = centerx-deltax, centerx+deltax+1
        ymin, ymax = centery-deltay, centery+deltay+1

    return xmin, xmax, ymin, ymax

def load_and_crop_sensitivity(path, map_dims):
    """Load a sensitivity matrix and crop it to the specified dimensions"""
    xmin, xmax, ymin, ymax = map_dims
    
    # Load the sensitivity matrix
    sensitivity_matrix = np.load(path)

    # Crop the sensitivity matrix
    cropped_sensitivity = sensitivity_matrix[:,xmin:xmax, ymin:ymax]
    
    # Load and crop the wetmask
    #drymask_path = 'drymask.npy'
    #drymask = np.load(drymask_path)
    #wetmask = np.where(drymask == 0, 1, 0)  # Invert the mask
    #cropped_wetmask = wetmask[xmin:xmax, ymin:ymax]
    # Reshape wetmas to [1, height, width] for masking
    #cropped_wetmask = cropped_wetmask.reshape(1, *cropped_wetmask.shape)

    #print(cropped_sensitivity.shape, cropped_wetmask.shape)
    # Apply the wetmask to the sensitivity
    #masked_sensitivity = np.ma.masked_array(
    #    cropped_sensitivity,
    #    mask=~cropped_wetmask.astype(bool)  # Invert wetmask to get drymask
    #)
    #mask_shape = ~cropped_wetmask.astype(bool)
    #expanded_mask = np.broadcast_to(mask_shape, (210, cropped_sensitivity.shape[1], cropped_sensitivity.shape[2]))
    #masked_sensitivity = np.ma.masked_array(
    #    cropped_sensitivity,
    #    mask=expanded_mask
    #)
    
    return cropped_sensitivity

def plot_sensitivity_monthly(t_months, t_end, t_end2, loc, map_dims, output_pixel, 
                             var_in, var_out, ch_in, ch_out, folder = '.'):
    """
    Create a 3x4 grid of monthly sensitivity plots, each with its own color scale
    
    Parameters:
    -----------
    t_months : list
        List of 12 monthly time points
    t_end : int
        End time point
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
    """
    # Sort months in descending order (most recent first)
    t_months = sorted(t_months, reverse=True)
    
    # Set up the figure with 3x4 grid
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 4, figure=fig, wspace=0.3, hspace=0.4)
    
    # Get output pixel coordinates
    output_lat, output_lon = output_pixel
    
    # Create a list to store all masked sensitivities
    all_sensitivities = []
    all_colorbars = []  # Store colorbar objects
    
    # First, load all data
    plot_path = Path(f'{folder}/{var_in}.npy')

    if plot_path.exists():
        masked_sensitivity = load_and_crop_sensitivity(plot_path, map_dims)
        print(np.nanmax(masked_sensitivity), np.nanmin(masked_sensitivity))
        print(f"Loaded sensitivity for {var_in} from {plot_path}")
    else:
        print(f"File not found: {plot_path}")
        return
    for i, month in enumerate(t_months):
        month_sensitivity = masked_sensitivity[-i-1, :, :]
        all_sensitivities.append(month_sensitivity)
    
    # Calculate tick positions and labels ONCE before the loop
    xmin, xmax, ymin, ymax = map_dims
    row_indices = np.arange(xmin, xmax)
    col_indices = np.arange(ymin, ymax)
    
    # Set ticks at appropriate positions using the shape of the first sensitivity matrix
    # (all should have the same shape)
    sample_sensitivity = all_sensitivities[0]
    x_pos = np.arange(sample_sensitivity.shape[1])
    y_pos = np.arange(sample_sensitivity.shape[0])
    
    # Calculate tick positions and labels
    xticks = 20 #5
    yticks = 20 #5
    
    x_tick_pos = x_pos[::xticks]
    x_tick_labs = col_indices[::xticks]
    
    y_tick_pos = y_pos[::yticks]
    y_tick_labs = row_indices[::yticks]
    
    # Now create each subplot
    for i, month in enumerate(t_months):
        # Calculate grid position (row, col)
        row = i // 4
        col = i % 4
        
        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        
        in_time, out_time = month, t_end
        masked_sensitivity = all_sensitivities[i]
        
        # Create a custom colormap with black for masked values
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad('black', 0.5)
        
        # Calculate individual color scale (symmetric around zero)
        abs_max = np.nanmax(np.abs(masked_sensitivity))#.compressed()))
        vmin, vmax = -abs_max, abs_max
        print(f"Month {i+1}: Using color limits [{vmin}, {vmax}]")
        
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
        all_colorbars.append(cbar)
        
        # Calculate months back from t_end
        months_back = (t_end - month) // 6  # Assuming 6 time units per month
        
        # Add title with months back
        ax.set_title(f'{months_back} months back', fontsize=12)
        
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
    
    # Clean up variable names for display
    var_in_clean = var_in.replace('(odd)','').replace('(even)','')
    var_out_clean = var_out.replace('(odd)','').replace('(even)','')
    
    # Add super title
    fig.suptitle(f'Monthly Sensitivity of {var_out_clean} at ({output_lat},{output_lon}) at t={t_end}\n'
                 f'to {var_in_clean} across map over preceding year', 
                 fontsize=16, y=0.98)
    
    # Create Plots directory if it doesn't exist
    Path("Plots").mkdir(exist_ok=True)
    
    # Save the figure
    view_name = f"({map_dims[0]},{map_dims[1]})x({map_dims[2]},{map_dims[3]})"
    filename = f'Plots/monthly_mitgcm_sensitivity_{view_name}_chin[{var_in_clean}]_chout[{var_out_clean}].png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved monthly grid plot with individual scales to {filename}")

if __name__ == "__main__":
    # Set parameters
    t_end = 72  # End time (approx. a year)
    t_end2 = 77 # End of range
    t_months = [t_end - 6*i for i in range(1, 13)]  # 12 months back
    
    #loc = '131,289'  # Nantucket
    loc = '126,324'  # North Atlantic Ocean
    #loc = '90,180'  # Equatorial Pacific Ocean

    #x_out, y_out = 126, 324  # North Atlantic
    x_out, y_out = eval(loc)   # Equatorial Pacific
    # Define output pixel (center of interest)
    output_pixel = (x_out,y_out)  # Center coordinates (lat, lon)
    
    # Define map dimensions - adjust as needed
    delta = 10
    map_dims = [x_out-delta , x_out+delta, y_out-delta , y_out+delta ]  # [xmin, xmax, ymin, ymax]
    #map_dims = [0, 180, 0, 360]  # Global view for now
    # Import variable dictionary from misc if it exists
    try:
        from misc import var_dict
    except ImportError:
        # Define default variable dictionary if import fails
        var_dict = {
            'zos(even)': 76,
            'hfds': 153,
            'hfds_anomalies': 154,
            'tauuo': 155,
            'tauvo': 156
        }
    
    # Choose variables to plot

    vars_in = ['hfds',]# 'hfds_anomalies', 'zos(even)', 'tauuo', 'tauvo']
    #var_in = 'hfds'  # Input variable
    #var_in = 'hfds_anomalies'  # Input variable
    #var_in = 'zos(even)'
    #var_in = 'tauuo'  # Input variable
    #var_in = 'tauvo'  # Input variable
    var_out = 'zos(even)'  # Output variable
    
    
    for var_in in vars_in:
        
        # Get channel numbers from dictionary
        ch_in = var_dict[var_in]
        ch_out = var_dict[var_out]
        folder = 'Converted_NETcdf/emu_adj_48_48_1_2_209_1/numpy_output' #'adjoint_arrays/North_Atlantic'#'

        # Create the monthly sensitivity grid plot
        plot_sensitivity_monthly(t_months, t_end, t_end2, loc, map_dims, output_pixel, 
                            var_in, var_out, ch_in, ch_out, folder=folder)