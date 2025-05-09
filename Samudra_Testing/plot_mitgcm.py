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
    
    return cropped_sensitivity

def plot_sensitivity_grid(t_days, loc, map_dims, output_pixel, 
                          var_in, var_out, ch_in, ch_out, 
                          folder = '.', circle=True):
    """
    Create a 2x3 grid of sensitivity plots, each with its own color scale
    
    Parameters:
    -----------
    t_times : list
        List of 6 time points
    t_end : int
        End time point
    loc : str
        Location string (for file naming)
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
    #t_days = sorted(t_days, reverse=True)

    # Set up the figure with 2x3 grid
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.3, 
                           hspace=-0.2) ############## OVERHERE!!!!!!!!!!!!!!!!###################### -0.7
    
    # Get output pixel coordinates
    output_lat, output_lon = output_pixel
    
    # Create a list to store all masked sensitivities
    all_sensitivities = []
    
    # First, load all data
    plot_path = Path(f'{folder}/{var_in}.npy')

    if plot_path.exists():
        masked_sensitivity = load_and_crop_sensitivity(plot_path, map_dims)
        print(np.nanmax(masked_sensitivity), np.nanmin(masked_sensitivity))
        print(f"Loaded sensitivity for {var_in} from {plot_path}")
    else:
        print(f"File not found: {plot_path}")
        return
    
    # Select 6 time points from the loaded data
    for i, t in enumerate(t_days):
        time_lag = t // 7 # One week per time step
        time_sensitivity = masked_sensitivity[-time_lag, :, :]
        all_sensitivities.append(time_sensitivity)
    
    # Calculate tick positions and labels ONCE before the loop
    xmin, xmax, ymin, ymax = map_dims
    row_indices = np.arange(xmin, xmax)
    col_indices = np.arange(ymin, ymax)
    
    # Set ticks at appropriate positions
    sample_sensitivity = all_sensitivities[0]
    x_pos = np.arange(sample_sensitivity.shape[1])
    y_pos = np.arange(sample_sensitivity.shape[0])
    
    # Calculate tick positions and labels
    xticks = 40
    yticks = 20
    
    x_tick_pos = x_pos[::xticks]
    x_tick_labs = col_indices[::xticks]
    
    y_tick_pos = y_pos[::yticks]
    y_tick_labs = row_indices[::yticks]
    
    # Now create each subplot
    for i, t in enumerate(t_days):
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
        abs_max = np.nanmax(np.abs(masked_sensitivity))
        vmin, vmax = -abs_max, abs_max
        print(f"Time {t}: Using color limits [{vmin}, {vmax}]")
        
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
        
        # Calculate months back or use time lag terminology
        time_lag = t
        
        # Add title with time information
        ax.set_title(f'{time_lag} days back', fontsize=12)
        
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
    #var_in_latex = var_in_clean.replace('_', '\_')
    #var_out_latex = var_out_clean.replace('_', '\_')

    samudra_coords_to_global_coords = {
        (126, 324): ('36N', '36W'),  
        (90, 180): ('0', '0'),  
        (131, 289): ('41N', '71W')
    }

    output_pixel = samudra_coords_to_global_coords[output_pixel]
    output_lat, output_lon = output_pixel

    
    # Add super title
    fig.suptitle(f'MITgcm: Sensitivity of {var_out_clean} at ({output_lat},{output_lon}) at day 1460\n'
                 f'wrt {var_in_clean} across map over various time scales', 
                 fontsize=16, 
                 y=0.9) ################################ OVERHERE!!!!!!!!!!!!!!!!###################### 0.75 
    
    # Create Plots directory if it doesn't exist
    Path("Plots").mkdir(exist_ok=True)
    
    # Save the figure
    view_name = f"({map_dims[0]},{map_dims[1]})x({map_dims[2]},{map_dims[3]})"
    filename = f'Plots/grid_mitgcm_sensitivity_[{output_lat},{output_lon}]_{view_name}_chin[{var_in_clean}]_chout[{var_out_clean}].png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved grid plot with individual scales to {filename}")

if __name__ == "__main__":
    # Set parameters - select 6 months for 2x3 grid
    t_days = [70, 140, 210, 350, 490, 700] #[7,21,28,42,49,63]#[7*i for i in range(1,7)]#

    # Try to import variable dictionary from misc
    from misc import var_dict

    
    # Choose variables to plot
    vars_in = ['hfds',]# 'tauuo', 'tauvo']  # Input variable
    var_out = 'zos(even)'  # Output variable
    
    for loc in ['(126,324)',]:# '(90,180)', '(131,289)']:
        #loc = '(126,324)'  # North Atlantic Ocean
        xout, yout = eval(loc)
        output_pixel = (xout, yout)  # Coordinates of the output pixel
        folder = f'Converted_NETcdf/{loc}'
    
        # Define map dimensions
        delta = 20
        #map_dims = [xout-delta, xout+delta+1, yout-delta, yout+delta+1]  # [xmin, xmax, ymin, ymax]
        map_dims = [0, 180, 180, 360]  # [xmin, xmax, ymin, ymax] # [0, 180, 0, 360]

        for var_in in vars_in:
            # Get channel numbers from dictionary
            ch_in = var_dict[var_in]
            ch_out = var_dict[var_out]

            

            # Create the grid sensitivity plot
            plot_sensitivity_grid(t_days, loc, map_dims, output_pixel, 
                            var_in, var_out, ch_in, ch_out, folder=folder, circle= False)