import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import misc

### PARAMETERS ###

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


def plot(path, map_dims, #Variables used to make the graph
         t0, t1, output_pixel, 
         output_var, input_var, # Pixels used to label the graph
         circle_coords=None, circle_radius=3, circle_color='lime', # Circle parameters
         xticks=20, yticks=20, #Variables used to add ticks to the graph
    ):

    ### PRE-PROCESSING ###
    xmin, xmax, ymin, ymax = map_dims
    output_lat, output_lon = output_pixel

    view_name = f"({xmin},{xmax})x({ymin},{ymax})"

    # Load the sensitivity matrix
    sensitivity_matrix = np.load(path)
    sensitivity_matrix = sensitivity_matrix.reshape(180, 360)

    cropped_sensitivity = sensitivity_matrix[xmin:xmax, ymin:ymax]

    drymask_path = 'drymask.npy'
    drymask = np.load(drymask_path)
    wetmask = np.where(drymask == 0, 1, 0)  # Invert the mask
    has_wetmask = True

    cropped_wetmask = wetmask[xmin:xmax, ymin:ymax]

    masked_sensitivity = np.ma.masked_array(
            cropped_sensitivity,
            mask=~cropped_wetmask.astype(bool)  # Invert wetmask to get drymask
        )
        
    ### PLOTTING ###
    # Plot the cropped and masked sensitivity matrix with indices as labels
    plt.figure(figsize=(10, 8))

    # Create symmetric limits around zero
    abs_max = np.max(np.abs(masked_sensitivity))
    vmin, vmax = -abs_max, abs_max
    print(f"Using symmetric color limits: [{vmin}, {vmax}]")

    # Create a custom colormap based on RdBu_r with black for masked values
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad('black', 0.5)
    
    im = plt.imshow(masked_sensitivity, cmap=cmap, aspect='equal', origin='lower', 
                    vmin=vmin, vmax=vmax)

    # Get the row and column indices for the cropped region
    row_indices = np.arange(xmin, xmax)
    col_indices = np.arange(ymin, ymax)

    # Set ticks at appropriate positions
    y_positions = np.arange(cropped_sensitivity.shape[0])
    x_positions = np.arange(cropped_sensitivity.shape[1])

    # Calculate tick positions and labels that match in length
    x_tick_positions = x_positions[::xticks]
    x_tick_labels = col_indices[::xticks]
    
    y_tick_positions = y_positions[::yticks]
    y_tick_labels = row_indices[::yticks]
    
    # Make sure we have the same number of positions and labels
    min_x_len = min(len(x_tick_positions), len(x_tick_labels))
    min_y_len = min(len(y_tick_positions), len(y_tick_labels))
    
    # Display actual matrix indices on the axes
    plt.xticks(x_tick_positions[:min_x_len], x_tick_labels[:min_x_len])
    plt.yticks(y_tick_positions[:min_y_len], y_tick_labels[:min_y_len])

    # After plotting the main sensitivity matrix but before saving
    if circle_coords is not None:
        # Convert from global to cropped coordinates
        circle_y, circle_x = circle_coords
        # Adjust for cropping
        circle_y_adj = circle_y - xmin
        circle_x_adj = circle_x - ymin
        
        # Only draw the circle if it's within the cropped region
        if (0 <= circle_y_adj < cropped_sensitivity.shape[0] and 
            0 <= circle_x_adj < cropped_sensitivity.shape[1]):
            # Create circle
            circle = plt.Circle((circle_x_adj, circle_y_adj), circle_radius, 
                            color=circle_color, fill=False, linewidth=2)
            plt.gca().add_patch(circle)

    plt.colorbar(label='Sensitivity Value')
    output_var = output_var.replace('(odd)','').replace('(even)','')
    input_var = input_var.replace('(odd)','').replace('(even)','')

    output_latex = output_var.replace('_', '\\_')
    input_latex = input_var.replace('_', '\\_')
    # in a 2x2 with top-left at
    numerator = f'\\partial \\left( {output_latex} \\text{{ at }}({output_lat},{output_lon}), t={t1} \\right)'
    denominator = f'\\partial \\left( {input_latex} \\text{{ across map}}, t={t0} \\right)'
    plt.title(f'${numerator} / {denominator}$')

    plt.xlabel('Longitude Index')
    plt.ylabel('Latitude Index')
    plt.grid(False)
    plt.tight_layout()

    # Create Plots directory if it doesn't exist
    Path("Plots").mkdir(exist_ok=True)
    
    name = f'Plots/avg_adjoint_map_{view_name}_chin[{input_var}]_chout[{output_var}]_t[{t0},{t1}].png'
    print(f"Saving plot to: {name}")
    plt.savefig(name, dpi=300, bbox_inches='tight')

    plt.close('all')
    print("Plot saved with symmetric color scale around zero.")


#t=0 is the start of 2014
# We want t_end to be December 2015
t_start = 0 
# 699 days between January 2014 and December 2015: 700/5=140
t_end = 10

t_2year =   0 # A little less than 2 years from t_end
t_1year =   140 - 73 # 1 year back from t_end
t_6months = 140 - 36 # 6 months back from t_end
t_1month =  140 - 6 # 1 month back from t_end


# Import var_dict from misc if it exists, otherwise define it here
from misc import var_dict

#var_in = 'hfds'
#var_in = 'hfds_anomalies'
#var_in = 'tauuo'
#var_in = 'tauvo'
var_in = 'zos(even)'
var_out = 'zos(even)' 

#ch_in = var_dict[var_in]
#ch_out = var_dict[var_out]

ch_in,ch_out=76,76

channel_to_var = misc.get_channel_to_var()
var_out = channel_to_var[ch_out]
var_in = channel_to_var[ch_in]

# center: (131, 289) corresponds to Nantucket #
# center: (90,180) corresponds to equatorial Pacific 
# Explicitly define map dimensions
delta = 50
map_dims =  [90-delta,90+delta+1,180-delta,180+delta+1] # [xmin, xmax, ymin, ymax]
#map_dims = [131-delta, 131+delta, 289-delta, 289+delta]  # Full global map
#map_dims = [111, 152+20, 269, 310+50]
map_dims = [50,130,110,250]
initial_times_dict = {'zos(even)': [t_1month, t_6months, t_1year, t_2year],
                      'tauuo': [t_1month, t_6months, t_1year],
                      'tauvo': [t_1month, t_6months, t_1year],
                      'hfds': [t_1year, t_2year],
                      'hfds_anomalies': [t_1year, t_2year],}

#initial_times = initial_times_dict[var_in] # [t_1month, t_6months, t_1year]#
initial_times = [0,2,4,6,8]#[0,2,4,6,8]
#map_dims = [0, 180, 0, 360]  # Full global map


output_pixel = (90, 180)#(131,289)#  # Coordinates for the output pixel

for initial_time in initial_times:
    in_time, out_time = initial_time, t_end
    plot_path = Path(f'avg_sensitivity_chin[{ch_in}]_chout[{ch_out}]_t[{in_time},{out_time}].npy')
    #plot_path = Path(f'sensitivity_arrays/Equatorial_Pacific/chunk_sensitivity_chin[{ch_in}]_chout[{ch_out}]_t[{in_time},{out_time}].npy')
    
    print("Data retrieved from:", plot_path)
    if plot_path.exists():
        plot(plot_path, map_dims=map_dims, t0=in_time, t1=out_time, 
             output_pixel=output_pixel, output_var=var_out, input_var=var_in,
             circle_coords=None, circle_radius=2, circle_color='black',
             xticks=10, yticks=10)
        print(f"Plot saved for initial time {initial_time} with output variable {var_out} and input variable {var_in}.")
    else:
        print(f"File not found: {plot_path}")