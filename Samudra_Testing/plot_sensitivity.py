import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

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

    if type(view_size) == list: #Manual mode
        deltax, deltay = view_size
        # Define the region of interest in the matrix for cropping
        xmin, xmax = centerx-deltax, centerx+deltax+1
        ymin, ymax = centery-deltay, centery+deltay+1


    return xmin, xmax, ymin, ymax


def plot(path, map_dims, #Variables used to make the graph
         t0, t1, output_pixel, 
         output_var, input_var, # Pixels used to label the graph
         xticks = 20, yticks = 20, #Variables used to add ticks to the graph
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

    im = plt.imshow(masked_sensitivity, cmap='RdBu_r', aspect='auto', origin='lower', 
                    vmin=vmin, vmax=vmax)

    # Get the row and column indices for the cropped region
    row_indices = np.arange(xmin, xmax)
    col_indices = np.arange(ymin, ymax)

    # Set ticks at appropriate positions
    y_positions = np.arange(cropped_sensitivity.shape[0])
    x_positions = np.arange(cropped_sensitivity.shape[1])

    # Display actual matrix indices on the axes
    plt.xticks(x_positions[::20], col_indices[::20])  # Show every 36th index for readability
    plt.yticks(y_positions[::20], row_indices[::20])  # Show every 18th index for readability

    plt.colorbar(label='Sensitivity Value')
    plt.title(f'$\\partial \\left( {output_var} \\text{{ at }}({output_lat},{output_lon}), t={t1} \\right) / \\partial \\left( {input_var} \\text{{ across map}}, t={t0} \\right)$')
    plt.xlabel('Longitude Index')
    plt.ylabel('Latitude Index')
    plt.grid(False)
    plt.tight_layout()

    name = f'Plots/adjoint_map_{view_name}_chout[{output_var}]_chin[{input_var}]_t[{t0},{t1}].png'
    print(name)
    plt.savefig(name, dpi=300, bbox_inches='tight')

    plt.close('all')
    print("Plots saved with symmetric color scale around zero.")

    



from misc import var_dict

var_in = 'hfds'
var_out = 'zos(even)'

ch_in = var_dict[var_in]
ch_out = var_dict[var_out]



t_start,t_end = 0,73
plot_path = Path(f"chunk_sensitivity_ch{ch_out}_t{t_start}-{t_end}.npy")

map_dims = center_bounds([20,20], 131, 289) # [xmin, xmax, ymin, ymax]
map_dims = [111, 152+20, 269, 310+20]
plot(plot_path, map_dims=map_dims, t0=t_start, t1=t_end, 
     output_pixel=(131, 298), output_var=var_out, input_var=var_in)