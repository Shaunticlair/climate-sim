import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import misc
import matplotlib.ticker as ticker

# Set global styling
x = 1.5
plt.rcParams['font.size'] = 24/2*x
plt.rcParams['axes.titlesize'] = 0.75*32/2*x
plt.rcParams['axes.labelsize'] = 28/2*x
plt.rcParams['xtick.labelsize'] = 24/2*x
plt.rcParams['ytick.labelsize'] = 24/2*x
plt.rcParams['legend.fontsize'] = 24/2*x
plt.rcParams['axes.titlepad'] = 22
plt.rcParams['figure.subplot.wspace'] = -0.25
plt.rcParams['figure.subplot.hspace'] = 0.25

def center_bounds(view_size, centerx, centery):
    """Calculate map bounds based on view size and center coordinates."""
    if view_size == "tiny":
        xmin, xmax = centerx-3, centerx+4
        ymin, ymax = centery-3, centery+4
    elif view_size == "local":
        xmin, xmax = centerx-20, centerx+20
        ymin, ymax = centery-20, centery+20
    elif view_size == "global":
        xmin, xmax = 0, 180
        ymin, ymax = 0, 360
    elif isinstance(view_size, list):
        deltax, deltay = view_size
        xmin, xmax = centerx-deltax, centerx+deltax+1
        ymin, ymax = centery-deltay, centery+deltay+1
    else:
        raise ValueError(f"Unknown view_size: {view_size}")
    
    return xmin, xmax, ymin, ymax

def load_and_crop_sensitivity(path, map_dims, data_format='adjoint', apply_wetmask=True):
    """
    Load sensitivity data and crop to specified dimensions.
    
    Parameters:
    -----------
    path : Path or str
        Path to the sensitivity file
    map_dims : list
        [xmin, xmax, ymin, ymax] defining map boundaries
    data_format : str
        'adjoint' - reshape to (180,360), 'perturbation' - use as-is, 'mitgcm' - 3D time series
    apply_wetmask : bool
        Whether to apply wetmask for ocean/land masking
    
    Returns:
    --------
    masked_sensitivity : numpy.ma.MaskedArray or numpy.ndarray
        Processed sensitivity data
    """
    xmin, xmax, ymin, ymax = map_dims
    
    # Load the sensitivity matrix
    sensitivity_matrix = np.load(path)
    
    # Handle different data formats
    if data_format == 'adjoint':
        # Reshape adjoint data to (180, 360)
        if len(sensitivity_matrix.shape) > 2:
            sensitivity_matrix = sensitivity_matrix.reshape(180, 360)
        cropped_sensitivity = sensitivity_matrix[xmin:xmax, ymin:ymax]
    elif data_format == 'perturbation':
        # Use perturbation data as-is (already shaped)
        cropped_sensitivity = sensitivity_matrix
    elif data_format == 'mitgcm':
        # MITgcm data is 3D (time, lat, lon), extract specific time slice
        if len(sensitivity_matrix.shape) == 3:
            # For now, take the first time slice - this can be parameterized
            sensitivity_matrix = sensitivity_matrix[0, :, :]
        cropped_sensitivity = sensitivity_matrix[xmin:xmax, ymin:ymax]
    else:
        raise ValueError(f"Unknown data_format: {data_format}")
    
    if not apply_wetmask:
        return cropped_sensitivity
    
    # Apply wetmask
    drymask_path = 'drymask.npy'
    if Path(drymask_path).exists():
        drymask = np.load(drymask_path)
        wetmask = np.where(drymask == 0, 1, 0)  # Invert the mask
        cropped_wetmask = wetmask[xmin:xmax, ymin:ymax]
        
        masked_sensitivity = np.ma.masked_array(
            cropped_sensitivity,
            mask=~cropped_wetmask.astype(bool)
        )
    else:
        masked_sensitivity = cropped_sensitivity
    
    return masked_sensitivity

def plot_sensitivity_map(path, map_dims, t0, t1, output_pixel, output_var, input_var,
                        data_format='adjoint', circle_coords=None, circle_radius=3, 
                        circle_color='lime', xticks=20, yticks=20, output_dir='Plots',
                        filename_prefix=None):
    """
    Create a single sensitivity map plot.
    
    Parameters:
    -----------
    path : Path or str
        Path to sensitivity file
    map_dims : list
        [xmin, xmax, ymin, ymax] defining map boundaries
    t0, t1 : int
        Initial and final time steps
    output_pixel : tuple
        (lat, lon) coordinates of output pixel
    output_var, input_var : str
        Variable names for labeling
    data_format : str
        Format of input data ('adjoint', 'perturbation', 'mitgcm')
    circle_coords : tuple, optional
        (lat, lon) for circle overlay
    circle_radius : int
        Radius of circle overlay
    circle_color : str
        Color of circle overlay
    xticks, yticks : int
        Tick spacing
    output_dir : str
        Directory to save plots
    filename_prefix : str, optional
        Prefix for output filename
    """
    xmin, xmax, ymin, ymax = map_dims
    output_lat, output_lon = output_pixel
    
    # Load and process data
    masked_sensitivity = load_and_crop_sensitivity(path, map_dims, data_format)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Create symmetric limits around zero
    if isinstance(masked_sensitivity, np.ma.MaskedArray):
        abs_max = np.max(np.abs(masked_sensitivity.compressed()))
    else:
        abs_max = np.max(np.abs(masked_sensitivity))
    
    vmin, vmax = -abs_max, abs_max
    print(f"Using symmetric color limits: [{vmin}, {vmax}]")
    
    # Create colormap
    cmap = plt.cm.RdBu_r.copy()
    if isinstance(masked_sensitivity, np.ma.MaskedArray):
        cmap.set_bad('black', 0.5)
    
    im = plt.imshow(masked_sensitivity, cmap=cmap, aspect='equal', origin='lower', 
                    vmin=vmin, vmax=vmax)
    
    # Set up ticks
    row_indices = np.arange(xmin, xmax)
    col_indices = np.arange(ymin, ymax)
    
    y_positions = np.arange(masked_sensitivity.shape[0])
    x_positions = np.arange(masked_sensitivity.shape[1])
    
    x_tick_positions = x_positions[::xticks]
    x_tick_labels = col_indices[::xticks]
    y_tick_positions = y_positions[::yticks]
    y_tick_labels = row_indices[::yticks]
    
    min_x_len = min(len(x_tick_positions), len(x_tick_labels))
    min_y_len = min(len(y_tick_positions), len(y_tick_labels))
    
    plt.xticks(x_tick_positions[:min_x_len], x_tick_labels[:min_x_len])
    plt.yticks(y_tick_positions[:min_y_len], y_tick_labels[:min_y_len])
    
    # Add circle if specified
    if circle_coords is not None:
        circle_y, circle_x = circle_coords
        circle_y_adj = circle_y - xmin
        circle_x_adj = circle_x - ymin
        
        if (0 <= circle_y_adj < masked_sensitivity.shape[0] and 
            0 <= circle_x_adj < masked_sensitivity.shape[1]):
            circle = plt.Circle((circle_x_adj, circle_y_adj), circle_radius, 
                            color=circle_color, fill=False, linewidth=2)
            plt.gca().add_patch(circle)
    
    # Labels and title
    cbar = plt.colorbar(im, label='Sensitivity Value')
    
    cbar.formatter.set_powerlimits((-2, 2))  # Adjust these limits as needed
    cbar.update_ticks()
    #cbar.ax.set_aspect('equal')
    
    
    # Clean variable names
    output_var_clean = output_var.replace('(odd)','').replace('(even)','')
    input_var_clean = input_var.replace('(odd)','').replace('(even)','')
    
    output_latex = output_var_clean.replace('_', '\\_')
    input_latex = input_var_clean.replace('_', '\\_')
    
    numerator = f'\\partial \\left( {output_latex} \\text{{ at }}({output_lat},{output_lon}), t={t1} \\right)'
    denominator = f'\\partial \\left( {input_latex} \\text{{ across map}}, t={t0} \\right)'
    plt.title(f'${numerator} / {denominator}$')
    
    plt.xlabel('Longitude Index')
    plt.ylabel('Latitude Index')
    plt.grid(False)
    plt.tight_layout()
    
    # Save plot
    Path(output_dir).mkdir(exist_ok=True)
    view_name = f"({xmin},{xmax})x({ymin},{ymax})"
    
    if filename_prefix is None:
        filename_prefix = data_format
    
    filename = f'{output_dir}/{filename_prefix}_map_{view_name}_chin[{input_var_clean}]_chout[{output_var_clean}]_t[{t0},{t1}].png'
    print(f"Saving plot to: {filename}")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved with symmetric color scale around zero.")

def plot_sensitivity_grid(t_times, t_end, map_dims, output_pixel, var_in, var_out, 
                         ch_in, ch_out, data_format='adjoint', folder='.', 
                         grid_shape=(2, 3), circle=True, time_slice_func=None):
    """
    Create a grid of sensitivity plots over multiple time points.
    
    Parameters:
    -----------
    t_times : list
        List of time points to plot
    t_end : int
        End time point
    map_dims : list
        [xmin, xmax, ymin, ymax] defining map boundaries
    output_pixel : tuple
        (lat, lon) coordinates of output pixel
    var_in, var_out : str
        Input and output variable names
    ch_in, ch_out : int
        Channel numbers
    data_format : str
        Format of input data
    folder : str
        Folder containing sensitivity files
    grid_shape : tuple
        (rows, cols) for subplot grid
    circle : bool
        Whether to add circle at output pixel
    time_slice_func : callable, optional
        Function to extract time slice from 3D data (for mitgcm)
    """
    # Set up the figure
    rows, cols = grid_shape
    fig = plt.figure(figsize=(22, 12))
    gs = gridspec.GridSpec(rows, cols, figure=fig)
    
    output_lat, output_lon = output_pixel
    xmin, xmax, ymin, ymax = map_dims
    
    # Load all data first
    all_sensitivities = []
    
    if data_format == 'mitgcm':
        # For MITgcm, load the single file and extract time slices
        plot_path = Path(f'{folder}/{var_in}.npy')
        if plot_path.exists():
            full_data = np.load(plot_path)
            for i, t in enumerate(t_times):
                if time_slice_func:
                    time_sensitivity = time_slice_func(full_data, t, i)
                else:
                    # Default: extract time slice based on index
                    time_idx = -len(t_times) + i
                    time_sensitivity = full_data[time_idx, xmin:xmax, ymin:ymax]
                all_sensitivities.append(time_sensitivity)
        else:
            print(f"File not found: {plot_path}")
            return
    else:
        # For adjoint/perturbation, load individual files
        for i, t in enumerate(t_times):
            if data_format == 'adjoint':
                plot_path = Path(f'{folder}/chunk_sensitivity_chin[{ch_in}]_chout[{ch_out}]_t[{t},{t_end}].npy')
            elif data_format == 'perturbation':
                plot_path = Path(f'{folder}/perturbation_grid_chin[{ch_in}]_chout[{ch_out}]_t[{t},{t_end}]_1e-2.npy')
            
            if plot_path.exists():
                sensitivity_data = load_and_crop_sensitivity(plot_path, map_dims, data_format)
                all_sensitivities.append(sensitivity_data)
            else:
                print(f"File not found: {plot_path}")
                return
    
    # Calculate common tick positions
    if all_sensitivities:
        sample_sensitivity = all_sensitivities[0]
        row_indices = np.arange(xmin, xmax)
        col_indices = np.arange(ymin, ymax)
        
        x_pos = np.arange(sample_sensitivity.shape[1])
        y_pos = np.arange(sample_sensitivity.shape[0])
        
        xticks = 40 if sample_sensitivity.shape[1] > 120 else 10
        yticks = 40 if sample_sensitivity.shape[0] > 120 else 10
        
        x_tick_pos = x_pos[::xticks]
        x_tick_labs = col_indices[::xticks]
        y_tick_pos = y_pos[::yticks]
        y_tick_labs = row_indices[::yticks]
    
    # Create subplots
    for i, t in enumerate(t_times):
        if i >= len(all_sensitivities):
            break
            
        row = i // cols
        col_idx = i % cols
        
        ax = fig.add_subplot(gs[row, col_idx])
        masked_sensitivity = all_sensitivities[i]
        
        # Create colormap
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad('black', 0.5)
        
        # Calculate color scale
        if isinstance(masked_sensitivity, np.ma.MaskedArray):
            abs_max = np.max(np.abs(masked_sensitivity.compressed()))
        else:
            abs_max = np.nanmax(np.abs(masked_sensitivity))
            
        vmin, vmax = -abs_max, abs_max
        print(f"Time {t}: Using color limits [{vmin}, {vmax}]")
        
        # Plot
        im = ax.imshow(masked_sensitivity, cmap=cmap, aspect='equal', origin='lower', 
                      vmin=vmin, vmax=vmax)
        

        
        # Title
        if data_format == 'mitgcm':
            time_lag = t
            ax.set_title(f'{time_lag} days back')
        else:
            time_lag = t_end - t
            ax.set_title(f'{time_lag} steps back')
        
        # Add circle
        if circle:
            circle_y_adj = output_lat - xmin
            circle_x_adj = output_lon - ymin
            
            if (0 <= circle_y_adj < masked_sensitivity.shape[0] and 
                0 <= circle_x_adj < masked_sensitivity.shape[1]):
                circle_obj = plt.Circle((circle_x_adj, circle_y_adj), 2, 
                                color='black', fill=False, linewidth=2)
                ax.add_patch(circle_obj)
        
        # Ticks and labels
        if row == rows - 1:  # Bottom row
            ax.set_xticks(x_tick_pos[:min(len(x_tick_pos), len(x_tick_labs))])
            ax.set_xticklabels(x_tick_labs[:min(len(x_tick_pos), len(x_tick_labs))], rotation=45)
            ax.set_xlabel('Longitude Index')
        else:
            ax.set_xticks([])
            
        if col_idx == 0:  # Left column
            ax.set_yticks(y_tick_pos[:min(len(y_tick_pos), len(y_tick_labs))])
            ax.set_yticklabels(y_tick_labs[:min(len(y_tick_pos), len(y_tick_labs))])
            ax.set_ylabel('Latitude Index')
        else:
            ax.set_yticks([])
    
    # Clean variable names
    var_in_clean = var_in.replace('(odd)','').replace('(even)','')
    var_out_clean = var_out.replace('(odd)','').replace('(even)','')
    
    # Global coordinates mapping
    samudra_coords_to_global_coords = {
        (126, 324): ('36N', '36W'),  
        (90, 180): ('0', '0'),  
        (131, 289): ('41N', '71W')
    }
    
    if tuple(output_pixel) in samudra_coords_to_global_coords:
        display_coords = samudra_coords_to_global_coords[tuple(output_pixel)]
    else:
        display_coords = f"({output_pixel[0]},{output_pixel[1]})"
    
    # Super title
    model_name = "MITgcm" if data_format == 'mitgcm' else "Samudra"
    end_day = 1460 if data_format == 'mitgcm' else t_end
    
    fig.suptitle(f'{model_name}: Sensitivity of {var_out_clean} at {display_coords} at day {end_day}\n'
                 f'wrt {var_in_clean} across map over various time scales', 
                 y=1.0)
    
    # Save
    Path("Plots").mkdir(exist_ok=True)
    view_name = f"({map_dims[0]},{map_dims[1]})x({map_dims[2]},{map_dims[3]})"
    filename = f'Plots/grid_{data_format}_sensitivity_{display_coords}_{view_name}_chin[{var_in_clean}]_chout[{var_out_clean}].png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved grid plot to {filename}")

if __name__ == "__main__":

    # Import variable dictionary
    try:
        from misc import var_dict
    except ImportError:
        var_dict = {}
        print("Warning: Could not import var_dict from misc")


    # =================================================================
    # DEMO 1: Plotting Samudra Autograd Adjoints
    # =================================================================
    print("="*60)
    print("DEMO 1: Plotting Samudra Autograd Adjoints")
    print("="*60)
    
    t_end = 12
    initial_times = [0, 2, 4, 6, 8, 10]
    var_in = 'zos(even)'
    var_out = 'zos(even)'
    ch_in, ch_out = 76, 76
    output_pixel = (90, 180)  # Equatorial Pacific
    delta = 10
    map_dims = [90-delta, 90+delta+1, 180-delta, 180+delta+1] # For a box around output_pixel
    map_dims = [0, 180, 0, 360]
    
    for initial_time in initial_times:
        in_time, out_time = initial_time, t_end
        plot_path = Path(f'adjoint_arrays/Equatorial_Pacific/chunk_sensitivity_chin[{ch_in}]_chout[{ch_out}]_t[{in_time},{out_time}].npy')
        
        if plot_path.exists():
            plot_sensitivity_map(
                plot_path, map_dims, in_time, out_time, output_pixel, 
                var_out, var_in, data_format='adjoint', filename_prefix='adjoint',
                xticks=40
            )
            print(f"Created adjoint plot for t=({in_time},{out_time})")
        else:
            print(f"File not found: {plot_path}")
    
    
    # =================================================================
    # DEMO 2: Plotting finite-difference approximations of adjoint
    # =================================================================
    print("\n" + "="*60)
    print("DEMO 2: Plotting finite-difference approximations of adjoint")
    print("="*60)
    
    t_end = 10
    initial_times = [0, 2, 4, 6, 8]
    var_in = 'zos(even)'
    var_out = 'zos(even)'
    ch_in, ch_out = 76, 76
    output_pixel = (90, 180)
    delta = 2
    map_dims = [90-delta, 90+delta+1, 180-delta, 180+delta+1]
    
    for initial_time in initial_times:
        in_time, out_time = initial_time, t_end
        plot_path = Path(f'perturbation_arrays/Short_Time_1e-2/perturbation_grid_chin[{ch_in}]_chout[{ch_out}]_t[{in_time},{out_time}]_1e-2.npy')
        
        if plot_path.exists():
            plot_sensitivity_map(
                plot_path, map_dims, in_time, out_time, output_pixel,
                var_out, var_in, data_format='perturbation', filename_prefix='perturb'
            )
            print(f"Created perturbation plot for t=({in_time},{out_time})")
        else:
            print(f"File not found: {plot_path}")