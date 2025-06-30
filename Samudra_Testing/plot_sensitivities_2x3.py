import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray as xr

# Configuration flags
full_map = False
small_map = True

x = 1.5
plt.rcParams['font.size'] = 24/2*x
plt.rcParams['axes.titlesize'] = 32/2*x
plt.rcParams['axes.labelsize'] = 28/2*x
plt.rcParams['xtick.labelsize'] = 24/2*x
plt.rcParams['ytick.labelsize'] = 24/2*x
plt.rcParams['legend.fontsize'] = 24/2*x

plt.rcParams['axes.titlepad'] = 22
plt.rcParams['figure.subplot.wspace'] = 0.1 if full_map else -0.25 if small_map else -0.25
plt.rcParams['figure.subplot.hspace'] = -0.5 if full_map else 0.25 if small_map else 0.25

def denormalize_sensitivity(sensitivity_tensor, var_out, var_in, data_std):
    """
    Denormalize the sensitivity tensor based on the variable configurations.
    """
    var_out_clean = var_out.replace('(odd)','').replace('(even)','')
    var_in_clean = var_in.replace('(odd)','').replace('(even)','')
    
    std_out = data_std[var_out_clean].values.item()
    std_in = data_std[var_in_clean].values.item()
    
    denorm_factor = std_out / std_in
    return sensitivity_tensor * denorm_factor

def load_and_crop_samudra_sensitivity(path, map_dims, var_out, var_in, denormalize=True):
    """Load and process Samudra sensitivity data."""
    xmin, xmax, ymin, ymax = map_dims
    
    sensitivity_matrix = np.load(path)
    sensitivity_matrix = sensitivity_matrix.reshape(180, 360)

    if denormalize:
        data_std = xr.open_zarr("./data_std.zarr")
        sensitivity_matrix = denormalize_sensitivity(sensitivity_matrix, var_out, var_in, data_std)

    cropped_sensitivity = sensitivity_matrix[xmin:xmax, ymin:ymax]
    
    # Apply wetmask
    drymask_path = 'drymask.npy'
    if Path(drymask_path).exists():
        drymask = np.load(drymask_path)
        wetmask = np.where(drymask == 0, 1, 0)
        cropped_wetmask = wetmask[xmin:xmax, ymin:ymax]

        masked_sensitivity = np.ma.masked_array(
            cropped_sensitivity,
            mask=~cropped_wetmask.astype(bool)
        )
    else:
        masked_sensitivity = cropped_sensitivity
    
    return masked_sensitivity

def load_and_crop_mitgcm_sensitivity(path, map_dims):
    """Load and process MITgcm sensitivity data."""
    xmin, xmax, ymin, ymax = map_dims
    
    sensitivity_matrix = np.load(path)
    cropped_sensitivity = sensitivity_matrix[:, xmin:xmax, ymin:ymax]
    
    return cropped_sensitivity

def plot_sensitivity_grid(t_values, map_dims, output_pixel, var_in, var_out, 
                         data_source='samudra', folder='.', circle=True, 
                         denormalize=True, ch_in=None, ch_out=None, t_end=None,
                         grid_shape=(2, 3), time_unit='days'):
    """
    Create a grid of sensitivity plots for either Samudra or MITgcm data.
    
    Parameters:
    -----------
    t_values : list
        List of time values or day lags
    map_dims : list
        [xmin, xmax, ymin, ymax] defining map boundaries
    output_pixel : tuple
        (lat, lon) coordinates of output pixel
    var_in, var_out : str
        Input and output variable names
    data_source : str
        'samudra' or 'mitgcm'
    folder : str
        Folder containing sensitivity files
    circle : bool
        Whether to add circle at output pixel
    denormalize : bool
        Whether to denormalize Samudra data
    ch_in, ch_out : int
        Channel numbers (for Samudra)
    t_end : int or str
        End time (for Samudra)
    grid_shape : tuple
        (rows, cols) for subplot grid
    time_unit : str
        'days' or 'steps' for time labeling
    """
    rows, cols = grid_shape
    fig = plt.figure(figsize=(22, 12))
    gs = gridspec.GridSpec(rows, cols, figure=fig)
    
    output_lat, output_lon = output_pixel
    xmin, xmax, ymin, ymax = map_dims
    
    # Load all data
    all_sensitivities = []
    
    if data_source == 'mitgcm':
        # Load MITgcm data (single file with time dimension)
        plot_path = Path(f'{folder}/{var_in}.npy')
        if plot_path.exists():
            masked_sensitivity = load_and_crop_mitgcm_sensitivity(plot_path, map_dims)
            print(f"MITgcm data shape: {masked_sensitivity.shape}")
            
            # Select time slices based on t_values
            for i, t in enumerate(t_values):
                time_lag = t // 7  # One week per time step
                time_sensitivity = masked_sensitivity[-time_lag-1, :, :]
                all_sensitivities.append(time_sensitivity)
        else:
            print(f"File not found: {plot_path}")
            return
    
    else:  # Samudra
        for i, t in enumerate(t_values):
            if t_end == '292-297':
                in_time = (1460 - t) // 5
                out_time = t_end
            else:
                in_time = (t_end - t) // 5
                out_time = (t_end) // 5

            # Try different file patterns
            possible_paths = [
                Path(f'{folder}/avg_sensitivity_chin[{ch_in}]_chout[{ch_out}]_loc[{output_lat},{output_lon}]_t[{in_time},{out_time}].npy'),
                Path(f'{folder}/chunk_sensitivity_chin[{ch_in}]_chout[{ch_out}]_t[{in_time},{out_time}].npy')
            ]
            
            found_file = False
            for plot_path in possible_paths:
                if plot_path.exists():
                    masked_sensitivity = load_and_crop_samudra_sensitivity(
                        plot_path, map_dims, var_out, var_in, denormalize
                    )
                    all_sensitivities.append(masked_sensitivity)
                    found_file = True
                    break
            
            if not found_file:
                print(f"No file found for time {t}")
                return
    
    # Calculate tick positions
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
    for i, t in enumerate(t_values):
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
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.update_ticks()
        
        # Title
        time_lag = t
        if data_source == 'samudra' and t_end != '292-297':
            # For Samudra, show as time step difference
            ax.set_title(f'{time_lag} {time_unit} back ($t_1-t_0 = {(t)//5}$)')
        else:
            ax.set_title(f'{time_lag} {time_unit} back')
        
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
    
    # Clean variable names and create title
    var_in_clean = var_in.replace('(odd)','').replace('(even)','')
    var_out_clean = var_out.replace('(odd)','').replace('(even)','')
    
    # Define unit mapping for variables
    var_units = {
        'zos': 'm',
        'tauuo': 'Pa',
        'tauvo': 'Pa', 
        'hfds': 'W m$^{-2}$',
        'hfds_anomalies': 'W m$^{-2}$',
    }
    
    # Global coordinates mapping
    samudra_coords_to_global_coords = {
        (126, 324): ('36N', '36W'),  
        (90, 180): ('0N', '0W'),  
        (131, 289): ('41N', '71W')
    }
    
    if tuple(output_pixel) in samudra_coords_to_global_coords:
        display_coords = samudra_coords_to_global_coords[tuple(output_pixel)]
    else:
        display_coords = f"({output_pixel[0]},{output_pixel[1]})"
    
    # Create unit string if both units are available
    out_unit = var_units.get(var_out_clean, '')
    in_unit = var_units.get(var_in_clean, '')
    unit_str = f" [{out_unit}/{in_unit}]" if (out_unit and in_unit) else ""
    
    # Super title
    model_name = "MITgcm" if data_source == 'mitgcm' else "Samudra"
    final_day = 1460 if data_source == 'mitgcm' else (1460 if t_end == '292-297' else t_end)
    
    y_pos = 0.85 if full_map else 1.0 if small_map else 0.95
    
    fig.suptitle(f'{model_name}: Sensitivity of {var_out_clean} at ({display_coords[0]},{display_coords[1]}) at day {final_day}\n'
                 f'wrt {var_in_clean} across map over various time scales{unit_str}', 
                 y=y_pos)
    
    # Save
    Path("Plots").mkdir(exist_ok=True)
    view_name = f"({map_dims[0]},{map_dims[1]})x({map_dims[2]},{map_dims[3]})"
    filename = f'Plots/grid_{data_source}_sensitivity_[{display_coords[0]},{display_coords[1]}]_{view_name}_chin[{var_in_clean}]_chout[{var_out_clean}].png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved grid plot to {filename}")

if __name__ == "__main__":
    
    # =================================================================
    # DEMO 1: Replicating plot_samudra.py behavior
    # =================================================================
    print("="*60)
    print("DEMO 1: Replicating plot_samudra.py behavior")
    print("="*60)
    
    # Import variable dictionary
    try:
        import misc
        var_dict = misc.var_dict
    except ImportError:
        var_dict = {
            'hfds': 156,
            'tauuo': 154,
            'tauvo': 155,
            'zos(even)': 76
        }
        print("Warning: Could not import var_dict from misc, using defaults")
    
    # Parameters from plot_samudra.py
    t_end = '292-297'
    t_days = [10*i for i in range(1, 7)]  # [10, 20, 30, 40, 50, 60]
    
    vars_in = ['hfds']
    var_out = 'zos(even)'
    
    locations = ['(126,324)']  # North Atlantic Ocean
    
    for loc in locations:
        xout, yout = eval(loc)
        output_pixel = (xout, yout)
        
        # Define map dimensions
        delta = 20
        map_dims = [xout-delta, xout+delta+1, yout-delta, yout+delta+1]
        
        for var_in in vars_in:
            ch_in = var_dict[var_in]
            ch_out = var_dict[var_out]
            folder = 'MITgcm_Replication/'
            
            print(f"Creating Samudra grid for {var_in} -> {var_out} at {loc}")
            plot_sensitivity_grid(
                t_days, map_dims, output_pixel, var_in, var_out,
                data_source='samudra', folder=folder, circle=False, 
                denormalize=True, ch_in=ch_in, ch_out=ch_out, t_end=t_end,
                time_unit='days'
            )
    
    # =================================================================
    # DEMO 2: Replicating plot_mitgcm.py behavior  
    # =================================================================
    print("\n" + "="*60)
    print("DEMO 2: Replicating plot_mitgcm.py behavior")
    print("="*60)
    
    # Parameters from plot_mitgcm.py
    t_days = [7, 21, 28, 42, 49, 63]  # Time lags in days
    
    vars_in = ['hfds', 'tauuo', 'tauvo']
    var_out = 'zos(even)'
    
    locations = ['(126,324)']  # North Atlantic Ocean
    
    for loc in locations:
        xout, yout = eval(loc)
        output_pixel = (xout, yout)
        folder = f'Converted_NETcdf/{loc}'
        
        # Define map dimensions  
        delta = 20
        map_dims = [xout-delta, xout+delta+1, yout-delta, yout+delta+1]
        
        for var_in in vars_in:
            ch_in = var_dict.get(var_in, 156)  # Default fallback
            ch_out = var_dict.get(var_out, 76)
            
            print(f"Creating MITgcm grid for {var_in} -> {var_out} at {loc}")
            plot_sensitivity_grid(
                t_days, map_dims, output_pixel, var_in, var_out,
                data_source='mitgcm', folder=folder, circle=False,
                ch_in=ch_in, ch_out=ch_out, time_unit='days'
            )
    
    print("\nAll grid plots completed!")