import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

x = 1.5
plt.rcParams['font.size'] = 24/2*x
plt.rcParams['axes.titlesize'] = 32/2*x
plt.rcParams['axes.labelsize'] = 28/2*x
plt.rcParams['xtick.labelsize'] = 24/2*x
plt.rcParams['ytick.labelsize'] = 24/2*x
plt.rcParams['legend.fontsize'] = 24/2*x

plt.rcParams['axes.titlepad'] = 22
plt.rcParams['figure.subplot.wspace'] = -0.5
plt.rcParams['figure.subplot.hspace'] = -0

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

def load_and_crop_samudra_sensitivity(path, map_dims):
    """Load a Samudra sensitivity matrix and crop it to the specified dimensions"""
    xmin, xmax, ymin, ymax = map_dims
    
    sensitivity_matrix = np.load(path)
    sensitivity_matrix = sensitivity_matrix.reshape(180, 360)
    cropped_sensitivity = sensitivity_matrix[xmin:xmax, ymin:ymax]
    
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
    """Load a MITgcm sensitivity matrix and crop it to the specified dimensions"""
    xmin, xmax, ymin, ymax = map_dims
    
    sensitivity_matrix = np.load(path)
    cropped_sensitivity = sensitivity_matrix[:,xmin:xmax, ymin:ymax]
    
    return cropped_sensitivity

def load_and_crop_correlation_data(path, map_dims):
    """Load correlation data and crop it to the specified dimensions"""
    xmin, xmax, ymin, ymax = map_dims
    
    correlation_matrix = np.load(path)
    cropped_correlation = correlation_matrix[:,xmin:xmax, ymin:ymax]
    
    # Apply wetmask for correlations too
    drymask_path = 'drymask.npy'
    if Path(drymask_path).exists():
        drymask = np.load(drymask_path)
        wetmask = np.where(drymask == 0, 1, 0)
        cropped_wetmask = wetmask[xmin:xmax, ymin:ymax]
        
        # Apply mask to all time slices
        masked_correlation = np.ma.masked_array(
            cropped_correlation,
            mask=np.broadcast_to(~cropped_wetmask.astype(bool), cropped_correlation.shape)
        )
    else:
        masked_correlation = cropped_correlation
    
    return masked_correlation

def plot_spatial_grid(data_source, file_pattern, map_dims, output_pixel, 
                     var_in, var_out, analysis_type='sensitivity', 
                     time_values=None, time_labels=None, title_template=None,
                     ch_in=None, ch_out=None, folder='.', filename_prefix=None):
    """
    Create a 3x4 grid of spatial plots for sensitivity, correlation, or other spatial data.
    
    Parameters:
    -----------
    data_source : str
        'samudra', 'mitgcm', or 'correlation'
    file_pattern : str or callable
        Pattern for finding files, or callable that returns file path given index/time
    map_dims : list
        [xmin, xmax, ymin, ymax] defining map boundaries
    output_pixel : tuple
        (lat, lon) coordinates of the output pixel
    var_in : str
        Input variable name
    var_out : str
        Output variable name
    analysis_type : str
        Type of analysis for labeling ('sensitivity', 'correlation', etc.)
    time_values : list
        List of time values/indices to plot
    time_labels : list, optional
        Custom labels for each subplot (e.g., ["Lag 0", "Lag 1", ...])
    title_template : str, optional
        Template for overall title with placeholders {analysis_type}, {var_out}, {var_in}, etc.
    ch_in, ch_out : int, optional
        Channel numbers (for Samudra/other data sources)
    folder : str
        Folder containing data files
    filename_prefix : str, optional
        Prefix for output filename
    """
    # Default time values if not provided
    if time_values is None:
        time_values = list(range(12))  # Default to 12 time points
    
    # Ensure we don't try to plot more than 12 values
    time_values = time_values[:12]
    
    # Generate default time labels if not provided
    if time_labels is None:
        if analysis_type == 'correlation':
            time_labels = [f'Lag {i}' for i in range(len(time_values))]
        else:  # sensitivity or other
            time_labels = [f'{i} back' for i in time_values]
    
    # Generate default title template if not provided
    if title_template is None:
        output_lat, output_lon = output_pixel
        var_in_clean = var_in.replace('(odd)','').replace('(even)','')
        var_out_clean = var_out.replace('(odd)','').replace('(even)','')
        
        if analysis_type == 'correlation':
            title_template = f"Lag Correlation of {var_out_clean} at ({output_lat},{output_lon}) with {var_in_clean} across map"
        else:
            model_name = "MITgcm" if data_source == 'mitgcm' else "Samudra"
            title_template = f"{model_name}: {analysis_type.title()} of {var_out_clean} at ({output_lat},{output_lon})\nwrt {var_in_clean} across map"
    
    # Set up the figure with 3x4 grid
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 4, figure=fig, wspace=0.3, hspace=0.4)
    
    output_lat, output_lon = output_pixel
    xmin, xmax, ymin, ymax = map_dims
    
    # Load all data
    all_data = []
    
    if data_source == 'correlation':
        # For correlation data, expect a single 3D file
        if callable(file_pattern):
            file_path = file_pattern(0)  # Get the file path
        else:
            file_path = Path(folder) / file_pattern
        
        if file_path.exists():
            masked_data = load_and_crop_correlation_data(file_path, map_dims)
            # Extract the specified time slices
            for i, time_val in enumerate(time_values):
                if time_val < masked_data.shape[0]:
                    all_data.append(masked_data[time_val, :, :])
                else:
                    print(f"Time index {time_val} out of range, skipping")
        else:
            print(f"Correlation file not found: {file_path}")
            return
            
    elif data_source == 'mitgcm':
        # For MITgcm, load the single file and extract time slices
        if callable(file_pattern):
            file_path = file_pattern(0)
        else:
            file_path = Path(folder) / file_pattern
        
        if file_path.exists():
            masked_data = load_and_crop_mitgcm_sensitivity(file_path, map_dims)
            for i, time_val in enumerate(time_values):
                time_idx = -time_val - 1 if time_val < masked_data.shape[0] else -1
                all_data.append(masked_data[time_idx, :, :])
        else:
            print(f"MITgcm file not found: {file_path}")
            return
            
    else:  # Samudra or other individual file sources
        for i, time_val in enumerate(time_values):
            if callable(file_pattern):
                file_path = file_pattern(time_val)
            else:
                # Replace placeholders in file pattern
                file_path = file_pattern.format(
                    time_val=time_val, ch_in=ch_in, ch_out=ch_out,
                    output_lat=output_lat, output_lon=output_lon
                )
                file_path = Path(folder) / file_path
            
            if file_path.exists():
                masked_data = load_and_crop_samudra_sensitivity(file_path, map_dims)
                all_data.append(masked_data)
            else:
                print(f"File not found: {file_path}")
                return
    
    if not all_data:
        print("No data loaded, cannot create plot")
        return
    
    # Calculate tick positions
    sample_data = all_data[0]
    row_indices = np.arange(xmin, xmax)
    col_indices = np.arange(ymin, ymax)
    
    x_pos = np.arange(sample_data.shape[1])
    y_pos = np.arange(sample_data.shape[0])
    
    xticks = 40 if sample_data.shape[1] > 120 else 20
    yticks = 40 if sample_data.shape[0] > 120 else 20
    
    x_tick_pos = x_pos[::xticks]
    x_tick_labs = col_indices[::xticks]
    y_tick_pos = y_pos[::yticks]
    y_tick_labs = row_indices[::yticks]
    
    # Create subplots
    for i in range(min(12, len(all_data))):
        row = i // 4
        col = i % 4
        
        ax = fig.add_subplot(gs[row, col])
        masked_data = all_data[i]
        
        # Create colormap
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad('black', 0.5)
        
        # Calculate color scale
        if isinstance(masked_data, np.ma.MaskedArray):
            abs_max = np.max(np.abs(masked_data.compressed()))
        else:
            abs_max = np.nanmax(np.abs(masked_data))
            
        vmin, vmax = -abs_max, abs_max
        
        # Plot the data
        im = ax.imshow(masked_data, cmap=cmap, aspect='equal', origin='lower', 
                      vmin=vmin, vmax=vmax)
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.update_ticks()
        
        # Add title with custom label
        if i < len(time_labels):
            ax.set_title(time_labels[i], fontsize=24)
        
        # Add output pixel circle
        circle_y_adj = output_lat - xmin
        circle_x_adj = output_lon - ymin
        
        if (0 <= circle_y_adj < masked_data.shape[0] and 
            0 <= circle_x_adj < masked_data.shape[1]):
            circle = plt.Circle((circle_x_adj, circle_y_adj), 2, 
                            color='black', fill=False, linewidth=2)
            ax.add_patch(circle)
        
        # Ticks and labels
        if row == 2:  # Bottom row
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
    fig.suptitle(title_template, fontsize=36, y=0.98)
    
    # Save the figure
    Path("Plots").mkdir(exist_ok=True)
    
    # Generate filename
    if filename_prefix is None:
        filename_prefix = f"{analysis_type}_{data_source}"
    
    var_in_clean = var_in.replace('(odd)','').replace('(even)','')
    var_out_clean = var_out.replace('(odd)','').replace('(even)','')
    view_name = f"({map_dims[0]},{map_dims[1]})x({map_dims[2]},{map_dims[3]})"
    
    filename = f'Plots/{filename_prefix}_grid_{view_name}_chin[{var_in_clean}]_chout[{var_out_clean}].png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {analysis_type} grid plot to {filename}")

# Legacy wrapper function for backward compatibility
def plot_sensitivity_monthly(t_months, t_end, t_end2, loc, map_dims, output_pixel, 
                             var_in, var_out, ch_in, ch_out, data_source='samudra', folder='.'):
    """Legacy wrapper for the original plot_sensitivity_monthly function"""
    
    if data_source == 'mitgcm':
        file_pattern = f"{var_in}.npy"
        time_labels = [f'{(t_end - month) // 6} months back' for month in sorted(t_months, reverse=True)]
    else:
        file_pattern = "chunk_sensitivity_chin[{ch_in}]_chout[{ch_out}]_t[{time_val},{t_end}].npy".replace('{t_end}', str(t_end))
        time_labels = [f'{(t_end - month) // 6} months back' for month in sorted(t_months, reverse=True)]
    
    plot_spatial_grid(
        data_source=data_source,
        file_pattern=file_pattern,
        map_dims=map_dims,
        output_pixel=output_pixel,
        var_in=var_in,
        var_out=var_out,
        analysis_type='sensitivity',
        time_values=sorted(t_months, reverse=True),
        time_labels=time_labels,
        ch_in=ch_in,
        ch_out=ch_out,
        folder=folder,
        filename_prefix=f"monthly_{data_source}_sensitivity"
    )

if __name__ == "__main__":
    
    # Import variable dictionary
    try:
        from misc import var_dict
    except ImportError:
        var_dict = {
            'hfds': 156,
            'tauuo': 154, 
            'tauvo': 155,
            'zos(even)': 76,
            'hfds_anomalies': 157
        }
        print("Warning: Could not import var_dict from misc, using defaults")
    
    # =================================================================
    # DEMO 1: Original Samudra Monthly Sensitivities
    # =================================================================
    print("="*60)
    print("DEMO 1: Samudra Monthly Sensitivities")
    print("="*60)
    
    t_end = 72
    t_months = [t_end - 6*i for i in range(1, 13)]
    x_out, y_out = 90, 180
    output_pixel = (x_out, y_out)
    
    delta = 50
    map_dims = [x_out-delta+20, x_out+delta-20, y_out-delta, y_out+delta]
    map_dims = [0, 180, 0, 360]
    
    vars_in = ['zos(even)']
    var_out = 'zos(even)'
    
    for var_in in vars_in:
        ch_in = var_dict[var_in]
        ch_out = var_dict[var_out]
        folder = 'adjoint_arrays/Equatorial_Pacific'
        
        time_labels = [f'{(t_end - month) // 6} months back' for month in sorted(t_months, reverse=True)]
        
        plot_spatial_grid(
            data_source='samudra',
            file_pattern="chunk_sensitivity_chin[{ch_in}]_chout[{ch_out}]_t[{time_val}," + str(t_end) + "].npy",
            map_dims=map_dims,
            output_pixel=output_pixel,
            var_in=var_in,
            var_out=var_out,
            analysis_type='Monthly Sensitivity',
            time_values=sorted(t_months, reverse=True),
            time_labels=time_labels,
            ch_in=ch_in,
            ch_out=ch_out,
            folder=folder,
            filename_prefix="monthly_samudra_sensitivity"
        )
    
    # =================================================================
    # DEMO 2: MITgcm Monthly Sensitivities
    # =================================================================
    print("\n" + "="*60)
    print("DEMO 2: MITgcm Monthly Sensitivities")
    print("="*60)
    
    t_end = 72
    t_months = [t_end - 6*i for i in range(1, 13)]
    
    loc = '(126,324)'
    x_out, y_out = eval(loc)
    output_pixel = (x_out, y_out)
    
    delta = 10
    map_dims = [x_out-delta, x_out+delta, y_out-delta, y_out+delta]
    
    vars_in = ['hfds']
    var_out = 'zos(even)'
    
    for var_in in vars_in:
        ch_in = var_dict[var_in]
        ch_out = var_dict[var_out]
        folder = f'Converted_NETcdf/{loc}'
        
        time_labels = [f'{(t_end - month) // 6} months back' for month in sorted(t_months, reverse=True)]
        
        plot_spatial_grid(
            data_source='mitgcm',
            file_pattern=f"{var_in}.npy",
            map_dims=map_dims,
            output_pixel=output_pixel,
            var_in=var_in,
            var_out=var_out,
            analysis_type='Monthly Sensitivity',
            time_values=list(range(12)),  # MITgcm uses simple indices
            time_labels=time_labels,
            ch_in=ch_in,
            ch_out=ch_out,
            folder=folder,
            filename_prefix="monthly_mitgcm_sensitivity"
        )
    
    # =================================================================
    # DEMO 3: Correlation Data (NEW)
    # =================================================================
    print("\n" + "="*60)
    print("DEMO 3: Lag Correlation Analysis")
    print("="*60)
    
    # Parameters for correlation analysis
    reference_point = (90, 180)  # Equatorial Pacific
    delta = 40
    x_ref, y_ref = reference_point
    map_dims = [x_ref-delta, x_ref+delta+1, y_ref-delta, y_ref+delta+1]
    map_dims = [50, 130, 110, 250]  # Or use a specific region
    
    var_out = 'zos'
    correlation_vars = ['zos', 'tauuo', 'tauvo', 'hfds', 'hfds_anomalies']
    
    for var_in in correlation_vars:
        correlation_file = f"CorrelationMaps/correlation_zos_at_90_180_with_{var_in}.npy"
        
        # Check if file exists
        if not Path(correlation_file).exists():
            print(f"Correlation file not found: {correlation_file}")
            continue
        
        # Custom title for correlation
        title_template = f"Lag Correlation of {var_out} at {reference_point} with {var_in} across map"
        
        plot_spatial_grid(
            data_source='correlation',
            file_pattern=correlation_file,
            map_dims=map_dims,
            output_pixel=reference_point,
            var_in=var_in,
            var_out=var_out,
            analysis_type='correlation',
            time_values=list(range(12)),  # Lags 0-11
            time_labels=[f'Lag {i}' for i in range(12)],
            title_template=title_template,
            folder='',  # File pattern includes full path
            filename_prefix="lag_correlation"
        )
    
    print("\nAll grid plots completed!")