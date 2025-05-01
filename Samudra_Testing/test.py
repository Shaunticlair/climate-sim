def compute_cell_thickness(depth_levels):
    """
    Compute the thickness of each ocean cell based on the center depths.
    
    Parameters:
    -----------
    depth_levels : list of str
        List of depth level strings like '2_5', '10_0', etc.
    
    Returns:
    --------
    dict
        Dictionary mapping each depth level to its thickness in meters
    """
    # Convert depth level strings to float values
    depths = [float(level.replace('_', '.')) for level in depth_levels]
    depths.sort()  # Ensure depths are in ascending order
    
    # Initialize thickness dictionary
    thickness = {}
    
    # Special case for the first (shallowest) layer
    # Assuming thickness extends from surface to midpoint between first and second depth
    if len(depths) > 1:
        thickness[str(depths[0]).replace('.', '_')] = (depths[0] + (depths[1] - depths[0]) / 2)
    else:
        thickness[str(depths[0]).replace('.', '_')] = depths[0] * 2  # Arbitrary if only one depth
    
    # Calculate thickness for middle layers
    # Each cell extends from midpoint between previous depth to midpoint between next depth
    for i in range(1, len(depths) - 1):
        half_dist_prev = (depths[i] - depths[i-1]) / 2
        half_dist_next = (depths[i+1] - depths[i]) / 2
        thickness[str(depths[i]).replace('.', '_')] = half_dist_prev + half_dist_next
    
    # Special case for the last (deepest) layer
    # Assuming thickness extends from midpoint between last two depths to twice the distance to the bottom
    if len(depths) > 1:
        last_idx = len(depths) - 1
        half_dist_prev = (depths[last_idx] - depths[last_idx-1]) / 2
        # For the deepest layer, we assume it extends as far below as the half-distance to the
        # previous level (this is a common approximation)
        thickness[str(depths[last_idx]).replace('.', '_')] = half_dist_prev * 2
    
    return thickness


# Use with the depth levels from the code
DEPTH_LEVELS = ['2_5', '10_0', '22_5', '40_0', '65_0', '105_0', '165_0', '250_0', 
                '375_0', '550_0', '775_0', '1050_0', '1400_0', '1850_0', '2400_0', 
                '3100_0', '4000_0', '5000_0', '6000_0']

thickness_values = compute_cell_thickness(DEPTH_LEVELS)

# Print the results
for depth, thick in thickness_values.items():
    print(f"Depth {depth} m: thickness = {thick:.2f} m")

raise Exception

from pathlib import Path
import xarray as xr
import numpy as np

# Replace these local zarr paths if required
data_mean_file = "./data_mean_analysis.zarr"
data_std_file = "./data_std_analysis.zarr"
data_file = "./data_analysis.zarr"

if not Path(data_mean_file).exists():
    data_mean = xr.open_dataset("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4_means", engine='zarr', chunks={})
    data_mean.to_zarr(data_mean_file, mode="w")
    
data_mean = xr.open_zarr(data_mean_file)

if not Path(data_std_file).exists():
    data_std = xr.open_dataset("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4_stds", engine='zarr', chunks={})
    data_std.to_zarr(data_std_file, mode="w")
data_std = xr.open_zarr(data_std_file)

if not Path(data_file).exists():
    data = xr.open_dataset("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4", engine='zarr', chunks={})
    data = data.isel(time=slice(2850-600, 2850+2)) # We only require the data in the test period
    data.to_zarr(data_file, mode="w")
data = xr.open_zarr(data_file)
data