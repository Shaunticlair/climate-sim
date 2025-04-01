#!/usr/bin/env python
# Compare first timestep between data.zarr and prediction zarr files

import xarray as xr
import numpy as np

# Open datasets
data_file = "./data.zarr"
thermo_file = "3D_thermo_all_prediction.zarr"

print("Opening datasets...")
data_ds = xr.open_zarr(data_file)
thermo_ds = xr.open_zarr(thermo_file)

# Check dimensions
print(f"\nData time dimension: {data_ds.dims['time']}")
print(f"Thermo time dimension: {thermo_ds.dims['time']}")

# Extract the structure of the datasets
print("\nData variable structure:")
for var in sorted(data_ds.variables):
    if var not in ['x', 'y', 'lev', 'time', 'x_b', 'y_b', 'lat', 'lon', 'lat_b', 'lon_b']:
        print(f"  {var}: {data_ds[var].dims}")

print("\nThermo variable structure:")
for var in sorted(thermo_ds.variables):
    if var not in ['x', 'y', 'lev', 'time', 'x_b', 'y_b', 'lat', 'lon', 'lat_b', 'lon_b']:
        print(f"  {var}: {thermo_ds[var].dims}")

# Compare timesteps
print("\nChecking time coordinates...")
print(f"First time in data: {data_ds.time.values[0]}")
print(f"Last time in data: {data_ds.time.values[-1]}")
print(f"First time in thermo: {thermo_ds.time.values[0]}")
print(f"Last time in thermo: {thermo_ds.time.values[-1]}")

# In the original script, the issue happens in this line:
# section_mask = np.isnan(da_temp).all('x').isel(time=0)
# where da_temp is the temperature at all depths

# Check structure of temperature variable in original data
print("\nStructure of temperature variables in data:")
temp_vars = [var for var in data_ds.variables if var.startswith('thetao_lev_')]
print(f"Temperature variables: {temp_vars}")
if temp_vars:
    print(f"Example temp var shape: {data_ds[temp_vars[0]].shape}")

# Check structure of converted temperature variable in thermo
print("\nStructure of temperature variable in thermo:")
print(f"thermo['thetao'] shape: {thermo_ds['thetao'].shape}")

# Look at the time indices - are they the same or offset?
print("\nComparing a few values to check alignment...")

# First, we need to find equivalent variables
# In data, we have individual level variables, in thermo we have a consolidated variable
print("\nComparing temp at level 2.5m for first timestep:")
try:
    data_temp_2_5 = data_ds['thetao_lev_2_5'].isel(time=0)
    # In thermo, first level should be 2.5m
    thermo_temp_first_level = thermo_ds['thetao'].isel(time=0, lev=0)
    
    # Compare a few sample points
    sample_points = [(90, 180), (45, 90), (135, 270)]
    for y, x in sample_points:
        data_val = data_temp_2_5.isel(y=y, x=x).values
        thermo_val = thermo_temp_first_level.isel(y=y, x=x).values
        print(f"  Point (y={y}, x={x}): data={data_val}, thermo={thermo_val}, diff={data_val-thermo_val}")
except Exception as e:
    print(f"Error comparing values: {str(e)}")

# Check if there's a one-step offset
print("\nTrying to compare with a one-step offset (data[1] vs thermo[0]):")
try:
    data_temp_2_5_offset = data_ds['thetao_lev_2_5'].isel(time=1)
    thermo_temp_first_level = thermo_ds['thetao'].isel(time=0, lev=0)
    
    for y, x in sample_points:
        data_val = data_temp_2_5_offset.isel(y=y, x=x).values
        thermo_val = thermo_temp_first_level.isel(y=y, x=x).values
        print(f"  Point (y={y}, x={x}): data[1]={data_val}, thermo[0]={thermo_val}, diff={data_val-thermo_val}")
except Exception as e:
    print(f"Error comparing offset values: {str(e)}")
