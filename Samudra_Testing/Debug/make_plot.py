#!/usr/bin/env python
import xarray as xr
import matplotlib.pyplot as plt
import cmocean as cm

# Load the prediction data
ds_prediction = xr.open_zarr("3D_thermo_dynamic_all_prediction.zarr")

# Get the last timestep index
last_timestep = ds_prediction.dims['time'] - 1

# Plot the last timestep's surface temperature
surface_temp = ds_prediction.thetao.isel(time=last_timestep, lev=0)
surface_temp.rename(r"Surface temperature [$\degree C$]").plot(vmin=0, vmax=30, cmap=cm.cm.balance)

plt.title(f"Surface Temperature (Last Timestep)")
plt.savefig("surface_temp_last_timestep.png", dpi=300, bbox_inches='tight')