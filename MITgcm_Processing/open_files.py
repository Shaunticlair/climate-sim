from MITgcmutils import mds

import numpy as np
import os

workingdir = '/nobackup/sruiz5/MITGCMTEST/ECCOV4/release4/run'
os.chdir(workingdir)

# Define iteration number(s) to read
iter_number = 00001  # Example iteration number

### State variables

uo_data = mds.rdmds('U', itrs=iter_number) # Read zonal velocity (uo)
vo_data = mds.rdmds('V', itrs=iter_number) # Meridional velocity (vo)
thetao_data = mds.rdmds('T', itrs=iter_number) # Potential temperature
so_data = mds.rdmds('S', itrs=iter_number)  # Salinity
zos_data = mds.rdmds('Eta', itrs=iter_number)  # Sea surface height

### Boundary conditions

bottom_pressure = mds.rdmds('diags/OBP_mon_mean/OBP_mon_mean') # Ocean bottom pressure
tau_x = mds.rdmds('diags/EXFtaux_mon_mean/EXFtaux_mon_mean') # x-axis wind stress
tau_y = mds.rdmds('diags/EXFtauy_mon_mean/EXFtauy_mon_mean') # y-axis wind stress
heat_flux = mds.rdmds('diags/oceQnet_mon_mean/oceQnet_mon_mean') #Diabatic heat flux
freshwater_flux = mds.rdmds('diags/oceFWflx_mon_mean/oceFWflx_mon_mean') #Surface mass flux

# If you need to read multiple iterations:
# iter_numbers = [10000, 20000, 30000]
# uo_data_multiple = mds.rdmds('U', itrs=iter_numbers)
# vo_data_multiple = mds.rdmds('V', itrs=iter_numbers)


# Display dimensions of state variables
print("\n--- State Variables Dimensions ---")
print(f"uo_data (zonal velocity): {uo_data.shape}")
print(f"vo_data (meridional velocity): {vo_data.shape}")
print(f"thetao_data (potential temperature): {thetao_data.shape}")
print(f"so_data (salinity): {so_data.shape}")
print(f"zos_data (sea surface height): {zos_data.shape}")

# Display dimensions of boundary conditions
print("\n--- Boundary Conditions Dimensions ---")
print(f"bottom_pressure: {bottom_pressure.shape}")
print(f"tau_x (x-axis wind stress): {tau_x.shape}")
print(f"tau_y (y-axis wind stress): {tau_y.shape}")
print(f"heat_flux (diabatic heat flux): {heat_flux.shape}")
print(f"freshwater_flux (surface mass flux): {freshwater_flux.shape}")

# Display data types
print("\n--- Data Types ---")
print(f"uo_data type: {uo_data.dtype}")
print(f"thetao_data type: {thetao_data.dtype}")

# Display some basic statistics for a variable (e.g., temperature)
print("\n--- Basic Statistics for Temperature ---")
print(f"Min temperature: {np.nanmin(thetao_data)}")
print(f"Max temperature: {np.nanmax(thetao_data)}")
print(f"Mean temperature: {np.nanmean(thetao_data)}")

# Display statistics for sea surface height
print("\n--- Basic Statistics for Sea Surface Height ---")
print(f"Min SSH: {np.nanmin(zos_data)}")
print(f"Max SSH: {np.nanmax(zos_data)}")
print(f"Mean SSH: {np.nanmean(zos_data)}")

# Count NaN values
print("\n--- NaN Count ---")
print(f"NaN count in temperature: {np.isnan(thetao_data).sum()}")
print(f"NaN count in zonal velocity: {np.isnan(uo_data).sum()}")