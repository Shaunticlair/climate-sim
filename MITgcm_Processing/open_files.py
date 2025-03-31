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
