import numpy as np

# Depth levels
DEPTH_LEVELS = ['2_5', '10_0', '22_5', '40_0', '65_0', '105_0', '165_0', 
                '250_0', '375_0', '550_0', '775_0', '1050_0', '1400_0', 
                '1850_0', '2400_0', '3100_0', '4000_0', '5000_0', '6000_0']


VARS = {
    "3D_thermo_dynamic_all": ['uo', 'vo', 'thetao', 'so', 'zos'],
    "3D_thermo_all": ['thetao', 'so', 'zos'],
}
# Define input and output variables
INPUT_VARS_LEV = {
    "3D_thermo_dynamic_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ] + ["zos"],
    "3D_thermo_all": [
        k + str(j)
        for k in ["thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ] + ["zos"]
}

#print(INPUT_VARS_LEV["3D_thermo_dynamic_all"])

arr = []

for t in range(2):
    for i in INPUT_VARS_LEV["3D_thermo_dynamic_all"]:
        arr.append(i + f"_t={t}")



BOUNDARY_VARS = {
    "3D_all_hfds_anom": ["tauuo", "tauvo", "hfds", "hfds_anomalies"]
}
#print(BOUNDARY_VARS["3D_all_hfds_anom"])

arr += BOUNDARY_VARS["3D_all_hfds_anom"]

print(arr)

arr = np.array(arr)

print(np.where(arr == 'zos_t=0'))
print(np.where(arr == 'zos_t=1'))
print(np.where(arr == 'tauuo'))
print(np.where(arr == 'tauvo'))
print(np.where(arr == 'hfds'))
print(np.where(arr == 'hfds_anomalies'))

