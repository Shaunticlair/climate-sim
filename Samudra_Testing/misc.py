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

var_list = []

for t in range(2):
    label = "odd" if t == 0 else "even"
    for i in INPUT_VARS_LEV["3D_thermo_dynamic_all"]:
        var_list.append(i + f"({label})")



BOUNDARY_VARS = {
    "3D_all_hfds_anom": ["tauuo", "tauvo", "hfds", "hfds_anomalies"]
}
#print(BOUNDARY_VARS["3D_all_hfds_anom"])

var_list += BOUNDARY_VARS["3D_all_hfds_anom"]

# Turn into a dictionary
var_dict = {var: i for i, var in enumerate(var_list)}

var_arr = np.array(var_list)

#print(np.where(arr == 'zos_t=0'))
#print(np.where(arr == 'zos_t=1'))
#print(np.where(arr == 'tauuo'))
#print(np.where(arr == 'tauvo'))
#print(np.where(arr == 'hfds'))
#print(np.where(arr == 'hfds_anomalies'))

