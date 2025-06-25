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

BOUNDARY_VARS = {
    "3D_all_hfds_anom": ["tauuo", "tauvo", "hfds", "hfds_anomalies"]
}

#print(INPUT_VARS_LEV["3D_thermo_dynamic_all"])

var_list = []

for t in range(2):
    label = "even" if t == 0 else "odd"
    for i in INPUT_VARS_LEV["3D_thermo_dynamic_all"]:
        var_list.append(i + f"({label})")



var_list += BOUNDARY_VARS["3D_all_hfds_anom"]

# Turn into a dictionary
var_dict = {var: i for i, var in enumerate(var_list)}

var_arr = np.array(var_list)
#print(var_arr)

# Obtained by printing directly from inside the Test class
var_list_printed = ['uo_lev_2_5', 'uo_lev_10_0', 'uo_lev_22_5', 'uo_lev_40_0', 
            'uo_lev_65_0', 'uo_lev_105_0', 'uo_lev_165_0', 'uo_lev_250_0', 
            'uo_lev_375_0', 'uo_lev_550_0', 'uo_lev_775_0', 'uo_lev_1050_0', 
            'uo_lev_1400_0', 'uo_lev_1850_0', 'uo_lev_2400_0', 'uo_lev_3100_0', 
            'uo_lev_4000_0', 'uo_lev_5000_0', 'uo_lev_6000_0', 'vo_lev_2_5', 
            'vo_lev_10_0', 'vo_lev_22_5', 'vo_lev_40_0', 'vo_lev_65_0', 
            'vo_lev_105_0', 'vo_lev_165_0', 'vo_lev_250_0', 'vo_lev_375_0', 
            'vo_lev_550_0', 'vo_lev_775_0', 'vo_lev_1050_0', 'vo_lev_1400_0', 
            'vo_lev_1850_0', 'vo_lev_2400_0', 'vo_lev_3100_0', 'vo_lev_4000_0', 
            'vo_lev_5000_0', 'vo_lev_6000_0', 'thetao_lev_2_5', 'thetao_lev_10_0', 
            'thetao_lev_22_5', 'thetao_lev_40_0', 'thetao_lev_65_0', 'thetao_lev_105_0', 
            'thetao_lev_165_0', 'thetao_lev_250_0', 'thetao_lev_375_0', 'thetao_lev_550_0', 
            'thetao_lev_775_0', 'thetao_lev_1050_0', 'thetao_lev_1400_0', 'thetao_lev_1850_0', 
            'thetao_lev_2400_0', 'thetao_lev_3100_0', 'thetao_lev_4000_0', 'thetao_lev_5000_0', 
            'thetao_lev_6000_0', 'so_lev_2_5', 'so_lev_10_0', 'so_lev_22_5', 'so_lev_40_0', 'so_lev_65_0', 
            'so_lev_105_0', 'so_lev_165_0', 'so_lev_250_0', 'so_lev_375_0', 'so_lev_550_0', 'so_lev_775_0', 
            'so_lev_1050_0', 'so_lev_1400_0', 'so_lev_1850_0', 'so_lev_2400_0', 'so_lev_3100_0', 'so_lev_4000_0', 
            'so_lev_5000_0', 'so_lev_6000_0', 'zos']

#print(np.where(arr == 'zos_t=0'))
#print(np.where(arr == 'zos_t=1'))
#print(np.where(var_arr == 'tauuo'))
#print(np.where(var_arr == 'tauvo'))
#print(np.where(var_arr == 'hfds'))
#print(np.where(arr == 'hfds_anomalies'))
#{"tauuo":154, "tauvo":155, "hfds":156}

def get_channel_to_var(state_in_vars_config="3D_thermo_dynamic_all", 
                 boundary_vars_config="3D_all_hfds_anom", hist=1):
    """
    Get the list of variables in the Samudra vector in order. 
    Index is the channel corresponding to the variable.
    Order is 
    - Input variables (state_in_vars_config), looped hist+1 times
    - Boundary variables
    """
    channel_to_var = []
    for _ in range(hist + 1):
        channel_to_var += INPUT_VARS_LEV[state_in_vars_config]

    channel_to_var += BOUNDARY_VARS[boundary_vars_config]

    return channel_to_var
    

### Denormalization function

def denormalize_sensitivity(sensitivity_tensor, var_out, var_in, data_std):

    """
    Denormalize the sensitivity tensor based on the variable configurations.
    
    Parameters:
    -----------
    sensitivity_tensor : torch.Tensor or numpy.ndarray
        The sensitivity tensor in normalized space
    var_out : str
        The output variable name (e.g., 'zos', 'thetao_lev_2_5')
    var_in : str
        The input variable name (e.g., 'hfds', 'tauuo')
    data_std : xarray.Dataset
        Dataset containing standard deviations for variables
        
    Returns:
    --------
    torch.Tensor or numpy.ndarray
        The denormalized sensitivity tensor
    
    Notes:
    ------
    Sensitivity is defined as ∂y/∂x, where y is the output and x is the input.
    To denormalize, we need to multiply by std_y/std_x, since:
    ∂(y/std_y)/∂(x/std_x) = (∂y/∂x) * (std_x/std_y)
    """
    
    # Get standard deviations for output and input variables
    std_out = data_std[var_out].values.item()
    std_in = data_std[var_in].values.item()
    
    # Calculate the denormalization factor
    denorm_factor = std_out / std_in
    
    # Apply the denormalization
    return sensitivity_tensor * denorm_factor
    
