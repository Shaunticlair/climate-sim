import setup

# Model choice
hist = 1
N_test = 40  # Timesteps to use for testing 
state_in_vars_config = "3D_thermo_dynamic_all"
state_out_vars_config = state_in_vars_config
boundary_vars_config = "3D_all_hfds_anom"

device = setup.torch_config_cuda_cpu_seed() 

# Based on our model, we get our list of variables
list_list_str, list_num_channels = setup.choose_model(state_in_vars_config, boundary_vars_config, hist)

# Unpack
input_list_str, boundary_list_str, output_list_str, vars_list_str = list_list_str
num_input_channels, num_output_channels = list_num_channels

data, data_mean, data_std = setup.load_data_raw(0,2)

loss_fn = setup.gen_weighted_loss_fn(data, state_in_vars_config=state_out_vars_config)
