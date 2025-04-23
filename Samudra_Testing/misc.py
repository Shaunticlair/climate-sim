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
#print(np.where(arr == 'tauuo'))
#print(np.where(arr == 'tauvo'))
#print(np.where(arr == 'hfds'))
#print(np.where(arr == 'hfds_anomalies'))

"""
class VisibleTest(data_loaders.Test):
        def __getitem__(self, idx):

            print("Input axes:", list(self.inputs.dims))
            print("Input vars:", list(self.inputs.data_vars))
            print("Inputs no extra", list(self.inputs_no_extra.data_vars))
                
            if type(idx) == slice:
                if idx.start == None and idx.stop == None:
                    idx = slice(0, self.size, idx.step)
                elif idx.start == None:
                    idx = slice(0, idx.stop, idx.step)
                elif idx.stop == None:
                    idx = slice(idx.start, self.size, idx.step)
            elif type(idx) == int:
                idx = slice(idx, idx + 1, 1)

            rolling_idx = self.rolling_indices.isel(window_dim=idx)
            x_index = xr.Variable(
                ["window_dim", "time"], rolling_idx
            )
            print("Out: ", (self.ind_start + x_index.isel(time=slice(self.hist + 1, None))).values, end=' ')
            data_in = self.inputs_no_extra.isel(time=x_index).isel(
                time=slice(None, self.hist + 1)
            )
            data_in = (
                (data_in - self.inputs_no_extra_mean) / self.inputs_no_extra_std
            ).fillna(0)
            print("data_in:", list(data_in.data_vars))
            print("data_in axes:", list(data_in.dims))
            shaunticlair_temp_array = data_in.to_array().transpose("window_dim", "time", "variable", "y", "x")
            print("data_in to_array:", list(shaunticlair_temp_array.coords["variable"].values))
            print("data_in shape:", shaunticlair_temp_array.shape)

            shaunticlair_transposed_array = rearrange(
                shaunticlair_temp_array, "window_dim time variable y x -> window_dim (time variable) y x"
            )
            print(shaunticlair_transposed_array)
            print("data_in time variable", shaunticlair_transposed_array.coords['time variable'])
            raise Exception
            data_in = (
                data_in.to_array()
                .transpose("window_dim", "time", "variable", "y", "x")
                .to_numpy()
            )
            data_in = rearrange(
                data_in, "window_dim time variable y x -> window_dim (time variable) y x"
            )
            print()
            if len(self.extras.variables) != 0:
                data_in_boundary = self.extras.isel(time=x_index).isel(time=self.hist)
                data_in_boundary = (
                    (data_in_boundary - self.extras_mean) / self.extras_std
                ).fillna(0)
                data_in_boundary = (
                    data_in_boundary.to_array()
                    .transpose("window_dim", "variable", "y", "x")
                    .to_numpy()
                )
                data_in = np.concatenate((data_in, data_in_boundary), axis=1)

            label = self.outputs.isel(time=x_index).isel(time=slice(self.hist + 1, None))
            label = ((label - self.out_mean) / self.out_std).fillna(0)
            label = (
                label.to_array()
                .transpose("window_dim", "time", "variable", "y", "x")
                .to_numpy()
            )
            label = rearrange(
                label, "window_dim time variable y x -> window_dim (time variable) y x"
            )

            items = (torch.from_numpy(data_in).float(), torch.from_numpy(label).float())

            return items
"""
