import torch
import numpy as np

def simple_sensitivity(model, data_input, source_coords, target_coords, perturbation_size=1e-4):
    """
    Computes sensitivity of target cell to a perturbation in source cell.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to evaluate sensitivity for
    data_input : torch.Tensor
        Input tensor to the model (b, c, h, w)
    source_coords : tuple
        (channel, y, x) coordinates of source cell to perturb
    target_coords : tuple
        (channel, y, x) coordinates of target cell to measure
    perturbation_size : float
        Size of perturbation to apply
        
    Returns:
    --------
    sensitivity : float
        Sensitivity value between source and target cells
    """
    model.eval()
    
    # Make sure we don't modify the original input
    data_input = data_input.clone().detach()
    
    # Unpack coordinates
    s_c, s_y, s_x = source_coords
    t_c, t_y, t_x = target_coords
    
    # Step 1: Run model on unperturbed input
    with torch.no_grad():
        baseline_output = model.forward_once(data_input)
        baseline_value = baseline_output[0, t_c, t_y, t_x].item()
    
    # Step 2: Perturb the input at source coordinates
    perturbed_input = data_input.clone()
    perturbed_input[0, s_c, s_y, s_x] += perturbation_size
    
    # Step 3: Run model on perturbed input
    with torch.no_grad():
        perturbed_output = model.forward_once(perturbed_input)
        perturbed_value = perturbed_output[0, t_c, t_y, t_x].item()
    
    # Step 4: Calculate sensitivity
    sensitivity = (perturbed_value - baseline_value) / perturbation_size
    
    return sensitivity

# Example usage:
if __name__ == "__main__":
    # Load model and data 
    from model import Samudra
    from data_loaders import Test
    from utils import extract_wet
    import xarray as xr
    
    # Quick setup for example
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = xr.open_zarr("./data.zarr")
    data_mean = xr.open_zarr("./data_mean.zarr")
    data_std = xr.open_zarr("./data_std.zarr")
    
    # Configure model (simplified from samudra_rollout.py)
    hist = 1
    exp_num_in = "3D_thermo_all"  # Could also be "3D_thermo_dynamic_all"
    exp_num_extra = "3D_all_hfds_anom"
    
    # Define input variables based on experiment type
    DEPTH_LEVELS = ['2_5', '10_0', '22_5', '40_0', '65_0', '105_0', '165_0', '250_0', '375_0', 
                   '550_0', '775_0', '1050_0', '1400_0', '1850_0', '2400_0', '3100_0', 
                   '4000_0', '5000_0', '6000_0']
    
    INPT_VARS = {
        "3D_thermo_dynamic_all": [f"{k}{j}" for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"] 
                                 for j in DEPTH_LEVELS] + ["zos"],
        "3D_thermo_all": [f"{k}{j}" for k in ["thetao_lev_", "so_lev_"] 
                         for j in DEPTH_LEVELS] + ["zos"]
    }
    EXTRA_VARS = {"3D_all_hfds_anom": ["tauuo", "tauvo", "hfds", "hfds_anomalies"]}
    OUT_VARS = INPT_VARS.copy()
    
    inputs_str = INPT_VARS[exp_num_in]
    extra_in_str = EXTRA_VARS[exp_num_extra]
    outputs_str = OUT_VARS[exp_num_in]
    
    N_in = len(inputs_str)
    N_extra = len(extra_in_str)
    N_out = len(outputs_str)
    
    num_in = int((hist + 1) * N_in + N_extra)
    num_out = int((hist + 1) * len(outputs_str))
    
    # Create model
    wet_zarr = data.wetmask
    wet = extract_wet(wet_zarr, outputs_str, hist)
    
    # Load model
    model = Samudra(n_out=num_out, ch_width=[num_in]+[200,250,300,400], wet=wet.to(device), hist=hist)
    model.load_state_dict(torch.load("samudra_thermo_dynamic_seed1.pt", map_location=device)["model"])
    model = model.to(device)
    
    # Create test data
    test_data = Test(
        data,
        inputs_str,
        extra_in_str,
        outputs_str,
        wet,
        data_mean,
        data_std,
        10,  # N_test
        hist,
        0,
        long_rollout=False,
        device=device,
    )
    
    # Get input data
    input_data = test_data[0][0].to(device)
    
    # Test sensitivity
    source_coords = (0, 90, 180)  # Surface temperature at center of grid
    target_coords = (0, 90, 180)  # Surface temperature at nearby point
    
    sensitivity = simple_sensitivity(model, input_data, source_coords, target_coords)
    print(f"Sensitivity from {source_coords} to {target_coords}: {sensitivity}")