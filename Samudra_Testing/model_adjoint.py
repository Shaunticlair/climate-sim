import torch
import torch.nn as nn
import numpy as np
from itertools import tee

import torch.utils.checkpoint as checkpoint

def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class CappedGELU(torch.nn.Module):
    """
    Implements a GeLU with capped maximum value.
    """

    def __init__(self, cap_value=1.0, **kwargs):
        """
        :param cap_value: float: value at which to clip activation
        :param kwargs: passed to torch.nn.LeadyReLU
        """
        super().__init__()
        self.add_module("gelu", torch.nn.GELU(**kwargs))
        self.register_buffer("cap", torch.tensor(cap_value, dtype=torch.float32))

    def forward(self, inputs):
        x = self.gelu(inputs)
        x = torch.clamp(x, max=self.cap)
        return x

class BilinearUpsample(torch.nn.Module):
    def __init__(self, upsampling: int = 2, **kwargs):
        super().__init__()
        self.upsampler = torch.nn.Upsample(scale_factor=upsampling, mode="bilinear")

    def forward(self, x):
        return self.upsampler(x)

class AvgPool(torch.nn.Module):
    def __init__(
        self,
        pooling: int = 2,
    ):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(pooling)

    def forward(self, x):
        return self.avgpool(x)

class ConvNeXtBlock(torch.nn.Module):
    """
    A convolution block as reported in https://github.com/CognitiveModeling/dlwp-hpx/blob/main/src/dlwp-hpx/dlwp/model/modules/blocks.py.

    This is a modified version of the actual ConvNextblock which is used in the HealPix paper.

    """

    def __init__(
        self,
        in_channels: int = 300,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,
        activation: torch.nn.Module = CappedGELU,
        pad="circular",
        upscale_factor: int = 4
    ):
        super().__init__()
        assert kernel_size % 2 != 0, "Cannot use even kernel sizes!"

        self.N_in = in_channels
        self.N_pad = int((kernel_size + (kernel_size - 1) * (dilation - 1) - 1) / 2)
        self.pad = pad

        assert n_layers == 1, "Can only use a single layer here!"

        # 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
            )

        # Convolution block
        convblock = []
        convblock.append(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=int(in_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
            )
        )
        convblock.append(torch.nn.BatchNorm2d(in_channels * upscale_factor))
        
        convblock.append(activation())
            
        convblock.append(
            torch.nn.Conv2d(
                in_channels=int(in_channels * upscale_factor),
                out_channels=int(in_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
            )
        )
        convblock.append(torch.nn.BatchNorm2d(in_channels * upscale_factor))

        convblock.append(activation())
            
        # Linear postprocessing
        convblock.append(
            torch.nn.Conv2d(
                in_channels=int(in_channels * upscale_factor),
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
            )
        )
        self.convblock = torch.nn.Sequential(*convblock)

    def forward(self, x):
        skip = self.skip_module(x)
        for l in self.convblock:
            if isinstance(l, nn.Conv2d) and l.kernel_size[0] != 1:
                x = torch.nn.functional.pad(
                    x, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                x = torch.nn.functional.pad(
                    x, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
            x = l(x)
        return skip + x

class BaseSamudra(torch.nn.Module):
    def __init__(
        self, ch_width, n_out, wet, hist, pred_residuals, last_kernel_size, pad
    ):
        """
        Base class for the Samudra model.
        
        Parameters:
        -----------
        ch_width : list
            List of channel widths for the network architecture, defining the feature dimensions 
            throughout the network. The first element corresponds to the input channels.
        n_out : int
            Number of output channels/variables for the final prediction.
        wet : torch.Tensor
            Wet mask tensor indicating ocean cells (vs. land), used for masking predictions.
        hist : int
            History parameter controlling how many previous steps are used for prediction.
        pred_residuals : bool
            If True, predict residuals (changes) rather than absolute values.
        last_kernel_size : int
            Kernel size for the final convolutional layer. Must be an odd number.
        pad : str
            Padding mode for convolutional operations, e.g., 'circular' for periodic boundary conditions.
        """
        super().__init__()
        assert last_kernel_size % 2 != 0, "Cannot use even kernel sizes!"
        self.N_in = ch_width[0]
        self.N_out = ch_width[-1]
        self.ch_width = ch_width
        self.wet = wet
        self.N_pad = int((last_kernel_size - 1) / 2)
        self.pad = pad
        self.pred_residuals = pred_residuals
        self.hist = hist
        self.input_channels = ch_width[0]
        self.output_channels = n_out

    def forward_once(self, fts):
        raise NotImplementedError()

    def forward(
        self,
        inputs,
        output_only_last=False,
        loss_fn=None,
    ) -> torch.Tensor:
        outputs = []
        loss = None
        N, C, H, W = inputs[0].shape

        for step in range(len(inputs) // 2):
            if step == 0:
                input_tensor = inputs[
                    0
                ]  # For HIST=1, [0->[0in, 1in], 1->[2out, 3out], 2->[2in, 3in], 3->[4out, 5out]
            else:
                inputs_0 = outputs[-1]  # Last output corresponds to input at current time step
                input_tensor = torch.cat(
                    [
                        inputs_0,
                        inputs[2 * step][
                            :, self.output_channels :
                        ],  # boundary conditions
                    ],
                    dim=1,
                )

            assert (
                input_tensor.shape[1] == self.input_channels
            ), f"Input shape is {input_tensor.shape[1]} but should be {self.input_channels}"
            decodings = self.forward_once(input_tensor)
            if self.pred_residuals:
                reshaped = (
                    input_tensor[
                        :,
                        : self.output_channels,
                    ]  # Residuals on last state in input
                    + decodings
                )  # Residual prediction
            else:
                reshaped = decodings  # Absolute prediction

            if loss_fn is not None:
                assert (
                    reshaped.shape == inputs[2 * step + 1].shape
                ), f"Output shape is {reshaped.shape} but should be {inputs[2 * step + 1].shape}"
                if loss is None:
                    loss = loss_fn(
                        reshaped,
                        inputs[2 * step + 1],
                    )
                else:
                    loss += loss_fn(
                        reshaped,
                        inputs[2 * step + 1],
                    )

            outputs.append(reshaped)

        if loss_fn is None:
            if output_only_last:
                res = outputs[-1]
            else:
                res = outputs
            return res

        else:
            return loss

    def inference(
        self, inputs, initial_input=None, num_steps=None, output_only_last=False, device="cuda"
    ) -> torch.Tensor:
        outputs = []
        for step in range(num_steps):
            if step == 0:
                input_tensor = inputs[0][0].to(
                        device=device
                    )  # inputs[0][0] is the input at step 0. For HIST=1 ; 0->[[0, 1], [2, 3]]; 1->[[2, 3], [4, 5]]; 2->[[4, 5], [6, 7]]; 3->[[6, 7], [8, 9]]
                
                if initial_input is not None:
                    input_tensor[:, :self.output_channels] = initial_input
            else:
                inputs_0 = outputs[-1].unsqueeze(
                    0
                )  # Last output corresponds to input at current time step
                input_tensor = torch.cat(
                    [
                        inputs_0,
                        inputs[step][0][
                            :, self.output_channels :
                        ].to(  # boundary conditions
                            device=device
                        ),
                    ],
                    dim=1,
                )

            assert (
                input_tensor.shape[1] == self.input_channels
            ), f"Input shape is {input_tensor.shape[1]} but should be {self.input_channels}"
            decodings = self.forward_once(input_tensor)
            if self.pred_residuals:
                reshaped = input_tensor[
                    0,
                    : self.output_channels,
                ].to(  # Residuals on last state in input
                    device=device
                ) + decodings.squeeze(
                    0
                )
            else:
                reshaped = decodings.squeeze(0)

            outputs.append(reshaped)

        if output_only_last:
            res = outputs[-1]
        else:
            res = outputs

        return res
    

    


    


class Samudra(BaseSamudra):
    def __init__(
        self,
        wet,
        hist,
        core_block=ConvNeXtBlock,
        down_sampling_block=AvgPool,
        up_sampling_block=BilinearUpsample,
        activation=CappedGELU,
        ch_width=[157,200,250,300,400],
        n_out=77,
        dilation=[1, 2, 4, 8],
        n_layers=[1, 1, 1, 1],
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
    ):
    
        super().__init__(
            ch_width, n_out, wet, hist, pred_residuals, last_kernel_size, pad
        )

        layers = []
        for i, (a, b) in enumerate(pairwise(ch_width)):
            layers.append(
                core_block(
                    a,
                    b,
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    activation=activation,
                    pad=pad,
                )
            )
            layers.append(down_sampling_block())
        layers.append(
            core_block(
                b,
                b,
                dilation=dilation[i],
                n_layers=n_layers[i],
                activation=activation,
                pad=pad,
            )
        )
        layers.append(up_sampling_block(in_channels=b, out_channels=b))
        ch_width.reverse()
        dilation.reverse()
        n_layers.reverse()
        for i, (a, b) in enumerate(pairwise(ch_width[:-1])):
            layers.append(
                core_block(
                    a,
                    b,
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    activation=activation,
                    pad=pad,
                )
            )
            layers.append(up_sampling_block(in_channels=b, out_channels=b))
        layers.append(
            core_block(
                b,
                b,
                dilation=dilation[i],
                n_layers=n_layers[i],
                activation=activation,
                pad=pad,
            )
        )
        layers.append(torch.nn.Conv2d(b, n_out, last_kernel_size))

        self.layers = nn.ModuleList(layers)
        self.num_steps = int(len(ch_width) - 1)

    def forward_once(self, fts):
        temp = []
        for i in range(self.num_steps):
            temp.append(None)
        count = 0
        for l in self.layers:
            crop = fts.shape[2:]
            if isinstance(l, nn.Conv2d):
                fts = torch.nn.functional.pad(
                    fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                fts = torch.nn.functional.pad(
                    fts, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
            fts = l(fts)
            if count < self.num_steps:
                if isinstance(l, ConvNeXtBlock):
                    temp[count] = fts
                    count += 1
            elif count >= self.num_steps:
                if isinstance(l, BilinearUpsample):
                    crop = np.array(fts.shape[2:])
                    shape = np.array(
                        temp[int(2 * self.num_steps - count - 1)].shape[2:]
                    )
                    pads = shape - crop
                    pads = [
                        pads[1] // 2,
                        pads[1] - pads[1] // 2,
                        pads[0] // 2,
                        pads[0] - pads[0] // 2,
                    ]
                    fts = nn.functional.pad(fts, pads)
                    fts += temp[int(2 * self.num_steps - count - 1)]
                    count += 1
        return torch.mul(fts, self.wet)


class SamudraAdjoint(Samudra):
    def __init__(
        self,
        wet,
        hist,
        core_block=ConvNeXtBlock,
        down_sampling_block=AvgPool,
        up_sampling_block=BilinearUpsample,
        activation=CappedGELU,
        ch_width=[157,200,250,300,400],
        n_out=77,
        dilation=[1, 2, 4, 8],
        n_layers=[1, 1, 1, 1],
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
    ):
        # Call the parent constructor
        super().__init__(
            wet,
            hist,
            core_block,
            down_sampling_block,
            up_sampling_block,
            activation,
            ch_width,
            n_out,
            dilation,
            n_layers,
            pred_residuals,
            last_kernel_size,
            pad
        )

    def process_indices(self, indices, dims, device="cuda"):
        processed = []
        for i, idx in enumerate(indices):
            if idx is None:
                processed.append(torch.arange(dims[i], device=device))
            else:
                processed.append(torch.tensor(idx, device=device))
        return processed
    
    def checkpointed_forward_once(self, x):
        """
        Wrapper for forward_once that uses gradient checkpointing for memory efficiency
        """
        # Define a custom forward function that can be checkpointed
        def custom_forward(*inputs):
            return self.forward_once(inputs[0])
        
        # Apply checkpointing
        return checkpoint.checkpoint(custom_forward, x, use_reentrant=False)

    def state_sensitivity_computation(self, inputs, 
                                  initial_indices,
                                  final_indices,
                                  initial_time=0, 
                                  final_time=-1,
                                  device="cuda",
                                  use_checkpointing=True):
        """
        Computes the sensitivity of the model state at time `final_time` with respect to 
        the model state at time `initial_time` using a full autoregressive rollout.
        Filters the sensitivity to only the specified indices in the state tensors.

        If we select which indices within that timestep to compute the sensitivity for, we get:
            d(state[final_time][final_indices])/d(state[initial_time][initial_indices])

        Params:
        -------
        inputs: Test object (torch.utils.data.Dataset) contains state evolution data.
                Each state is a 4D tensor of shape (1, num_channels, height, width).
                inputs[i][0] has num_channels = self.input_channels
                inputs[i][1] has num_channels = self.output_channels

        initial_indices: list of numpy arrays
            The indices of the state at initial_time we are taking the sensitivity with respect to.
            Format: [channel_indices, height_indices, width_indices]
            If one of these indices is None, it will be set to all indices in that dimension.

        final_indices: list of numpy arrays
            The indices of the state at final_time we are computing the sensitivity for.
            Format: [channel_indices, height_indices, width_indices]
            If one of these indices is None, it will be set to all indices in that dimension.

        initial_time: int
            The time step we are taking the sensitivity with respect to. 
        final_time: int
            The time step we are computing the sensitivity for.

        device: str
            The device to run the computation on. Default is "cuda".
            
        use_checkpointing: bool
            Whether to use gradient checkpointing to save memory. Default is True.

        Returns:
        --------
        sensitivity: torch.Tensor
            The sensitivity of the final state at `final_time` with respect to the initial state at `initial_time`.
            This will be a tensor with the shape
            ( 
                len(final_indices[0]),   len(final_indices[1]),   len(final_indices[2]),
                len(initial_indices[0]), len(initial_indices[1]), len(initial_indices[2]) 
            )
        """
        ### SETUP PHASE
        assert self.hist == 1, "Sensitivity computation currently only supports hist=1"

        # Convert negative indices to positive ones using Python's standard behavior
        if final_time < 0:
            final_time = len(inputs) + final_time
        
        # Get the initial input
        initial_input = inputs[initial_time][0].to(device)
        
        # Extract tensor dimensions
        batch_size, n_channels, height, width = initial_input.shape
        assert batch_size == 1, "Sensitivity computation currently only supports batch size of 1"
        
        # Process initial indices
        initial_c_idx, initial_h_idx, initial_w_idx = self.process_indices(
            initial_indices, [n_channels, height, width], device=device
        )

        # Only get gradient for the desired indices
        initial_input = initial_input.detach().clone()
        initial_input.requires_grad_(False)

        # Selectively enable gradients only for indices we're interested in
        for init_c in initial_c_idx:
            for init_h in initial_h_idx:
                for init_w in initial_w_idx:
                    initial_input[0, init_c, init_h, init_w].requires_grad_(True)

        ### FORWARD PASS PHASE - Full Autoregressive Rollout
        # Set the model to evaluation mode
        self.eval()
        
        # Initialize outputs list to store all states
        outputs = []
        
        # Choose the forward function based on checkpointing flag
        forward_func = self.checkpointed_forward_once if use_checkpointing else self.forward_once
        
        # Compute the forward pass from initial_time to final_time with gradients
        with torch.enable_grad():
            current_input = initial_input
            
            # Perform the autoregressive rollout
            for t in range(initial_time, final_time + 1):
                if t == initial_time:
                    # First step uses the initial input which already requires gradients
                    output = forward_func(current_input)
                else:
                    current_state = outputs[-1].unsqueeze(0)  # Unsqueeze to maintain the batch dimension
                    # For boundary conditions, we need to get the appropriate input from the test dataset
                    boundary_conditions = inputs[t][0][:, self.output_channels:].to(device)
                    
                    # Construct the new input by combining the previous output with boundary conditions
                    current_input = torch.cat([
                        current_state, 
                        boundary_conditions 
                    ], dim=1)
                    
                    # Forward pass for this step 
                    output = forward_func(current_input.to(device))
                
                # Store the output
                outputs.append(output.squeeze(0))  # Remove batch dimension
                
            # The final output is the last element in outputs
            final_output = outputs[-1]

            print("Output shape at final time step: ", final_output.shape)
            
            # Process final indices based on the final output shape
            _, final_height, final_width = final_output.shape
            final_c_idx, final_h_idx, final_w_idx = self.process_indices(
                final_indices, [self.output_channels, final_height, final_width], device=device
            )
            
            # Initialize sensitivity tensor
            sensitivity_shape = (
                len(final_c_idx), len(final_h_idx), len(final_w_idx),
                len(initial_c_idx), len(initial_h_idx), len(initial_w_idx)
            )
            sensitivity = torch.zeros(sensitivity_shape, device=device)
            
            # Compute gradients for each target index in the final output
            for i, c in enumerate(final_c_idx):
                for j, h in enumerate(final_h_idx):
                    for k, w in enumerate(final_w_idx):
                        # Clear previous gradients
                        if initial_input.grad is not None:
                            initial_input.grad.zero_()
                        
                        # Create a mask tensor to select only the current index (ensuring it's on the right device)
                        mask = torch.zeros_like(final_output, device=device)
                        mask[c, h, w] = 1.0
                        
                        # Backward pass through the entire chain
                        final_output.backward(mask, retain_graph=True)
                        
                        # Extract the gradients for the initial indices
                        for l, init_c in enumerate(initial_c_idx):
                            for m, init_h in enumerate(initial_h_idx):
                                for n, init_w in enumerate(initial_w_idx):
                                    sensitivity[i, j, k, l, m, n] = initial_input.grad[0, init_c, init_h, init_w].item()
        
        return sensitivity
    
    def state_sensitivity_computation(self, inputs, 
                              initial_indices,
                              final_indices,
                              initial_time=0, 
                              final_time=-1,
                              device="cuda",
                              use_checkpointing=True):
        """
        Computes the sensitivity with improved gradient tracking
        """
        # SETUP PHASE - similar to your original code
        assert self.hist == 1, "Sensitivity computation currently only supports hist=1"
        
        if final_time < 0:
            final_time = len(inputs) + final_time
        
        # Get dimensions from the first input
        _, n_channels, height, width = inputs[initial_time][0].shape
        
        # Process initial indices
        initial_c_idx, initial_h_idx, initial_w_idx = self.process_indices(
            initial_indices, [n_channels, height, width], device=device
        )
        
        # Initialize sensitivity tensor
        sensitivity_shape = (
            len(initial_c_idx), len(initial_h_idx), len(initial_w_idx)
        )
        sensitivity = torch.zeros(sensitivity_shape, device=device)
        
        for c_idx in range(len(initial_c_idx)):
            for h_idx in range(len(initial_h_idx)):
                for w_idx in range(len(initial_w_idx)):
                    # Reset computation for each point of interest
                    # Start with a fresh initial input for each sensitivity computation
                    initial_input = inputs[initial_time][0].detach().clone().to(device)
                    # Set requires_grad only for the specific index we're analyzing
                    initial_input.requires_grad_(False)  # Turn off grad for all
                    c = initial_c_idx[c_idx]
                    h = initial_h_idx[h_idx]
                    w = initial_w_idx[w_idx]
                    initial_input[0, c, h, w].requires_grad_(True)  # Turn on for specific element
                    
                    # Choose the forward function
                    forward_func = self.checkpointed_forward_once if use_checkpointing else self.forward_once
                    
                    # Perform autoregressive rollout with careful gradient tracking
                    current_input = initial_input
                    for t in range(initial_time, final_time + 1):
                        # Forward pass
                        output = forward_func(current_input)
                        
                        # Prepare for next time step if needed
                        if t < final_time:
                            # Get boundary conditions from inputs
                            boundary = inputs[t+1][0][:, self.output_channels:].to(device)
                            
                            # Create next input with gradient tracking preserved
                            current_input = torch.cat([
                                output,  # Keep batch dimension
                                boundary
                            ], dim=1)

                    print("Forward pass completed for sensitivity computation at final time step: ", final_time)

                    # Process final indices based on output dimensions
                    output_channels, output_height, output_width = output.shape
                    final_c_idx, final_h_idx, final_w_idx = self.process_indices(
                        final_indices, [output_channels, output_height, output_width], device=device
                    )

                    # Use the correct final index for this computation
                    fc = final_c_idx[c_idx % len(final_c_idx)]  # Cycle through if dimensions don't match
                    fh = final_h_idx[h_idx % len(final_h_idx)]
                    fw = final_w_idx[w_idx % len(final_w_idx)]

                    # Create gradient mask for the specific final point
                    grad_mask = torch.zeros_like(output, device=device)
                    grad_mask[fc, fh, fw] = 1.0

                    # Get gradient with respect to this final point
                    output.backward(grad_mask)

                    # Store the sensitivity value
                    sensitivity[c_idx, h_idx, w_idx] = initial_input.grad[0, c, h, w].item()
        
        return sensitivity

def generate_model_rollout(
    N_eval, test_data, model, hist, N_out, N_extra, initial_input=None, train=False
):

    N_test = test_data.size

    mean_out = test_data.out_mean.to_array().to_numpy().reshape(-1)
    std_out = test_data.out_std.to_array().to_numpy().reshape(-1)
    norm_vals = {"m_out": mean_out, "s_out": std_out}

    model.eval()
    model_pred = np.zeros((N_eval, 180, 360, N_out))

    with torch.no_grad():
        outs = model.inference(test_data, initial_input, num_steps=N_eval // (hist + 1))
    for i in range(N_eval // (hist + 1)):
        pred_temp = outs[i]
        pred_temp = torch.nan_to_num(pred_temp)
        pred_temp = torch.clip(pred_temp, min=-1e5, max=1e5)
        C, H, W = pred_temp.shape
        pred_temp = torch.reshape(pred_temp, (hist + 1, C // (hist + 1), H, W))
        model_pred[i * (hist + 1) : (i + 1) * (hist + 1)] = torch.swapaxes(
            torch.swapaxes(pred_temp, 3, 1), 2, 1
        ).cpu()

    if train:
        return model_pred
    else:
        return model_pred * norm_vals["s_out"] + norm_vals["m_out"], outs
