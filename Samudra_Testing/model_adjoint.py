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

    def checkpointed_forward_once(self, x):
        """
        Wrapper for forward_once that uses gradient checkpointing for memory efficiency
        """
        # Define a custom forward function that can be checkpointed
        def custom_forward(*inputs):
            return self.forward_once(inputs[0])
        
        # Apply checkpointing
        return checkpoint.checkpoint(custom_forward, x, use_reentrant=False)
    
    def grad_track_one_element(self, state_tensor, initial_index,
                               device="cuda"):
        """
        Utility function to track the gradient of a single element in the output with respect to a single input element.
        
        state_tensor : torch.Tensor
            The state tensor from which to track the gradient.
        initial_index : tuple
            A tuple (b, c, h, w) indicating the batch, channel, height, and width index of the element to track.
        """
        state_tensor = state_tensor.clone().detach().to(device)  # Ensure we don't modify the original tensor

        # Retrieve desired element
        element_value = state_tensor[initial_index].item()
        # Create a tensor with requires_grad=True for the specific element
        input_element = torch.tensor([element_value], requires_grad=True, device=device)

        # Inject the input element into the state tensor
        state_tensor[initial_index] = input_element

        return state_tensor, input_element
    
    def grad_track_multiple_elements(self, state_tensor, 
                                     initial_indices,
                                     device="cuda"):
        """
        Utility function to track the gradient of multiple elements in the output with respect to multiple input elements.
        """
        state_tensor = state_tensor.clone().detach().to(device)

        # Create a tensor to hold the gradients for each of the specified elements
        input_elements = []
        for initial_index in initial_indices:
            # Retrieve the value of the specific element
            element_value = state_tensor[initial_index].item()  # Get the value of the element at the specified index
            # Create a tensor with requires_grad=True for this element
            input_element = torch.tensor([element_value], requires_grad=True, device=device)

            # Inject the input element into the state tensor
            state_tensor[initial_index] = input_element

            input_elements.append(input_element)

        # Return the modified state tensor and the list of input elements
        return state_tensor, input_elements
        
    def compute_single_element_sensitivity(self, inputs, 
                                     initial_index,  # Initial element indices
                                     final_index,        # Final element indices
                                     initial_time=0,
                                     final_time=-1,
                                     device="cuda"):
        """
        Computes sensitivity of a single output element with respect to a single input element.

        Parameters:
        -----------
        inputs : list of tensors
            The input tensors for each time step in the sequence.
        initial_time : int
            The initial time step for the autoregressive rollout.
        final_time : int
            The final time step for the autoregressive rollout.
        initial_index : tuple
            A tuple (b, c, h, w) indicating the batch, channel, height, and width index 
            of the initial element to track.
        final_index : tuple
            A tuple (b,c, h, w) indicating the channel, height, and width index of the final output element.

        """
        if final_time < 0:
            final_time = len(inputs) + final_time

        model_input = inputs[initial_time][0].clone().detach().to(device)  # Start with the initial input tensor

        # Track the gradient of the desired element in the input tensor
        model_input, input_element = self.grad_track_one_element(
            model_input,
            initial_index,
            device=device
        )

        # Run the full autoregressive rollout
        current_input = model_input
        for t in range(initial_time, final_time + 1):
            # Forward pass
            output = self.forward_once(current_input)
            
            # Prepare for next time step if needed
            if t < final_time:
                boundary = inputs[t+1][0][:, self.output_channels:].to(device)
                current_input = torch.cat([output, boundary], dim=1)
        
        # Get the specific output element we're interested in
        output_value = output[final_index]  
        
        # Compute the gradient
        output_value.backward()
        gradient = input_element.grad.item()
        
        return gradient
    
    def compute_state_sensitivity_iterative(self, inputs, 
                          initial_indices,
                          final_indices,
                          initial_time=0, 
                          final_time=-1,
                          device="cuda",
                          use_checkpointing=True):
        """
        Computes the full sensitivity matrix showing how each initial element
        affects each final element.

        Parameters:
        -----------
        inputs : list of tensors
            The input tensors for each time step in the sequence.
        initial_indices : list of tuples
            A list of tuples (b, c, h, w) indicating the batch, channel, height, and width indices
            of the initial elements to track. This should be a list of tuples for multiple elements.
        final_indices : list of tuples
            A list of tuples (b, c, h, w) indicating the batch, channel, height, and width indices
            of the final output elements to compute sensitivities for. This should also be a list of tuples. 
        initial_time : int
            The initial time step for the autoregressive rollout.
        final_time : int    
            The final time step for the autoregressive rollout. 
            If negative, it will be computed as len(inputs) + final_time.
        """
        # Process final time if negative
        if final_time < 0:
            final_time = len(inputs) + final_time
        
        # Initialize 2D sensitivity tensor
        sensitivity_shape = (len(initial_indices), len(final_indices))
        sensitivity = torch.zeros(sensitivity_shape, device=device)
        
        # Total computations for progress reporting
        total_computations = len(initial_indices) * len(final_indices)
        completed = 0
        
        # Iterate through each initial point
        for ic_idx, initial_index in enumerate(initial_indices):
            for ih_idx, final_index in enumerate(final_indices):
                # Compute single element sensitivity
                gradient = self.compute_single_element_sensitivity(
                    inputs,
                    initial_time=initial_time,
                    final_time=final_time,
                    initial_index=initial_index,  # Initial element index (b, c, h, w)
                    final_index=final_index,        # Final element index (b, c, h, w)  
                    device=device
                )
                                
                # Store the result in the tensor
                sensitivity[ic_idx, ih_idx] = gradient
                
                # Update progress
                completed += 1
                if completed % 10 == 0 or completed == total_computations:
                    print(f"Progress: {completed}/{total_computations} sensitivities computed "
                        f"({100*completed/total_computations:.2f}%)")
        
        return sensitivity
    
    def compute_state_sensitivity(self, inputs,
                            initial_indices,
                            final_indices,
                            initial_time=0, 
                            final_time=-1,
                            device="cuda",
                            use_checkpointing=True):
        """
        Alternate version of compute_state_sensitivity_iterative that 
        uses autograd.grad to compute the sensitivity in a more efficient manner.
        """
        # Process final time if negative
        if final_time < 0:
            final_time = len(inputs) + final_time
        
        
        # Run the forward pass for the entire sequence to get the final output
        model_input = inputs[initial_time][0].clone().detach().to(device)  # Start with the initial input tensor
        # Run the full autoregressive rollout

        model_input, input_elements = self.grad_track_multiple_elements(
            model_input,
            initial_indices,  # List of initial indices to track
            device=device
        )

        # Run the full autoregressive rollout
        current_input = model_input
        for t in range(initial_time, final_time + 1):
            # Forward pass
            if use_checkpointing:
                output = self.checkpointed_forward_once(current_input)
            else:
                output = self.forward_once(current_input)
            
            # Prepare for next time step if needed
            if t < final_time:
                boundary = inputs[t+1][0][:, self.output_channels:].to(device)
                current_input = torch.cat([output, boundary], dim=1)
        # Get the final output tensor
        
        final_output = output

        # Create a vector for the final output elements we want to compute sensitivities for
        # Create a vector for the intial input elements we want to compute sensitivities for
        # Use torch.autograd.grad to compute the sensitivities for each of the final points 
        #   with respect to the initial input elements, at the same time

    def compute_state_sensitivity(self, inputs,
                        initial_indices,
                        final_indices,
                        initial_time=0, 
                        final_time=-1,
                        device="cuda",
                        use_checkpointing=True):
        """
        Computes the sensitivity of multiple final output elements with respect to 
        multiple initial elements in a single backward pass.
        
        Parameters:
        -----------
        inputs : list of tensors
            The input tensors for each time step in the sequence.
        initial_indices : list of tuples
            A list of tuples (b, c, h, w) indicating the batch, channel, height, and width indices
            of the initial elements to track.
        final_indices : list of tuples
            A list of tuples (b, c, h, w) indicating the batch, channel, height, and width indices
            of the final output elements to compute sensitivities for.
        initial_time : int
            The initial time step for the autoregressive rollout.
        final_time : int    
            The final time step for the autoregressive rollout. 
            If negative, it will be computed as len(inputs) + final_time.
        device : str
            The device to run the computation on ('cuda' or 'cpu').
        use_checkpointing : bool
            Whether to use gradient checkpointing for memory efficiency.
            
        Returns:
        --------
        torch.Tensor
            A 2D tensor of shape (len(initial_indices), len(final_indices)) containing 
            the sensitivity of each final element with respect to each initial element.
        """
        # 0->[[0, 1], [2, 3]]; 1->[[2, 3], [4, 5]]; 2->[[4, 5], [6, 7]]; 3->[[6, 7], [8, 9]]
        N, C, H, W = inputs[0].shape
        print(f"Input shape: {inputs[0].shape}")
        print(f"First input shape: {inputs[0][0].shape}")
        raise ValueError("Debug")
        
        # Process final time if negative
        if final_time < 0:
            final_time = len(inputs) + final_time
        
        # Initialize the model input
        model_input = inputs[initial_time][0].clone().detach().to(device)
        model_input.requires_grad_(True)
        
        # Store the initial input elements that we want to track
        initial_elements = []
        for idx in initial_indices:
            b, c, h, w = idx
            initial_elements.append(model_input[b, c, h, w])
        
        # Run the full autoregressive rollout
        current_input = model_input
        for t in range(initial_time, final_time // 2 ):
            # Forward pass
            if use_checkpointing and t < final_time:
                output = self.checkpointed_forward_once(current_input)
            else:
                output = self.forward_once(current_input)
            
            # Prepare for next time step if needed
            if t < final_time:
                boundary = inputs[t+1][0][:, self.output_channels:].to(device)
                current_input = torch.cat([output, boundary], dim=1)
        
        # Get the final output tensor
        final_output = output
        
        # Create a sensitivity matrix to store results
        sensitivity_matrix = torch.zeros(len(initial_indices), len(final_indices), device=device)
        
        # Extract the final output elements we're interested in
        final_elements = []
        for idx in final_indices:
            b, c, h, w = idx
            final_elements.append(final_output[b, c, h, w])
        
        # For each final element, compute gradients w.r.t all initial elements
        for j, final_element in enumerate(final_elements):
            # Clear any previous gradients
            if model_input.grad is not None:
                model_input.grad.zero_()
            
            # Compute gradients of the current final element w.r.t the input
            final_element.backward(retain_graph=(j < len(final_elements) - 1))
            
            # Extract gradients for the initial elements we're tracking
            for i, idx in enumerate(initial_indices):
                b, c, h, w = idx
                if model_input.grad is not None:
                    sensitivity_matrix[i, j] = model_input.grad[b, c, h, w].item()
        
        return sensitivity_matrix
        
        


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
