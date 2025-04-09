import torch
import torch.nn as nn
import numpy as np
import sys

import torch.utils.checkpoint as checkpoint

path ="/path/to/samudra/" # Replace with the actual path to the Samudra package
path = "/nobackup/sruiz5/SAMUDRATEST/Samudra/samudra/"
sys.path.append(path)
import model


class SamudraAdjoint(model.Samudra):
    """
    This class extends the functionality of the Samudra class, to also compute adjoint sensitivities.
    """
    def __init__(
        self,
        wet,
        hist,
        core_block=model.ConvNeXtBlock,
        down_sampling_block=model.AvgPool,
        up_sampling_block=model.BilinearUpsample,
        activation=model.CappedGELU,
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
            initial_index = tuple(initial_index)  # Ensure it's a tuple
            element_value = state_tensor[initial_index].item()  # Get the value of the element at the specified index
            # Create a tensor with requires_grad=True for this element
            input_element = torch.tensor([element_value], requires_grad=True, device=device)

            # Inject the input element into the state tensor
            state_tensor[initial_index] = input_element

            input_elements.append(input_element)

        # Return the modified state tensor and the list of input elements
        return state_tensor, input_elements
        
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
            If negative, it will be computed as 2*len(inputs) + final_time.
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
        # Make sure we're in evaluation mode
        self.eval()

        # Process final time if negative
        if final_time < 0:
            final_time = 2*len(inputs) + final_time

        initial_iter, final_iter = initial_time//2, final_time//2
        
        # Initialize the model input
        model_input = inputs[initial_iter][0].clone().detach().to(device)
        
        # Track the gradients of only the specific initial elements we care about
        model_input, initial_elements = self.grad_track_multiple_elements(model_input, initial_indices, device=device)
        
        # Run the full autoregressive rollout
        current_input = model_input
        for it in range(initial_iter, final_iter):
            print(f"Processing time step: {it}")
            
            # Forward pass
            if use_checkpointing and it < final_iter - 1:
                output = self.checkpointed_forward_once(current_input)
            else:
                output = self.forward_once(current_input)
            
            # Prepare for next time step if needed
            if it < final_iter - 1:
                boundary = inputs[it+1][0][:, self.output_channels:].to(device)
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
        
        # For each final element, compute gradients w.r.t. all initial elements
        for j, final_element in enumerate(final_elements):
            # Clear any previous gradients for our tracked elements
            for elem in initial_elements:
                if elem.grad is not None:
                    elem.grad.zero_()
            
            # Compute gradients of the current final element w.r.t the input
            final_element.backward(retain_graph=(j < len(final_elements) - 1))
            
            # Extract gradients directly from our tracked input elements
            for i, elem in enumerate(initial_elements):
                sensitivity_matrix[i, j] = elem.grad.item() if elem.grad is not None else 0.0
        
        return sensitivity_matrix
        
        