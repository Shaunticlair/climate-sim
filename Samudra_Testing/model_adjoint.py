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
    
    
    def grad_track_multiple_elements_DEFUNCT(self, state_tensor, 
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

    def grad_track_multiple_elements(self, state_tensor, 
                                     initial_indices,
                                     device="cuda"):
        """
        Utility function to track the gradient of multiple elements in the output with respect to multiple input elements.
        This implementation avoids breaking the computational graph by adding a zeroed tensor with grad tracking
        at the positions we want to track.
        """
        state_tensor = state_tensor.clone().to(device)  # Clone but don't detach to preserve the computational graph

        # Create a zeroed tensor with the same shape as state_tensor
        grad_tensor = torch.zeros_like(state_tensor, requires_grad=False)
        
        # Create a list to track the tensor elements with gradients
        input_elements = []
        
        for initial_index in initial_indices:
            initial_index = tuple(initial_index)  # Ensure it's a tuple
            # Create a tensor with requires_grad=True initialized to zero
            element = torch.zeros(1, device=device, requires_grad=True)
            input_elements.append(element)
            
            # Set the corresponding position in grad_tensor
            grad_tensor[initial_index] = element

        # Add the grad_tensor to state_tensor
        augmented_tensor = state_tensor + grad_tensor
        
        # Return the augmented tensor and the list of input elements
        return augmented_tensor, input_elements
    
    def compute_state_sensitivity(self, inputs,
                  in_indices,
                  out_indices,
                  device="cuda",
                  use_checkpointing=True):
        """
        Computes the sensitivity of output elements with respect to 
        input elements in a single backward pass, with time specified in the indices.
        
        Parameters:
        -----------
        inputs : list of tensors
            The input tensors for each time step in the sequence.
        in_indices : list of tuples
            A list of tuples (b, c, h, w, t) indicating the batch, channel, height, width
            indices, and time step of the input elements to track.
        out_indices : list of tuples
            A list of tuples (b, c, h, w, t) indicating the batch, channel, height, width
            indices, and time step of the output elements to compute sensitivities for.
        device : str
            The device to run the computation on ('cuda' or 'cpu').
        use_checkpointing : bool
            Whether to use gradient checkpointing for memory efficiency.
            
        Returns:
        --------
        torch.Tensor
            A 2D tensor of shape (len(in_indices), len(out_indices)) containing 
            the sensitivity of each output element with respect to each input element.
        """
        # Make sure we're in evaluation mode
        self.eval()

        # Determine the time range we need to process
        in_times = [idx[4] for idx in in_indices]
        out_times = [idx[4] for idx in out_indices]
        min_time = min(min(in_times), min(out_times))
        max_time = max(out_times)
        
        # Create mapping from time step to indices that belong to that time step
        in_indices_by_time = {}
        for i, idx in enumerate(in_indices):
            time = idx[4]
            if time not in in_indices_by_time:
                in_indices_by_time[time] = []
            # Store the original index position and the spatial indices (without time)
            in_indices_by_time[time].append((i, idx[:4]))
        
        # Create mapping for output indices as well
        out_indices_by_time = {}
        for i, idx in enumerate(out_indices):
            time = idx[4]
            if time not in out_indices_by_time:
                out_indices_by_time[time] = []
            out_indices_by_time[time].append((i, idx[:4]))
        
        # Initialize the model input for the minimum time step
        initial_iter = min_time // 2
        model_input = inputs[initial_iter][0].clone().to(device)
        
        # Track the gradients for input elements at the initial time step (even)
        if min_time in in_indices_by_time:
            spatial_indices = [idx for _, idx in in_indices_by_time[min_time]]
            model_input, initial_elements = self.grad_track_multiple_elements(model_input, spatial_indices, device=device)
            # Create a map to link original indices to tracked elements
            element_map = {in_indices_by_time[min_time][i][0]: elem for i, elem in enumerate(initial_elements)}
        else:
            element_map = {}

        # Check for odd timestep (second half of state vector)
        if min_time + 1 in in_indices_by_time and min_time + 1 <= max_time:
            spatial_indices = [idx for _, idx in in_indices_by_time[min_time + 1]]
            model_input, odd_elements = self.grad_track_multiple_elements(model_input, spatial_indices, device=device)
            # Add to the element map
            for i, elem in enumerate(odd_elements):
                element_map[in_indices_by_time[min_time + 1][i][0]] = elem
        
        # Run the autoregressive rollout
        current_input = model_input
        outputs = {min_time: current_input}  # Store inputs/outputs by time step
        
        for it in range(initial_iter, max_time // 2 + 1):
            print(f"Processing time step: {it}")
            
            # Forward pass
            if use_checkpointing and it < max_time // 2:
                output = self.checkpointed_forward_once(current_input)
            else:
                output = self.forward_once(current_input)
            
            # Store output for this time step
            time_step = it * 2 + 1
            outputs[time_step] = output
            
            # Prepare for next time step if needed
            if it < max_time // 2:
                boundary = inputs[it+1][0][:, self.output_channels:].to(device)
                current_input = torch.cat([output, boundary], dim=1)
                
            # Store the even timestep
            next_time = (it + 1) * 2
            outputs[next_time] = current_input
            
            # Check both the even and odd timesteps for the next iteration
            # Even timestep (first part of next state vector)
            if next_time in in_indices_by_time:
                spatial_indices = [idx for _, idx in in_indices_by_time[next_time]]
                current_input, new_elements = self.grad_track_multiple_elements(current_input, spatial_indices, device=device)
                # Add new elements to the map
                for i, elem in enumerate(new_elements):
                    element_map[in_indices_by_time[next_time][i][0]] = elem
            
            # Odd timestep (second part of next state vector)
            if next_time + 1 in in_indices_by_time and next_time + 1 <= max_time:
                spatial_indices = [idx for _, idx in in_indices_by_time[next_time + 1]]
                current_input, odd_elements = self.grad_track_multiple_elements(current_input, spatial_indices, device=device)
                # Add to the element map
                for i, elem in enumerate(odd_elements):
                    element_map[in_indices_by_time[next_time + 1][i][0]] = elem


        
        # Create a sensitivity matrix to store results
        sensitivity_matrix = torch.zeros(len(in_indices), len(out_indices), device=device)
        
        # For each output time step and indices
        for out_time, out_idx_list in out_indices_by_time.items():
            output_tensor = outputs[out_time]
            
            # Extract the output elements we're interested in
            for orig_idx, spatial_idx in out_idx_list:
                b, c, h, w = spatial_idx
                output_element = output_tensor[b, c, h, w]
                
                # Clear gradients before backward pass
                for elem in element_map.values():
                    if elem.grad is not None:
                        elem.grad.zero_()
                
                # Compute gradients
                output_element.backward(retain_graph=True)
                
                # Extract gradients for all input elements
                for in_idx, elem in element_map.items():
                    sensitivity_matrix[in_idx, orig_idx] = elem.grad.item() if elem.grad is not None else 0.0
        
        return sensitivity_matrix

