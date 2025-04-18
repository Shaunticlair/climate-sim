import torch
import torch.nn as nn
import numpy as np
import sys

import torch.utils.checkpoint as checkpoint

path ="/path/to/samudra/" # Replace with the actual path to the Samudra package
path = "/nobackup/sruiz5/SAMUDRATEST/Samudra/samudra/"
sys.path.append(path)
import model
import setup

null_timer = setup.NullTimer() # Timer class that does nothing, used as default


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
                  use_checkpointing=True,
                  timer =null_timer):
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
        
        timer.checkpoint("Finished setting up model input")
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

        timer.checkpoint("Finished setting up element tracking (maybe expensive?)")
        
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
            
            timer.checkpoint(f"Ran model forward for model iteration {it}")
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

            timer.checkpoint(f"Added gradient tracking for model iteration {it+1} (maybe expensive?)")


        
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
                
                print("Backwards run initiated")
                # Compute gradients
                output_element.backward(retain_graph=True)
                
                # Extract gradients for all input elements
                for in_idx, elem in element_map.items():
                    sensitivity_matrix[in_idx, orig_idx] = elem.grad.item() if elem.grad is not None else 0.0
        
        timer.checkpoint("Finished computing sensitivity matrix (includes backward pass)")

        return sensitivity_matrix

    def compute_fd_sensitivity(self, inputs, 
                            source_coords_list, 
                            target_coords_list, 
                            perturbation_size=1e-4,
                            device="cuda"):
        """
        Computes finite difference sensitivity between source and target coordinates.
        
        Parameters:
        -----------
        inputs : list of tensors
            The input tensors for each time step in the sequence.
        source_coords_list : list of tuples
            A list of tuples (b, c, h, w, t) indicating the batch, channel, height, width
            indices, and time step of the input elements to perturb.
        target_coords_list : list of tuples
            A list of tuples (b, c, h, w, t) indicating the batch, channel, height, width
            indices, and time step of the output elements to measure.
        perturbation_size : float
            Size of perturbation to apply for finite difference calculation.
        device : str
            The device to run the computation on ('cuda' or 'cpu').
                
        Returns:
        --------
        torch.Tensor
            A 2D tensor of shape (len(source_coords_list), len(target_coords_list)) containing 
            the finite difference sensitivity of each target element with respect to each source element.
        """
        # Make sure we're in evaluation mode
        self.eval()
        
        # Determine the time range
        source_times = [idx[4] for idx in source_coords_list]
        target_times = [idx[4] for idx in target_coords_list]
        min_time = min(min(source_times), min(target_times))
        max_time = max(target_times)
        
        # Create a sensitivity matrix to store results
        sensitivity_matrix = torch.zeros(len(source_coords_list), len(target_coords_list), device=device)
        
        # First get baseline outputs
        print("Computing baseline outputs...")
        baseline_outputs = {}
        
        # Run the baseline model once to get reference values
        current_input = inputs[min_time // 2][0].clone().to(device)
        baseline_outputs[min_time] = current_input
        
        for it in range(min_time // 2, max_time // 2 + 1):
            # Forward pass
            output = self.forward_once(current_input)
            
            # Store output for this time step
            time_step = it * 2 + 1
            baseline_outputs[time_step] = output
            
            # Prepare for next time step if needed
            if it < max_time // 2:
                boundary = inputs[it+1][0][:, self.output_channels:].to(device)
                current_input = torch.cat([output, boundary], dim=1)
                baseline_outputs[it * 2 + 2] = current_input
        
        # For each source coordinate, perturb and compute sensitivity
        for s_idx, source_coord in enumerate(source_coords_list):
            if s_idx % 10 == 0:
                print(f"Processing source coordinate {s_idx+1}/{len(source_coords_list)}")
            
            b, c, h, w, t = source_coord
            
            # Skip if time is outside our range
            if t < min_time or t > max_time:
                continue
            
            # Start from beginning for each perturbation
            current_input = inputs[min_time // 2][0].clone().to(device)
            
            # If perturbation is at initial timestep, apply it
            if t == min_time:
                current_input[b, c, h, w] += perturbation_size
            
            # Run the perturbed model forward
            perturbed_outputs = {min_time: current_input}
            
            for it in range(min_time // 2, max_time // 2 + 1):
                # Forward pass
                output = self.forward_once(current_input)
                
                # Store output for this time step
                time_step = it * 2 + 1
                perturbed_outputs[time_step] = output
                
                # Prepare for next time step if needed
                if it < max_time // 2:
                    boundary = inputs[it+1][0][:, self.output_channels:].to(device)
                    current_input = torch.cat([output, boundary], dim=1)
                    
                    # Apply perturbation if this is the right timestep
                    if it * 2 + 2 == t:
                        current_input[b, c, h, w] += perturbation_size
                    
                    perturbed_outputs[it * 2 + 2] = current_input
            
            # For each target coordinate, compute sensitivity
            for t_idx, target_coord in enumerate(target_coords_list):
                b_t, c_t, h_t, w_t, t_t = target_coord
                
                # Skip if target time is before source time (no causality)
                if t_t < t:
                    continue
                
                # Get baseline and perturbed values at target point
                baseline_value = baseline_outputs[t_t][b_t, c_t, h_t, w_t].item()
                perturbed_value = perturbed_outputs[t_t][b_t, c_t, h_t, w_t].item()
                
                # Calculate sensitivity
                sensitivity = (perturbed_value - baseline_value) / perturbation_size
                sensitivity_matrix[s_idx, t_idx] = sensitivity
        
        return sensitivity_matrix

    def grad_track_chunked(self, state_tensor, slices_list, device="cuda"):
        """
        Utility function to track the gradient of chunks of elements in the state tensor.
        This implementation uses slicing to efficiently track gradients for regions of interest.
        
        Parameters:
        -----------
        state_tensor : torch.Tensor
            The state tensor from which to track the gradient.
        slices_list : list of tuples
            A list of tuples (batch_slice, channel_slice, height_slice, width_slice) defining
            the regions to track gradients for.
        device : str
            The device to use for computation.
            
        Returns:
        --------
        augmented_tensor : torch.Tensor
            The state tensor with gradient tracking added for the specified chunks.
        tracked_elements : list
            A list of tensors with requires_grad=True for the specified chunks.
        """
        state_tensor = state_tensor.clone().to(device)  # Clone but don't detach to preserve the computational graph
        
        # Create a list to track the tensor elements with gradients
        tracked_elements = []
        
        for slice_tuple in slices_list:
            batch_slice, channel_slice, height_slice, width_slice = slice_tuple
            
            # Get the shape of this slice
            if isinstance(batch_slice, slice):
                batch_size = (batch_slice.stop or state_tensor.shape[0]) - (batch_slice.start or 0)
            else:
                batch_size = 1
                
            if isinstance(channel_slice, slice):
                channel_size = (channel_slice.stop or state_tensor.shape[1]) - (channel_slice.start or 0)
            else:
                channel_size = 1
                
            if isinstance(height_slice, slice):
                height_size = (height_slice.stop or state_tensor.shape[2]) - (height_slice.start or 0)
            else:
                height_size = 1
                
            if isinstance(width_slice, slice):
                width_size = (width_slice.stop or state_tensor.shape[3]) - (width_slice.start or 0)
            else:
                width_size = 1
            
            # Create a zeroed tensor with the shape of this slice
            chunk_shape = (batch_size, channel_size, height_size, width_size)
            grad_chunk = torch.zeros(chunk_shape, device=device, requires_grad=True)
            tracked_elements.append(grad_chunk)
            
            # Add the grad_chunk to the corresponding slice of state_tensor
            state_tensor[batch_slice, channel_slice, height_slice, width_slice] += grad_chunk
        
        return state_tensor, tracked_elements

    def track_gradients_for_timestep(self, state_tensor, chunks_dict, timestep, max_time, tracked_chunks, device="cuda"):
        """
        Adds gradient tracking for chunks specified for a given timestep and the next timestep if present.
        
        Parameters:
        -----------
        state_tensor : torch.Tensor
            The state tensor to augment with gradient tracking.
        chunks_dict : dict
            Dictionary mapping timesteps to lists of chunk specifications.
        timestep : int
            Current timestep to check for chunks.
        max_time : int
            Maximum time to consider.
        tracked_chunks : dict
            Dictionary to store the tracked chunks, will be updated in-place.
        device : str
            The device to use for computation.
            
        Returns:
        --------
        torch.Tensor
            The augmented state tensor with gradient tracking.
        """
        # Track gradients for the current timestep if specified
        if timestep in chunks_dict:
            state_tensor, chunks = self.grad_track_chunked(
                state_tensor, chunks_dict[timestep], device=device
            )
            tracked_chunks[timestep] = chunks
        
        # Track gradients for the next timestep if it's within range and specified
        next_timestep = timestep + 1
        if next_timestep in chunks_dict and next_timestep <= max_time:
            state_tensor, next_chunks = self.grad_track_chunked(
                state_tensor, chunks_dict[next_timestep], device=device
            )
            tracked_chunks[next_timestep] = next_chunks
        
        return state_tensor

    def compute_state_sensitivity_chunked(self, inputs,
                    in_chunks_dict,
                    out_indices,
                    device="cuda",
                    use_checkpointing=True,
                    timer=null_timer):
        """
        Computes the sensitivity of output elements with respect to 
        chunks of input elements in a single backward pass, with time specified in the dictionary.
        
        Parameters:
        -----------
        inputs : list of tensors
            The input tensors for each time step in the sequence.
        in_chunks_dict : dict
            A dictionary mapping time steps to lists of slice tuples 
            (batch_slice, channel_slice, height_slice, width_slice) for gradient tracking.
        out_indices : list of tuples
            A list of tuples (b, c, h, w, t) indicating the batch, channel, height, width
            indices, and time step of the output elements to compute sensitivities for.
        device : str
            The device to run the computation on ('cuda' or 'cpu').
        use_checkpointing : bool
            Whether to use gradient checkpointing for memory efficiency.
                
        Returns:
        --------
        list of torch.Tensor
            A list of tensors, one per output element, containing the sensitivities
            of that output element with respect to each chunk of input elements.
        """
        # Make sure we're in evaluation mode
        self.eval()

        # Determine the time range we need to process
        in_times = list(in_chunks_dict.keys())
        out_times = [idx[4] for idx in out_indices]
        min_time = min(min(in_times), min(out_times))
        max_time = max(out_times)
        
        # Create mapping for output indices
        out_indices_by_time = {}
        for i, idx in enumerate(out_indices):
            time = idx[4]
            if time not in out_indices_by_time:
                out_indices_by_time[time] = []
            out_indices_by_time[time].append((i, idx[:4]))
        
        # Initialize the model input for the minimum time step
        initial_iter = min_time // 2
        model_input = inputs[initial_iter][0].clone().to(device)
        
        timer.checkpoint("Finished setting up model input")
        
        # Dictionary to store tracked elements for each time step and chunk
        tracked_chunks = {}
        
        # Add gradient tracking for initial timesteps
        model_input = self.track_gradients_for_timestep(
            model_input, in_chunks_dict, min_time, max_time, tracked_chunks, device
        )

        timer.checkpoint("Finished setting up chunk tracking")
        
        # Run the autoregressive rollout
        current_input = model_input
        outputs = {min_time: current_input}  # Store inputs/outputs by time step
        
        for it in range(initial_iter, max_time // 2 ):
            print(f"Processing time step: {it}")
            
            # Forward pass
            if use_checkpointing and it < max_time // 2:
                output = self.checkpointed_forward_once(current_input)
            else:
                output = self.forward_once(current_input)
            
            timer.checkpoint(f"Ran model forward for model iteration {it}")
            
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
                
                # Add gradient tracking for next timesteps
                current_input = self.track_gradients_for_timestep(
                    current_input, in_chunks_dict, next_time, max_time, tracked_chunks, device
                )

                timer.checkpoint(f"Added gradient tracking for model iteration {it+1}")
        
        # List to store sensitivity tensors for each output element
        sensitivity_results = []
        
        # For each output time step and indices
        for out_time, out_idx_list in out_indices_by_time.items():
            output_tensor = outputs[out_time]
            
            # Extract the output elements we're interested in
            for orig_idx, spatial_idx in out_idx_list:
                b, c, h, w = spatial_idx
                output_element = output_tensor[b, c, h, w]
                
                # Clear gradients before backward pass
                for time_chunks in tracked_chunks.values():
                    for chunk in time_chunks:
                        if chunk.grad is not None:
                            chunk.grad.zero_()
                
                timer.checkpoint("Cleared gradients before backward pass")
                # Compute gradients
                output_element.backward(retain_graph=True)

                timer.checkpoint("Backwards pass complete")
                
                # Gather sensitivity tensors for this output element
                sensitivity_for_element = []
                for time in sorted(tracked_chunks.keys()):
                    time_chunks = tracked_chunks[time]
                    for chunk in time_chunks:
                        sensitivity_for_element.append(chunk.grad.clone() if chunk.grad is not None else torch.zeros_like(chunk))
                
                sensitivity_results.append(sensitivity_for_element)

                timer.checkpoint("Gathered sensitivity for output element")
        
        timer.checkpoint("Finished computing sensitivity matrix (includes backward pass)")
        
        return sensitivity_results