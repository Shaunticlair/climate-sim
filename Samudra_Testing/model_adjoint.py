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
    
    def compute_fd_sensitivity(self, inputs, 
                            source_coords_dict, 
                            target_coords_dict, 
                            perturbation_size=1e-4,
                            device="cuda",
                            use_checkpointing=True,
                            timer=null_timer):
        """
        Computes finite difference sensitivity between source coordinates and target coordinates,
        structured similarly to compute_state_sensitivity.
        
        Parameters:
        -----------
        inputs : list of tensors
            The input tensors for each time step in the sequence.
        source_coords_dict : dict
            Dictionary mapping timesteps to lists of tuples.
            Each tuple contains (b, c, h, w) indicating the batch, channel, height, width
            indices of the input elements to perturb at the specified timestep.
        target_coords_dict : dict
            Dictionary mapping timesteps to lists of tuples.
            Each tuple contains (b, c, h, w) indicating the batch, channel, height, width
            indices of the output elements to measure at the specified timestep.
        perturbation_size : float
            Size of perturbation to apply for finite difference calculation.
        device : str
            The device to run the computation on ('cuda' or 'cpu').
        use_checkpointing : bool
            Whether to use gradient checkpointing for memory efficiency.
        timer : Timer
            Timer object for performance tracking.
                
        Returns:
        --------
        dict
            A dictionary where each key is a tuple (target_time, target_coord, source_time, source_coord)
            and each value is the sensitivity of the target with respect to the source.
        """
        # Make sure we're in evaluation mode
        self.eval()
        
        # Determine the time range
        source_times = list(source_coords_dict.keys())
        target_times = list(target_coords_dict.keys())
        min_time = min(min(source_times), min(target_times))
        max_time = max(max(source_times), max(target_times))
        
        # Process the initial time step
        initial_iter = min_time // 2
        
        timer.checkpoint("Starting finite difference sensitivity calculation")
        
        # Dictionary to store sensitivity results
        sensitivity_results = {}
        
        # First get baseline outputs
        print("Computing baseline outputs...")
        
        # Run the baseline model once to get reference values
        current_input = inputs[initial_iter][0].clone().to(device)
        baseline_outputs = {min_time: current_input,
                            min_time+1: current_input}
        
        # Forward pass without any perturbation to establish baseline
        for it in range(initial_iter, max_time // 2):
            # Forward pass
            if use_checkpointing and it < max_time // 2:
                output = self.checkpointed_forward_once(current_input)
            else:
                output = self.forward_once(current_input)
            
            # Store output for this time step
            time_step = it * 2 + 2
            baseline_outputs[time_step] = output
            next_odd_time = time_step + 1
            baseline_outputs[next_odd_time] = output
            
            # Prepare for next time step if needed
            if it < max_time // 2:
                boundary = inputs[it+1][0][:, self.output_channels:].to(device)
                current_input = torch.cat([output, boundary], dim=1)
                baseline_outputs[it * 2 + 2] = current_input
        
        timer.checkpoint("Baseline model run complete")
        
        # For each source time and source coordinates
        total_sources = sum(len(coords) for coords in source_coords_dict.values())
        source_count = 0
        
        for source_time, source_coords_list in source_coords_dict.items():
            for source_coord in source_coords_list:
                source_count += 1
                print(f"Processing source {source_count}/{total_sources} at time {source_time}")
                
                # Reset for this perturbation
                current_input = inputs[initial_iter][0].clone().to(device)
                perturbed_outputs = {min_time: current_input,
                                     min_time+1: current_input}  # Store perturbed inputs/outputs by time step
                
                # Apply perturbation if this is the starting time step
                if source_time == min_time:
                    b, c, h, w = source_coord
                    current_input[b, c, h, w] += perturbation_size


                # Forward pass with perturbation
                for it in range(initial_iter, max_time // 2 ):
                    # Forward pass
                    if use_checkpointing and it < max_time // 2:
                        output = self.checkpointed_forward_once(current_input)
                    else:
                        output = self.forward_once(current_input)
                    
                    # Store output for this time step
                    time_step = it * 2 + 2
                    perturbed_outputs[time_step] = output
                    next_odd_time = time_step + 1
                    perturbed_outputs[next_odd_time] = output
                    
                    # Prepare for next time step if needed
                    if it < max_time // 2:
                        boundary = inputs[it+1][0][:, self.output_channels:].to(device)
                        current_input = torch.cat([output, boundary], dim=1)
                        
                        # Apply perturbation if this is the right timestep
                        if it * 2 + 2 == source_time:
                            b, c, h, w = source_coord
                            current_input[b, c, h, w] += perturbation_size

                
                timer.checkpoint(f"Forward pass with perturbation at time {source_time}, coord {source_coord}")
                
                # For each target time and target coordinate, compute sensitivity
                for target_time, target_coords_list in target_coords_dict.items():
                    # Skip if target time is before source time (no causality)
                    if target_time < source_time:
                        continue
                    
                    for target_coord in target_coords_list:
                        b_t, c_t, h_t, w_t = target_coord
                        
                        # Get baseline and perturbed values at target point
                        baseline_value = baseline_outputs[target_time][b_t, c_t, h_t, w_t].item()
                        perturbed_value = perturbed_outputs[target_time][b_t, c_t, h_t, w_t].item()
                        
                        # Calculate sensitivity
                        sensitivity = (perturbed_value - baseline_value) / perturbation_size
                        
                        # Store result
                        index = (target_time, target_coord, source_time, source_coord)
                        sensitivity_results[index] = torch.tensor(sensitivity, device=device)
                
                timer.checkpoint(f"Computed sensitivities for source at time {source_time}, coord {source_coord}")
        
        timer.checkpoint("Finished computing finite difference sensitivity matrix")
        
        return sensitivity_results

    def track_gradients_for_chunk(self, state_tensor, slices_list, device="cuda"):
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
        tracked_elements : dict
            A dictionary mapping each slice tuple to the corresponding gradient tensor.

        """
        state_tensor = state_tensor.clone().to(device)  # Clone but don't detach to preserve the computational graph
        
        # Create a list to track the tensor elements with gradients
        tracked_dict_in_time = {}
        
        for slice_tuple in slices_list:
            batch_slice, channel_slice, height_slice, width_slice = slice_tuple
            
            chunk_example = state_tensor[batch_slice, channel_slice, height_slice, width_slice]
            # Get the sizes of each dimension from the example slice
            chunk_shape = chunk_example.shape

            grad_chunk = torch.zeros(chunk_shape, device=device, requires_grad=True)
            tracked_dict_in_time[str(slice_tuple)] = grad_chunk  # Store the chunk for later use
            
            # Add the grad_chunk to the corresponding slice of state_tensor
            state_tensor[batch_slice, channel_slice, height_slice, width_slice] += grad_chunk
        
        return state_tensor, tracked_dict_in_time

    def track_gradients_for_timestep(self, state_tensor, in_chunks_dict, timestep, max_time, tracked_dict, device="cuda"):
        """
        Adds gradient tracking for chunks specified for a given timestep and the next timestep if present.
        
        Parameters:
        -----------
        state_tensor : torch.Tensor
            The state tensor to augment with gradient tracking.
        in_chunks_dict : dict
            Dictionary mapping timesteps to lists of tuples of slices.
            Each tuple contains (batch_slice, channel_slice, height_slice, width_slice) indicating a 
            region of the state tensor to track gradients for. The timestep indicates which time step
            the chunks correspond to.
        timestep : int
            Current timestep to check for chunks. Should be even (if not, we will subtract one to make it even).
        max_time : int
            Maximum time to consider.
        tracked_dict : dict of dict of torch.Tensor
            A dictionary to store tracked elements for each timestep.
            Dictionary with timestep as keys and dictionaries of tracked chunks as values.
                These dictionaries have keys as spatial indices (b, c, h, w) for each input chunk
                and values as sensitivity tensors (gradients) for that input chunk.
            {in_time: {in_slice: sensitivity_tensor}}
            
        device : str
            The device to use for computation.
            
        Returns:
        --------
        torch.Tensor
            The augmented state tensor with gradient tracking.
        """
        timestep = (timestep // 2) * 2  # Ensure we are working with even timesteps

        # Track gradients for the current timestep if specified
        if timestep in in_chunks_dict:
            state_tensor, tracked_dict_in_time = self.track_gradients_for_chunk(
                state_tensor, in_chunks_dict[timestep], device=device
            )
            tracked_dict[timestep] = tracked_dict_in_time
        
        # Track gradients for the next timestep if it's within range and specified
        next_timestep = timestep + 1
        if next_timestep in in_chunks_dict and next_timestep <= max_time:
            state_tensor, tracked_dict_in_time = self.track_gradients_for_chunk(
                state_tensor, in_chunks_dict[next_timestep], device=device
            )
            tracked_dict[next_timestep] = tracked_dict_in_time
        
        return state_tensor

    def compute_state_sensitivity(self, inputs,
                in_chunks_dict,
                out_boxes_dict,
                device="cuda",
                use_checkpointing=True,
                timer=null_timer):
        """
        Computes the sensitivity of sums of output elements with respect to 
        chunks of input elements in a single backward pass, with time specified in the dictionary.
        
        Parameters:
        -----------
        inputs : list of tensors
            A Test dataset object
            When indexed into, it returns a tuple (input_tensor, target_tensor) for each time step.
        in_chunks_dict : dict
            Dictionary mapping timesteps to lists of tuples of slices.
            Each tuple contains (batch_slice, channel_slice, height_slice, width_slice) indicating a 
            region of the state tensor to track gradients for. The timestep indicates which time step
            the chunks correspond to.
        out_boxes_dict : dict
            A dictionary mapping time steps to lists of tuples of slices.
            Each tuple contains (batch_slice, channel_slice, height_slice, width_slice) indicating a 
            region of the output state tensor to sum over. The sensitivities will be computed for
            the sum of all elements in each box. The timestep indicates which time step
            the output boxes correspond to.
        device : str
            The device to run the computation on ('cuda' or 'cpu').
        use_checkpointing : bool
            Whether to use gradient checkpointing for memory efficiency.
        timer : Timer
            Timer object for performance tracking.
                
        Returns:
        --------
        dict of torch.Tensor
            Each key is a tuple (out_time, str(out_box), in_time, str(in_slice)) where:
                out_time : int
                    The time step of the output box.
                str(out_box) : str
                    String representation of slice tuple (batch_slice, channel_slice, height_slice, width_slice)
                    defining the output box. String representation is used because slices are not hashable.
                in_time : int
                    The time step of the input chunk.
                str(in_slice) : str
                    String representation of slice tuple (batch_slice, channel_slice, height_slice, width_slice)
                    defining the input chunk. String representation is used because slices are not hashable.
            
        """
        # Make sure we're in evaluation mode
        self.eval()

        # Determine the time range we need to process
        in_times = list(in_chunks_dict.keys())
        out_times = out_boxes_dict.keys()
        min_time = min(min(in_times), min(out_times))
        max_time = max(out_times)
        
        # Initialize the model input for the minimum time step
        initial_iter = min_time // 2
        model_input = inputs[initial_iter][0].clone().to(device)
        
        timer.checkpoint("Finished setting up model input")
        
        # Dictionary to store tracked elements for each time step and chunk
        tracked_dict = {}
        
        # Add gradient tracking for initial timesteps: even and odd
        model_input = self.track_gradients_for_timestep(
            model_input, in_chunks_dict, min_time, max_time, tracked_dict, device
        )

        timer.checkpoint("Finished setting up chunk tracking")
        
        # Run the autoregressive rollout
        current_input = model_input
        outputs = {min_time: current_input,
                   min_time+1: current_input}  # Store inputs/outputs by time step
        
        for it in range(initial_iter, max_time // 2 ):
            print(f"Processing time step: {it}")
            
            # Forward pass
            if use_checkpointing and it < max_time // 2:
                output = self.checkpointed_forward_once(current_input)
            else:
                output = self.forward_once(current_input)
            
            timer.checkpoint(f"Ran model forward for model iteration {it}")
            
            # Store output for this time step
            next_even_time = it * 2 + 2
            outputs[next_even_time] = output
            next_odd_time = next_even_time + 1  
            outputs[next_odd_time] = output
            
            # Prepare for next time step if needed
            if it < max_time // 2:
                boundary = inputs[it+1][0][:, self.output_channels:].to(device)
                current_input = torch.cat([output, boundary], dim=1)
                
                # Add gradient tracking for next timesteps
                current_input = self.track_gradients_for_timestep(
                    current_input, in_chunks_dict, next_even_time, max_time, tracked_dict, device
                )

                timer.checkpoint(f"Added gradient tracking for model iteration {it+1}")
        
        # Dict to store sensitivity tensors for each output element
        sensitivity_results = {}
        
        # For each output time step and output boxes
        for out_time, out_list_boxes in out_boxes_dict.items():   #### LOOP OVER OUTPUT TIME STEP
            output_tensor = outputs[out_time]

            # Process each output box
            for out_box in out_list_boxes:                          ### LOOP OVER OUTPUT BOXES
                batch_slice, channel_slice, height_slice, width_slice = out_box
                
                # Extract the output box and compute the sum of all elements
                output_box = output_tensor[batch_slice, channel_slice, height_slice, width_slice]
                output_sum = output_box.sum()  # Sum all elements in the box
                
                # Clear gradients before backward pass
                for tracked_dict_in_time in tracked_dict.values(): #Each time step has its own tracked chunks
                    for chunk in tracked_dict_in_time.values(): # Each chunk is a tensor
                        if chunk.grad is not None:
                            chunk.grad.zero_()
                
                timer.checkpoint("Cleared gradients before backward pass")
                
                # Compute gradients of the sum with respect to all inputs
                output_sum.backward(retain_graph=True)

                timer.checkpoint("Backwards pass complete")
                
                # Gather sensitivity tensors for this output sum
                for in_time, tracked_dict_in_time in tracked_dict.items(): ## LOOP OVER INPUT TIME STEPS
                    for in_slice, chunk in tracked_dict_in_time.items(): # LOOP OVER INPUT SLICES
                        # Get the gradient for this chunk
                        grad_chunk = chunk.grad.clone() if chunk.grad is not None else torch.zeros_like(chunk)

                        # Store the sensitivity tensor in the results dict using string representation of slices
                        index = (out_time, str(out_box), in_time, str(in_slice))
                        sensitivity_results[index] = grad_chunk
                        
                timer.checkpoint("Gathered sensitivity for output box")
        
        timer.checkpoint("Finished computing sensitivity matrix (includes backward pass)")
        
        return sensitivity_results
    
    def compute_loss_sensitivity(self, inputs,
                             in_chunks_dict,
                             initial_state=None,
                             max_time=None,
                             loss_fn=None,
                             device="cuda",
                             use_checkpointing=True,
                             timer=null_timer):
        """
        Computes the sensitivity of the accumulated loss function J with respect to 
        chunks of input elements in a single backward pass, with time specified in the dictionary.
        
        Parameters:
        -----------
        inputs : list of tensors
            A Test dataset object
            When indexed into, it returns a tuple (input_tensor, target_tensor) for each time step.
        in_chunks_dict : dict
            Dictionary mapping timesteps to lists of tuples of slices.
            Each tuple contains (batch_slice, channel_slice, height_slice, width_slice) indicating a 
            region of the state tensor to track gradients for. The timestep indicates which time step
            the chunks correspond to.
        initial_state : torch.Tensor, optional
            Optional initial state to use instead of inputs[min_time//2][0].
            If None, will use inputs[min_time//2][0].
        max_time : int, optional
            Maximum time step to compute loss for. If None, will use the maximum time in in_chunks_dict.
        loss_fn : callable, optional
            Loss function to use. Should take prediction and target tensors and return a scalar loss.
            If None, will use mean squared error.
        device : str
            The device to run the computation on ('cuda' or 'cpu').
        use_checkpointing : bool
            Whether to use gradient checkpointing for memory efficiency.
        timer : Timer
            Timer object for performance tracking.
                
        Returns:
        --------
        dict of torch.Tensor
            Each key is a tuple (in_time, str(in_slice)) where:
                in_time : int
                    The time step of the input chunk.
                str(in_slice) : str
                    String representation of slice tuple (batch_slice, channel_slice, height_slice, width_slice)
                    defining the input chunk. String representation is used because slices are not hashable.
            
        """
        # Make sure we're in evaluation mode
        self.eval()
        
        # Define default loss function if none provided
        if loss_fn is None:
            loss_fn = lambda pred, target: ((pred - target) ** 2).mean()

        # Determine the time range we need to process
        in_times = list(in_chunks_dict.keys())
        min_time = min(in_times)
        if max_time is None:
            max_time = max(in_times)
        
        # Initialize the model input for the minimum time step
        initial_iter = min_time // 2
        if initial_state is not None:
            model_input = initial_state.clone().to(device)
        else:
            model_input = inputs[initial_iter][0].clone().to(device)
        
        timer.checkpoint("Finished setting up model input")
        
        # Dictionary to store tracked elements for each time step and chunk
        tracked_dict = {}
        
        # Add gradient tracking for initial timesteps: even and odd
        model_input = self.track_gradients_for_timestep(
            model_input, in_chunks_dict, min_time, max_time, tracked_dict, device
        )

        timer.checkpoint("Finished setting up chunk tracking")
        
        # Run the autoregressive rollout
        current_input = model_input
        outputs = {min_time: current_input,
                min_time+1: current_input}  # Store inputs/outputs by time step
        
        # Initialize accumulated loss
        total_loss = 0.0
        
        for it in range(initial_iter, max_time // 2 + 1):
            print(f"Processing time step: {it}")
            
            # Forward pass
            if use_checkpointing and it < max_time // 2:
                output = self.checkpointed_forward_once(current_input)
            else:
                output = self.forward_once(current_input)
            
            timer.checkpoint(f"Ran model forward for model iteration {it}")
            
            # Store output for this time step
            next_even_time = it * 2 + 2
            outputs[next_even_time] = output
            next_odd_time = next_even_time + 1  
            outputs[next_odd_time] = output
            
            # Compute loss if this timestep is within max_time
            if next_even_time <= max_time:
                # Get target for this timestep
                target = inputs[it][1].to(device)
                
                # Compute loss for this timestep
                timestep_loss = loss_fn(output, target)
                
                # Accumulate loss
                total_loss = total_loss + timestep_loss
            
            # Prepare for next time step if needed
            if it < max_time // 2:
                boundary = inputs[it+1][0][:, self.output_channels:].to(device)
                current_input = torch.cat([output, boundary], dim=1)
                
                # Add gradient tracking for next timesteps
                current_input = self.track_gradients_for_timestep(
                    current_input, in_chunks_dict, next_even_time, max_time, tracked_dict, device
                )

                timer.checkpoint(f"Added gradient tracking for model iteration {it+1}")
        
        timer.checkpoint("Forward pass complete, accumulated loss")
        
        # Dict to store sensitivity tensors for each input chunk
        sensitivity_results = {}
        
        # Clear gradients before backward pass
        for tracked_dict_in_time in tracked_dict.values():
            for chunk in tracked_dict_in_time.values():
                if chunk.grad is not None:
                    chunk.grad.zero_()
        
        timer.checkpoint("Cleared gradients before backward pass")
        
        # Compute gradients of the accumulated loss with respect to all inputs
        total_loss.backward()

        timer.checkpoint("Backwards pass complete")
        
        # Gather sensitivity tensors for all input chunks
        for in_time, tracked_dict_in_time in tracked_dict.items():
            for in_slice, chunk in tracked_dict_in_time.items():
                # Get the gradient for this chunk
                grad_chunk = chunk.grad.clone() if chunk.grad is not None else torch.zeros_like(chunk)

                # Store the sensitivity tensor in the results dict using string representation of slices
                index = (in_time, in_slice)
                sensitivity_results[index] = grad_chunk
                
        timer.checkpoint("Gathered sensitivities for all input chunks")
        
        return sensitivity_results

