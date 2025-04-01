import torch
import numpy as np
import sys

sys.path.append("/nobackup/sruiz5/SAMUDRATEST/Samudra/samudra/")

from model import Samudra, generate_model_rollout

class SamudraAdjoint:
    """
    Implementation of the adjoint method for the Samudra ocean model.
    
    This class provides functionality to compute gradients of a loss function
    with respect to model inputs or parameters using the adjoint method.
    """
    
    def __init__(self, model, device='cuda'):
        """
        Initialize the adjoint method for Samudra.
        
        Args:
            model: The Samudra model instance
            device: Device to run computations on (default: 'cuda')
        """
        self.model = model
        #self.data_loader = data_loader
        self.device = device
        self.model.to(device)
        
    def compute_loss(self, simulated_states, observed_states):
        """
        Compute the loss between simulated states and observations.
        
        Args:
            simulated_states: Tensor of simulated ocean states
            observed_states: Tensor of observed ocean states
            
        Returns:
            Tensor containing the total loss
        """
        # Create a mask for valid (non-NaN) observations
        mask = ~torch.isnan(observed_states)
        
        # Compute squared difference for valid observations
        squared_diff = ((simulated_states - observed_states) ** 2)
        masked_squared_diff = squared_diff * mask
        
        # Sum over all valid observations
        loss = torch.sum(masked_squared_diff)
        
        return loss

    
    def compute_initial_state_gradient(self, initial_state, observed_states):
        """
        Compute the gradient of loss with respect to initial state using a single backward pass.
        
        Args:
            initial_state: Initial state to start simulation from
            observed_states: List of observed states at each timestep
            
        Returns:
            Gradient of loss with respect to initial state (dJ/dx_0)
        """
        # Start gradient tracking from the initial state
        x_0 = initial_state.clone().detach().requires_grad_(True)
        
        # Run forward simulation and accumulate loss
        simulated_states = [x_0]
        total_loss = 0.0
        
        # Forward pass through all timesteps
        for t in range(len(observed_states) - 1):
            next_state = self.model.forward_once(simulated_states[-1])
            simulated_states.append(next_state)
            
            # Accumulate loss at each timestep
            current_loss = self.compute_loss(simulated_states[-1], observed_states[t+1])
            total_loss += current_loss
        
        # Add loss from initial state
        initial_loss = self.compute_loss(simulated_states[0], observed_states[0])
        total_loss += initial_loss
        
        # Backward pass to compute gradient
        total_loss.backward()
        
        # Get gradient with respect to initial state
        initial_gradient = x_0.grad.clone()
        
        return initial_gradient, simulated_states, total_loss.detach()
    
    

