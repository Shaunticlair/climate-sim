import torch

def modify_tensor(tensor):
    tensor += 1  # In-place modification
    return tensor

# Create a list of tensors
tensor_list = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]

# Original state
print("Original:", tensor_list[0])

# Modify through function
x = tensor_list[0]
modify_tensor(x)

# Check if modification is reflected in the original list
print("After function call:", tensor_list[0])

def modify_slice(tensor):
    tensor[0:2] += 10  # In-place modification of a slice
    return tensor

y = tensor_list[1]  # The second tensor [4, 5, 6]
modify_slice(y)

# Check original list
print("After slice modification:", tensor_list[1])  # Will show [14, 15, 6]