import xarray as xr
import gc
import resource
import zarr
import numpy as np

def get_memory_usage():
    """Return the memory usage in MB using the resource module."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Convert to MB

def print_memory(message):
    print(f"{message}: {get_memory_usage():.2f} MB")
    
# Define input and output files
chunks = [
    "3D_thermo_dynamic_all_prediction_chunk1.zarr",
    "3D_thermo_dynamic_all_prediction_chunk2.zarr"
]
output_file = "3D_thermo_dynamic_all_prediction_merged.zarr"

print_memory("Initial memory usage")

# Let's examine the first chunk to understand its structure
with xr.open_zarr(chunks[0]) as ds:
    print(f"Dataset structure: {list(ds.data_vars)}")
    print(f"Coordinates: {list(ds.coords)}")
    
    # Create a new empty dataset for output with the first timestep
    print("Creating initial dataset structure")
    ds_init = ds.isel(time=slice(0, 1))
    ds_init.to_zarr(output_file, mode="w")
    
    print_memory("After creating initial structure")

# Now process each chunk and append to the output
time_lengths = []
for i, chunk_file in enumerate(chunks):
    print(f"Processing chunk {i+1}")
    print_memory(f"Before opening chunk {i+1}")
    
    with xr.open_zarr(chunk_file) as ds:
        time_lengths.append(len(ds.time))
        print(f"Chunk {i+1} has {len(ds.time)} time steps")
        
        if i == 0:
            # First chunk is already in the output
            # Just verify the time dimension matches what we expect
            print("First chunk already initialized")
        else:
            # For subsequent chunks, append along time dimension
            print(f"Appending chunk {i+1}")
            ds.to_zarr(output_file, append_dim="time")
        
        print_memory(f"After processing chunk {i+1}")
    
    # Force garbage collection
    gc.collect()
    print_memory(f"After garbage collection")

# Validate the merge
with xr.open_zarr(output_file) as ds:
    print(f"Final dataset has {len(ds.time)} time steps")
    expected_length = sum(time_lengths)
    if len(ds.time) == expected_length:
        print(f"Merge successful! Total time steps: {len(ds.time)}")
    else:
        print(f"Warning: Expected {expected_length} time steps, got {len(ds.time)}")

print("Merge complete!")
print_memory("Final memory usage")
