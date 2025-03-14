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

# Get metadata without loading data
zarr_group1 = zarr.open(chunks[0])
zarr_group2 = zarr.open(chunks[1])

print_memory("After opening zarr groups")

# Get the sizes and variable info from the first chunk
var_names = []
for name in zarr_group1:
    if name.startswith('.') or name == 'predictions':
        continue
    var_names.append(name)

print(f"Variables found: {var_names}")
print_memory("After collecting variable names")

# Now we'll read each variable from each chunk and append to the output
for var_name in var_names:
    print(f"Processing variable: {var_name}")
    
    # Open output file for this variable
    gc.collect()
    print_memory(f"Before merging {var_name}")
    
    # Collect variable data from all chunks
    datas = []
    for i, chunk_file in enumerate(chunks):
        print(f"  Reading from chunk {i+1}")
        with xr.open_zarr(chunk_file) as ds:
            # Only load one variable at a time
            var_data = ds[var_name]
            datas.append(var_data)
        
        # Force GC after reading each chunk
        gc.collect()
        print_memory(f"  After reading chunk {i+1}")
    
    # Concatenate and write only this variable
    print(f"  Concatenating {var_name}")
    concat_var = xr.concat(datas, dim="time")
    
    # Write to disk and clear memory
    if var_name == var_names[0]:
        # For first variable, create new zarr
        print(f"  Writing first variable {var_name}")
        concat_var.to_dataset().to_zarr(output_file, mode="w")
    else:
        # For subsequent variables, append to existing zarr
        print(f"  Appending variable {var_name}")
        concat_var.to_dataset().to_zarr(output_file, mode="a")
    
    # Clear memory after writing
    datas = None
    concat_var = None
    gc.collect()
    print_memory(f"  After writing {var_name}")

print("Merge complete!")
print_memory("Final memory usage")
