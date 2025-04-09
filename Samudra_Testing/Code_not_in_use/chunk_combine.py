import xarray as xr
import gc
import resource

def get_memory_usage():
    """Return the memory usage in MB using the resource module."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Convert to MB

exp_num_in = "3D_thermo_all" # "3D_thermo_all" or "3D_thermo_dynamic_all"
exp_num_out = exp_num_in

# Define input and output files
chunks = [
    f"{exp_num_in}_prediction_chunk{i}.zarr"
         for i in range(1,7)]
output_file = f"{exp_num_out}_prediction.zarr"

print(f"Initial memory usage: {get_memory_usage():.2f} MB")

# Process one chunk at a time to examine structure
with xr.open_zarr(chunks[0]) as first_chunk:
    # Create output zarr file with same structure but empty
    first_chunk.isel(time=0).expand_dims("time", 0).to_zarr(output_file, mode="w")
    print(f"Memory after creating output structure: {get_memory_usage():.2f} MB")

# Append each chunk to the output one at a time
for i, chunk_file in enumerate(chunks):
    print(f"Processing chunk {i+1}/{len(chunks)}")
    print(f"Memory before opening chunk {i+1}: {get_memory_usage():.2f} MB")
    
    # Open current chunk
    with xr.open_zarr(chunk_file) as ds:
        print(f"Memory after opening chunk {i+1}: {get_memory_usage():.2f} MB")
        # Append to existing zarr
        ds.to_zarr(output_file, append_dim="time")
        print(f"Memory after writing chunk {i+1}: {get_memory_usage():.2f} MB")
    
    # Force garbage collection after each chunk
    gc.collect()
    print(f"Memory after garbage collection: {get_memory_usage():.2f} MB")
    print(f"Chunk {i+1} appended and memory cleared")

print("Merge complete!")
print(f"Final memory usage: {get_memory_usage():.2f} MB")
