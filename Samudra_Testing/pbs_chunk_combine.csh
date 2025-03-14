#!/bin/csh
#PBS -S /bin/csh
#PBS -q normal
#PBS -l select=1:ncpus=20:model=ivy
#PBS -l walltime=0:10:00
#PBS -j oe
#PBS -o ./samudra_output.log
#PBS -m bea
#PBS -r n

# The last PBS line tells us not to re-run a script that's failed

# Set unlimited stack size
limit stacksize unlimited

# Load required modules
module purge
module use -a /swbuild/analytix/tools/modulefiles
module load miniconda3/v4

# Set environment variables
#setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}

set CONDA_PYTHON=/nobackup/sruiz5/conda/envs/samudra/bin/python

# Activate conda environment
#source /swbuild/analytix/tools/miniconda3_220407/bin/activate
#source /swbuild/analytix/tools/miniconda3_220407/etc/profile.d/conda.csh
#conda init tcsh

#conda activate samudra

# Set working directory
set basedir = .
cd ${basedir}

# Make sure data directories exist
if (! -d data.zarr ) then
  echo 'Data directory "data.zarr" does not exist.'
  echo 'Please ensure data is downloaded before running.'
  exit 1
endif

if (! -d data_mean.zarr ) then
  echo 'Data directory "data_mean.zarr" does not exist.'
  echo 'Please ensure mean data is downloaded before running.'
  exit 1
endif

if (! -d data_std.zarr ) then
  echo 'Data directory "data_std.zarr" does not exist.'
  echo 'Please ensure std data is downloaded before running.'
  exit 1
endif

# Check if model weights exist
if (! -f samudra_thermo_dynamic_seed1.pt ) then
  echo 'Model weights file "samudra_thermo_dynamic_seed1.pt" does not exist.'
  echo 'Please ensure model weights are downloaded before running.'
  exit 1
endif

# Run the samudra rollout script
echo "Starting Samudra model run at `date`"
${CONDA_PYTHON} combine_files.py
#${CONDA_PYTHON} samudra_rollout.py
echo "Finished Samudra model run at `date`"

# Check if output was created
#if ( -d 3D_thermo_dynamic_all_prediction.zarr ) then
#  echo "Output successfully created at: 3D_thermo_dynamic_all_prediction.zarr"
#else
#  echo "Error: Output not created"
#  exit 1
#endif
