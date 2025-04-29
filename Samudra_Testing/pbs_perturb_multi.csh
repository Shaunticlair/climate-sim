#!/bin/csh
#PBS -S /bin/csh
#PBS -q R21940709
#PBS -l select=6:ncpus=40:model=sky_ele
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o ./perturbation_${PBS_ARRAY_INDEX}.log
#PBS -m bea
#PBS -r n
#PBS -J 1-6

# The last PBS line tells us not to re-run a script that's failed

# Set unlimited stack size
limit stacksize unlimited

# Load required modules
module purge
module use -a /swbuild/analytix/tools/modulefiles
module load miniconda3/v4

set CONDA_PYTHON=/nobackup/sruiz5/conda/envs/samudra/bin/python

# Set working directory
set basedir = .
cd ${basedir}

# Define the array of channel numbers
set channels = (76 153 154 155 156 157)

# Calculate the index (PBS array jobs are 1-indexed)
@ idx = $PBS_ARRAY_INDEX - 1

# Get the channel for this array job
set channel = $channels[$idx]

# Run the samudra rollout script with the specific channel
echo "Starting Samudra model run for channel $channel at `date`"
${CONDA_PYTHON} samudra_perturbation_configurable.py $channel
echo "Finished Samudra model run for channel $channel at `date`"