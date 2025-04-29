#!/bin/csh
#PBS -S /bin/csh
#PBS -q R21940709
#PBS -l select=1:ncpus=40:model=sky_ele
#PBS -l walltime=12:00:00
#PBS -o ./perturbation_ch76.log
#PBS -j oe
#PBS -m bea

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

# Define the channel for this script
set channel = 76

# Run the samudra rollout script with the specific channel
echo "Starting Samudra model run for channel $channel at `date`"
$CONDA_PYTHON samudra_perturbation_configurable.py $channel
echo "Finished Samudra model run for channel $channel at `date`"