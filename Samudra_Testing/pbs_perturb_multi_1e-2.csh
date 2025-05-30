#!/bin/csh
#PBS -S /bin/csh
#PBS -q R21940709
#PBS -l select=6:ncpus=40:model=sky_ele
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -m bea
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

# Create log file with array index
set LOGFILE="./perturbation_${PBS_ARRAY_INDEX}.log"
echo "Starting job ${PBS_ARRAY_INDEX}" >! $LOGFILE

# Define the array of channel numbers
set channels = (76 153 154 155 156 157)

# PBS array jobs are 1-indexed
set idx = $PBS_ARRAY_INDEX
@ idx = $idx 

# Get the channel for this array job
set channel = $channels[$idx]

# Run the samudra rollout script with the specific channel
echo "Starting Samudra model run for channel $channel at `date`" >> $LOGFILE
$CONDA_PYTHON samudra_perturbation_configurable.py $channel 0.01 >>& $LOGFILE #0.01 = 1e-2
echo "Finished Samudra model run for channel $channel at `date`" >> $LOGFILE
