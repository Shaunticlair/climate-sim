#!/bin/csh
#PBS -S /bin/csh
#PBS -q normal
#PBS -l select=1:ncpus=20:model=ivy
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -o ./samudra_plotting.log
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
set CONDA_PYTHON=/nobackup/sruiz5/conda/envs/samudra/bin/python

# Set working directory
set basedir = .
cd ${basedir}

# Run the parallel plotting script
echo "Starting Samudra parallel plotting at `date`"
${CONDA_PYTHON} parallel_plotting.py
echo "Finished Samudra parallel plotting at `date`"
