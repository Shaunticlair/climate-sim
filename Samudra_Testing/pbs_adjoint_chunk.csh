#!/bin/csh
#PBS -S /bin/csh
#PBS -q R21940709
#PBS -l select=1:ncpus=40:model=sky_ele
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -o ./samudra_adjoint_single_chunk_timer.log
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

# Set working directory
set basedir = .
cd ${basedir}

# Run the samudra rollout script
echo "Starting Samudra model run at `date`"
${CONDA_PYTHON} samudra_adjoint_chunk.py
#${CONDA_PYTHON} samudra_rollout.py
echo "Finished Samudra model run at `date`"

