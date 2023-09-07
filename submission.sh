#!/bin/bash
# number of compute nodes
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node 1
#SBATCH -t 00:30:00
#SBATCH -p t4_normal_q
#SBATCH -A cmda3634_rjh
#SBATCH -o foo.out

# Submit this file as a job request with
# sbatch submission.sh

# Change to the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR

# Unload all except default modules
module reset

# Load the modules you need
module load gompi/2021b

# build kmeans code
mpicc -D_FILE_OFFSET_BITS=64 -o kmeans kmeansit.c -lm

# Print the number of threads for future reference
echo "Running kmeans"

# Run the program. Don't forget arguments!
mpirun -n 4 ./kmeans s1.txt 5000 2 15


# The script will exit whether we give the "exit" command or not.
exit
