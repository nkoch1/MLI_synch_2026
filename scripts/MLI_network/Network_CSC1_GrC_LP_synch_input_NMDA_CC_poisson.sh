#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2-00:00				# max walltime
#SBATCH --cpus-per-task=1         # number of cores
#SBATCH --mem=10G
#SBATCH --output=%j.out		# file name for the output
#SBATCH --error=%j.err		# file name for errors

module load StdEnv/2023 julia/1.10.0
export JULIA_DEPOT_PATH="./julia:$JULIA_DEPOT_PATH"
julia -p 1 --threads 1 --project=@. Network_CSC1_GrC_LP_synch_input_NMDA_CC_poisson.jl $1 $2 $3 $4 $5 $6 $7 $8