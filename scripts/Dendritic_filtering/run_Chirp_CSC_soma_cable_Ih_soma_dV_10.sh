#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=3-12:00				# max walltime
#SBATCH --cpus-per-task=1         # number of cores
#SBATCH --ntasks=1
#SBATCH --mem=12GB
#SBATCH --output=%j.out		# file name for the output
#SBATCH --error=%j.err		# file name for errors

module load StdEnv/2023 julia/1.10.0
export JULIA_DEPOT_PATH="./julia:$JULIA_DEPOT_PATH"
julia -p 1 --threads 1 --project=@. Chirp_CSC_soma_cable_Ih_soma_dV.jl 10
