#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=5-12:00				# max walltime
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7         # number of cores
#SBATCH --mem=10G
#SBATCH --output=%j.out		# file name for the output
#SBATCH --error=%j.err		# file name for errors

module load StdEnv/2023 julia/1.10.0
export JULIA_DEPOT_PATH="./julia:$JULIA_DEPOT_PATH"
julia --threads ${SLURM_CPUS_PER_TASK} --project=@. CSC_cable_soma_Ih_Ihold_fitting_-50.jl
