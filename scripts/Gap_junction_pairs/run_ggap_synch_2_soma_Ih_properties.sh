#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --time=4-12:00				# max walltime
#SBATCH --cpus-per-task=1         # number of cores
#SBATCH --mem=24G
#SBATCH --output=%j.out		# file name for the output
#SBATCH --error=%j.err		# file name for errors

module load StdEnv/2023 julia/1.10.0
export JULIA_DEPOT_PATH="./julia:$JULIA_DEPOT_PATH"
julia -p ${SLURM_CPUS_PER_TASK} --threads ${SLURM_CPUS_PER_TASK} --project=@. ggap_synch_2_soma_Ih_properties.jl $1 $2 
