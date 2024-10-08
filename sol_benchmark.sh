#!/bin/bash
#SBATCH -p general
#SBATCH -t 2-00:0:00
#SBATCH --mem=50GB
#SBATCH -G a100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<CHANGE-THIS-TO-YOUR-ASU-EMAIL>
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --export=NONE

module purge
module load mamba/latest
source activate /scratch/phegde7/.conda/envs/attention_guidance_v2

python benchmark.py -b <benchmark-name> -m <model-name>
