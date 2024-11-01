#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=t4:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --output=train-%j.out  #%j for jobID

#source /home/mohsenh/grCPuMsaEnv/bin/activate
source /home/mohsenh/projects/def-ilie/mohsenh/mohsenh/ENV/ppiENV/bin/activate

python PPI_Classifier.py