#!/bin/bash
#
#SBATCH --job-name=space_cpm
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=10000M
#SBATCH  --gres=gpu:1

python src/space_ray.py  --mode spl --test features/cpm_test.csv --instances features/cpm_train.csv
