#!/bin/bash
#SBATCH --ntasks=6     ### How many CPU cores do you need?
#SBATCH --mem=64G      ### How much RAM memory do you need?
#SBATCH --gres=gpu:1   ### How many GPUs do you need?
#SBATCH --exclude=gpu-hm-001
#SBATCH -p hm        ### The queue to submit to: express, short, long, interactive
#SBATCH -t 10-23:59:59    ### The time limit in D-hh:mm:ss format
#SBATCH -o /data/scratch/dspaanderman/logs/Server/interactive_output_%j.log    ### Where to store the console output (%j is the job number)
#SBATCH -e /data/scratch/dspaanderman/logs/Server/interactive_error_%j.log  ### Where to store the error output
#SBATCH --job-name=IServer  ### Name your job so you can distinguish between jobs

module purge
module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1
module load OpenBLAS/0.3.15-GCC-10.3.0

source /trinity/home/dspaanderman/InteractiveNet/venv/bin/activate
python --version
which python

echo "Starting"
hostname
ssh -N -f -R 5044:localhost:5044 rad-hpc-master-001
monailabel start_server --app /trinity/home/dspaanderman/InteractiveNet/GUI/apps/interactivenet --studies /data/scratch/dspaanderman/STT_TTV --conf models Task801_WORC_CT+fastR --port 5044

echo "Ending"
