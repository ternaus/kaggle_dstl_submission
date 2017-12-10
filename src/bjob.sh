#!/bin/bash -l
#
#SBATCH --job-name="gioDSTL"
#SBATCH --exclusive
#SBATCH --partition=tflow
#SBATCH --time=24:00:00
#SBATCH --output=unet.%j.out
#SBATCH --error=unet.%j.err

#module load python/2.7.12
module load python/3.5.0
module load cudnn/8.0
module load gcc/6.1.0

#/cluster_nfs/Data_Apps/home/stud40/kaggle_dstl_submission/kaggle/lib/python3.5/site-packages/shapely/.libs/
export LD_LIBRARY_PATH="/apps/nvidia/cuda/lib64":$LD_LIBRARY_PATH
export CUDA_HOME="/apps/nvidia/cuda"

echo "working directory = "$SLURM_SUBMIT_DIR
echo "LD = "$LD_LIBRARY_PATH

#srun python get_3_band_shapes.py
#srun python cache_train.py
#srun python unet_buildings.py
#srun python make_prediction_cropped_buildings.py
#srun python merge_predictions.py
#srun python post_processing.py joined.csv
srun python visualize.py #-> need to export $DISPLAY

# remember to load the module python and gcc
# create a virtual env to install the pips
# remember to activate the virtalen
# 