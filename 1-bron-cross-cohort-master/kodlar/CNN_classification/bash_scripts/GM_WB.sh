#!/bin/bash
#SBATCH --ntasks=6
#SBATCH --mem=14G
#SBATCH --gres=gpu:GTX:1
#SBATCH -t 9-00:00:00
#SBATCH -o /media/data/ebron/output/out_%j.log
#SBATCH -e /media/data/ebron/output/error_%j.log

# activate virtual environment
source /media/data/ebron/virtualenvs/CNN-for-AD/bin/activate

## mount BIGR cluster
#sshfs -o IdentityFile=/media/data/ebron/.ssh/id_rsa -o StrictHostKeyChecking=no ebron@bigr-app003:/scratch /media/data/ebron/bigr_mount/

module load python/3.6.7
module load tensorflow/1.12.0

# copy data to temp job dir
ROI=GM_WB
MY_TMP_DIR=/slurmtmp/${SLURM_JOB_USER}.${SLURM_JOB_ID}
cp -r /media/data/ebron/bigr_mount/tmichelotti/data_ADNI_${ROI}/*_bl* ${MY_TMP_DIR}
#cp -r /media/data/jlinders/data_${ROI}/* ${MY_TMP_DIR}

# run python script
python /media/data/ebron/cnn-for-ad-classification/main.py ${MY_TMP_DIR} ${SLURM_JOB_ID} ${ROI}

cp /media/data/ebron/output/out_${SLURM_JOB_ID}.log /media/data/ebron/results/${SLURM_JOB_ID}*/
cp /media/data/ebron/output/error_${SLURM_JOB_ID}.log /media/data/ebron/results/${SLURM_JOB_ID}*/

chown -R 1013:1013 /media/data/ebron/output/*${SLURM_JOB_ID}*
chown -R 1013:1013 /media/data/ebron/results/${SLURM_JOB_ID}*/

## unmount BIGR cluster
#fusermount -u /media/data/ebron/bigr_mount

deactivate
