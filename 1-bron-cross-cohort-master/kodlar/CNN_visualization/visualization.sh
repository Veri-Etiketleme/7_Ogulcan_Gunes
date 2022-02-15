#!/bin/bash
#SBATCH --ntasks=6
#SBATCH --mem=11G
#SBATCH --gres=gpu:GTX:1
#SBATCH -p GPU_short
#SBATCH -t 8:00:00
#SBATCH -o /media/data/ebron/output/out_%j.log
#SBATCH -e /media/data/ebron/output/error_%j.log

module load python/3.6.7
module load tensorflow/1.12.0

## mount BIGR cluster
#sshfs -o IdentityFile=/media/data/ebron/.ssh/id_rsa -o StrictHostKeyChecking=no ebron@bigr-app003:/scratch /media/data/ebron/bigr_mount/

# activate virtual environment
source /media/data/ebron/virtualenvs/CNN-for-AD/bin/activate

# copy data to temp job dir
ROI=GM_WB
TASK=AD

MY_TMP_DIR=/slurmtmp/${SLURM_JOB_USER}.${SLURM_JOB_ID}
cp -r /media/data/ebron/bigr_mount/tmichelotti/data_ADNI_${ROI}/*_bl* ${MY_TMP_DIR}
#cp -r /media/data/ebron/bigr_mount/ebron/ADNI/data_ADNI_${ROI}/*_bl* ${MY_TMP_DIR}

python /media/data/ebron/cnn-for-ad-classification/visualization/get_missclassifications.py ${MY_TMP_DIR} ${SLURM_JOB_ID}

# run python scripts
for i in {0..281}
do
	python /media/data/ebron/cnn-for-ad-classification/visualization/vis_main.py ${MY_TMP_DIR} ${SLURM_JOB_ID} $i
done

python /media/data/ebron/cnn-for-ad-classification/visualization/vis_average.py ${MY_TMP_DIR} ${SLURM_JOB_ID}

cp /media/data/ebron/output/out_${SLURM_JOB_ID}.log /media/data/ebron/saliency/${ROI}/${TASK}/${SLURM_JOB_ID}_*/
cp /media/data/ebron/output/error_${SLURM_JOB_ID}.log /media/data/ebron/saliency/${ROI}/${TASK}/${SLURM_JOB_ID}_*/

chown -R 1013:1013 /media/data/ebron/output/*${SLURM_JOB_ID}*
chown -R 1013:1013 /media/data/ebron/saliency/${ROI}/${TASK}/${SLURM_JOB_ID}_*/

## unmount BIGR cluster
#fusermount -u /media/data/ebron/bigr_mount

deactivate
