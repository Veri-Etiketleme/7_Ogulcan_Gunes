"""
CONFIGURATION FILE

This file contains all deep learning settings for Alzheimer's classification
Esther
"""

# task
task = "AD"                                     # AD / MCI
model = "allCNN"                                # allCNN
roi = "GM_WB"                                   # T1_WB / T1m_WB / GM_WB
location = "server"                              # local / server
comments = ""                                   # additional comments
WB = True                                       # if True: apply model to whole brain data
parelsnoer = True                              # if True: applied to parelsnoer data
train_only = False

# pre-training
pre_train = False                               # if True: uses pre-trained model
pre_train_model = "/path/to/model/model.hdf5"   # path to pre-trained model
freeze = False                                  # if True: freezes layers of the network
freeze_until = 7                                # Freeze until this layer

# MCI evaluation (use with MCI_crossval.py) or Parelsnoer evaluation (use with main.py)
test_only = True                                # only evaluation, no training
all_data = True                                # use all data in once for evaluation
pretrain_path = "/media/data/ebron/results/7243_GM_WB_AD_allCNN/" # path to pre-trained AD model
mean_file = "/media/data/ebron/results/7243_GM_WB_AD_allCNN/k0/mean.npy"            # path to mean of AD training data for normalisation 
std_file = "/media/data/ebron/results/7243_GM_WB_AD_allCNN/k0/std.npy"              # path to std of AD training data for normalisation 

# parameters
k_cross_validation = 1 
epochs = 200
batch_size = 4
test_size = 0.1                                   # .. % test set
val_size = 0.1                                  # .. % validation set

# split
train_val_test_seed = 0                         # use a seed for data splitting, set to number for repeatable experiments
shuffle_split = True                            # True: stratified shuffle split, for MCI testing and quick experiments
                                                # False: stratified K fold, for AD training/testing

# augmentation
augmentation = True                             # True: apply data augmentation
class_total = 1000                              # augment to ... images per class
aug_factor = 0.2                                # mix images with factor ...

# params
lr = 0.001                                      # learning rate
rho = 0.7                                       # Standard value for Adam optimizer
epsilon = 1e-8                                  # Standard value for Adam optimizer
decay = 0.0                                     
dropout = 0.2

# callback options
epoch_performance = True                        # True: compute AUC, sens & spec after every epoch
early_stopping = True                           # True: stops when es_patience is reached without improvement
es_patience = 20                     
lr_scheduler = True                             # True: reduces lr with lr_drop after lr_epochs_drop epochs
lr_drop = 0.5
lr_epochs_drop = 10
acc_checkpoint = False                          # Old function for epoch_performance
acc_early_stopping = False
tensorboard = False

# seed
fixed_seed = None                               # set fixed seed to compare runs, initialisation Keras/Tensorflow, not guaranteed to work on GPU cluster
from numpy.random import seed
from tensorflow import set_random_seed
seed(fixed_seed)
set_random_seed(fixed_seed)

# regularization
from keras import regularizers                  # Experiment done on regularisation, mostly set to None.
weight_regularize = None     # regularizers.l2(0.01) / None

#######################################################################################

# directories

if test_only:
    comments = comments + "_notrain"

import datetime
stamp = datetime.datetime.now().isoformat()[:16]

if location == "local":
    output_dir = f"/local/path/to/output/dir/{stamp}_{roi}_{task}_{model}{comments}/"
    data_dir = f"/local/path/to/data/dir/data_{roi}/"
    aug_dir = f"/local/path/to/augmentation/dir/augdata_{roi}/{task}_{model}{comments}/"
    config_file = "/local/path/to/this/config/file/config.py"
    subjectfile = "/local/path/to/labels/AllSubjectsDiagnosis.csv"
    if parelsnoer:
        subjectfile = "/local/path/to/parelsnoer/labels/Parelsnoer_AllSubjectsBaselineDiagnosis.csv"
else:
    all_results_dir = "/media/data/ebron/results/"
    output_dir = f"{all_results_dir}{stamp}_{roi}_{task}_{model}{comments}/"
    data_dir = f"/media/data/ebron/data_ADNI_{roi}/"
    aug_dir = f"{data_dir}/augmented/"
    config_file = "/media/data/ebron/cnn-for-ad-classification/config.py"
    #subjectfile = "/media/data/jlinders/labels/AllSubjectsDiagnosis.csv"
    subjectfile = "/media/data/ebron/labels_new/labels_adni.csv"
    if parelsnoer:
        data_dir = f"/media/data/ebron/data_PSI_{roi}/"
        subjectfile = "/media/data/ebron/labels_new/labels_psi.csv"

# set classes
if task == "AD":
    class0 = "CN"
    class1 = "AD"
elif task == "MCI":
    class0 = "MCI"
    class1 = "MCIc"

# set input shape
import numpy as np
import h5py
if not parelsnoer:
    
    #input_shape = np.load(data_dir + "002_S_0295.npy").shape
    with h5py.File(data_dir + "002_S_0295_bl.h5py", "r") as hf:
        input_shape  = np.squeeze(hf["002_S_0295_bl"][:]).shape
    
else: 
        #input_shape = np.load(data_dir + "PSI_00752.npy").shape
    with h5py.File(data_dir + "PSI_00752.h5py", "r") as hf:
        input_shape  = np.squeeze(hf["PSI_00752"][:]).shape
fpr_interval = np.linspace(0, 1, 100)           # Range of false positive rates used for generating ROC curve.

