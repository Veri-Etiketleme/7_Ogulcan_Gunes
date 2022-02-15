"""
CONFIGURATION FILE

This file contains all settings for the Grad-CAM computations of the all-CNN trained for MRI AD classification.

"""

roi = "GM_WB"                       # GM_WB / T1_WB / T1m_WB
task = "AD"                         # AD / MCI
label = "AD"                        # AD / CN / MCI-s / MCI-c

classification_type = "correct"     # miss / correct
data_set = "train"                  # train / test
val = False                         # True: if test set for pre-trained model is needed

gc_layer = 3                        # layer to visualize gradcam
gb_layer = 14                       # layer to visualize guided backprops
class_limit = 1                    # amount of subjects to be processed per run
server = True                       # True: if run on server
temp = True                         # if using temporary slurm directory

analysis = "guided-backprop"        # all / gradcam / guided-backprop