import datetime
import os
import sys
from shutil import copyfile

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.engine.saving import load_model
from skimage.transform import resize
from tensorflow.python.framework import ops
import h5py

# import configuration parameters
from vis_config import roi, task, label, classification_type, data_set, val, gc_layer, gb_layer, class_limit, server, temp, analysis


def main():
    """
    The current script implements a gradCAM analysis of an AD classification model.
    In the configuration file 'vis_config.py' the settings can be specified.

    This script computes the Grad-CAM mean and variation of a batch of images. Since in most
    cases memory is restricted to calculating the Grad-CAM of only 10 subjects in one run, this script
    is designed to be run several times after each other until a Grad-CAM of all subjects is calculated.
    For this reason this script uses an input argument which defines the run, and can based on this select
    the 10 subjects which belong in that run.

    After running this script the Grad-CAMs created in each run can be calculated using the script
    'vis_average.py'. Here a final nifti version of both the mean and variation will be created.

    The gradcam computations are adapted from the code on:
    https://github.com/eclique/keras-gradcam/blob/master/grad_cam.py
    https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py

    Implementation is based on the paper of Selvaraju et al. (2017):
    https://arxiv.org/pdf/1610.02391.pdf
    """

    # set info paths
    if server:
        info_path = f"/media/data/ebron/saliency/{roi}/{task}/info/"
    else:
        info_path = f"/local/path/to/gradcam/info/{roi}/{task}/info/"

    # load subject file
    all_subjects = np.load(f"{info_path}{classification_type}_classified_subjects_{label}_{data_set}.npy")
    k_splits = all_subjects.shape[0]

    # set paths (when running on server based on data dir + job nr input)
    if server:
        # run indicates which split to take
        run = int(sys.argv[3])
        data_path = sys.argv[1] + "/" if temp else f"/media/data/ebron/data_ADNI_{roi}/"
        save_path = f"/media/data/ebron/saliency/{roi}/{task}/{sys.argv[2]}_n{k_splits*class_limit}_{label}_c{gc_layer}_{classification_type}/"
        run_path = f"{save_path}{sys.argv[2]}_run{run}_{roi}_{data_set}_n{class_limit}_gc{gc_layer}_gb{gb_layer}_{label}_guidedbackprop/"
    else:
        run = 0
        data_path = f"/local/path/to/data/data_{roi}/"
        stamp = datetime.datetime.now().isoformat()[:16]
        save_path = f"/home/jara/gradcam/{roi}/{task}/{stamp}_n{class_limit*k_splits}_{label}_c{gc_layer}_{classification_type}/"
        run_path = f"{save_path}{roi}_{data_set}_n{class_limit}_gc{gc_layer}_gb{gb_layer}_{label}_GradCAM/"

    create_data_directory(run_path)
    if run == 0:
        if server:
            copyfile("/media/data/ebron/cnn-for-ad-classification/visualization/vis_config.py", f"{save_path}configuration.py")
        else:
            copyfile("/local/path/to/config/file/vis_config.py", f"{save_path}configuration.py")

    # set model information files
    mean_file = info_path + "mean.npy"
    std_file = info_path + "std.npy"
    model_file = info_path + "model.hdf5"

    # load mean + std + model
    mean = np.load(mean_file)
    std = np.load(std_file)
    model = load_model(model_file)
    guided_model = build_guided_model(model_file)

    # select subjects for this run
    subjects = all_subjects[run]

    # set class0
    class0 = "CN" if task == "AD" else "MCI-s"

    # get gradcam + gb visualizations
    run_visualization(subjects, data_path, run_path, mean, std, model, guided_model, class0, run)

    print('\nend')

def build_model(model_path):
    """Function returning keras model instance."""
    return load_model(model_path)


def build_guided_model(model_path):
    """Function returning modified model.

    Changes gradient function for all ReLu activations
    according to Guided Backpropagation.
    """
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)

    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = build_model(model_path)
    return new_model

def get_layer(model, vis_layer):
    """
    Returns the name of the conv3D layer to visualize
    which should be specified in the config file with an integer
    """
    conv_cnt = 0
    for layer in model.layers:
        if layer.name[:6] == 'conv3d':
            conv_cnt += 1
            if conv_cnt == vis_layer:
                layer_name = layer.name
    return layer_name



def run_visualization(subjects, data_path, run_path, mean, std, model, guided_model, class0, run):
    """
    Performs a gradcam analysis of multiple subjects of 2 different classes.
    Calculates the mean gradcam, mean guided backprop and mean guided gradcam image.
    Also a statistical test is performed.
    Everything is saved as nifti files.
    """

    X, Y, Z = model.input_shape[1], model.input_shape[2], model.input_shape[3]
    gc_layer_name = get_layer(model, gc_layer)
    gb_layer_name = get_layer(guided_model, gb_layer)

    file = open(f"{run_path}subjects_run{run}.txt", 'w')
    file.write(f"\nRUN {run} - Analysis of {class_limit} {label} subjects\n")
    print(f"\nRUN {run} - Analysis of {class_limit} {classification_type} classified {label} subjects\n")

    images = np.zeros((len(subjects), X, Y, Z))

    for i, subject in enumerate(subjects):

        img_file = f"{data_path}{subject}.h5py"
        print(f"\n{i} - Working on subject: {subject} - with true label: {label}")

        # load + standardize image
        image = load_image(img_file, mean, std, subject)
        images[i] = image

    cls = 0 if label == class0 else 1
    classes = [cls] * len(subjects)

    # expand to network input shape
    images = np.expand_dims(images, axis=5)

    # calculate and save mean gradcam and guided backpropagation
    if analysis == "gradcam" or analysis == "all":
        gradcam_mean, gradcam_var = grad_cam_batch(model, images, classes, gc_layer_name, X, Y, Z)

        # if run contains less subjects, these gradcams should count less in the total average
        if len(subjects) is not class_limit:
            weight_factor = len(subjects) / class_limit
            gradcam_mean = gradcam_mean * weight_factor
            gradcam_var = gradcam_var * weight_factor

        save_npy(gradcam_mean, run_path, gc_layer, "gradcam")
        save_npy(gradcam_var, run_path, gc_layer, "gradcam-VAR")

    if analysis == "guided-backprop" or analysis == "all":
        gb_mean = guided_backprop(guided_model, images, gb_layer_name)

        clip_gb_mean = clip_image(gb_mean)
        save_npy(clip_gb_mean, run_path, gb_layer, "guided-backprop")

    file.close()

def save_npy(image, save_path, vis_layer, title):

    np.save(f"{save_path}{title}_c{vis_layer}_{label}.npy", image)

def normalize(x):
    """utility function to normalize a tensor.
    # Arguments
        x: An input tensor.
    # Returns
        The normalized input tensor.
    """
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())
    
def load_image(img_file, mean, std, subject):
    """Load and normalize image"""

    with h5py.File(img_file, 'r') as hf:
        x = hf[subject][:]
    x = np.subtract(x, mean)
    x = np.divide(x, (std + 1e-10))

    return x

def clip_image(x):
    """
    Normalize image and clip values between 0 and 1
    """
    x -= x.mean()
    x /= (x.std() + 1e-10)
    x = np.clip(x, 0, 1)

    return x

def grad_cam_batch(model, images, classes, layer_name, X, Y, Z):
    """
    GradCAM method to process multiple images in one run.

        INPUT:
            model - the model for which the gradcams should be computed
            images - the batch of images for which the gradcams should be computed
            classes - an array indicating the classes corresponding to the image batch
            X, Y, Z - the input shapes of the images

        OUTPUT
            cam_mean - the average gradcam of the image batch
            cam_var - the variation of the mean gradcam of the image batch
    """

    # get loss, output and gradients
    loss = tf.gather_nd(model.output, np.dstack([range(images.shape[0]), classes])[0])
    layer_output = model.get_layer(layer_name).output
    grads = K.gradients(loss, layer_output)[0]

    # create gradient function
    gradient_fn = K.function([model.input, K.learning_phase()], [layer_output, grads])

    # calculate class activation maps for image batch
    conv_output, grads_val = gradient_fn([images, 0])
    weights = np.mean(grads_val, axis=(1, 2, 3))
    cams = np.einsum('ijklm,im->ijkl', conv_output, weights)

    # process CAMs
    new_cams = np.empty((images.shape[0], X, Y, Z))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i) + 1e-10)
        new_cams[i] = resize(cam_i, (X, Y, Z), order=1, mode='constant', cval=0, anti_aliasing=False)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()

    # calculate mean and variation
    cam_mean = np.mean(new_cams, axis=0)
    cam_var = np.var(new_cams, axis=0)

    return cam_mean, cam_var

def guided_backprop(guided_model, images, layer_name):
    """
    Computes guided backpropagation for 1 or multiple images.
    """
    # get input and output
    model_input = guided_model.input
    layer_output = guided_model.get_layer(layer_name).output

    # calculate and normalize gradients
    grads = K.gradients(layer_output, model_input)[0]
    grads = normalize(grads)

    # create backprop function
    backprop_fn = K.function([model_input, K.learning_phase()], [grads])

    # apply backprop to images and calculate mean
    gb = backprop_fn([images, 0])[0]
    gb_mean = np.mean(gb[:, :, :, :, 0], axis=0)

    return gb_mean

def create_data_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    main()
