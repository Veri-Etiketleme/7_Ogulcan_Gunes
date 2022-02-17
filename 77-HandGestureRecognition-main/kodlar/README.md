# Building a HandGestureRecognition Neural Network Inference Model with Raspberry PI. 

Real-time recognition of dynamic hand gestures from video streams is a challenging task since (i) there is no indication when a gesture starts and ends in the video, (ii) performed gestures should only be recognized once, and (iii) the entire architecture should be designed considering the memory and power budget. In this work, we address these challenges by proposing a hierarchical structure enabling offline-working convolutional neural network (CNN) architectures to operate online efficiently by using sliding window approach. The proposed architecture consists of two models: (1) A detector which is a lightweight CNN architecture to detect gestures and (2) a classifier which is a deep CNN to classify the detected gestures. In order to evaluate the single-time activations of the detected gestures, we propose to use the Levenshtein distance as an evaluation metric since it can measure misclassifications, multiple detections, and missing detections at the same time. We evaluate our architecture on two publicly available datasets - EgoGesture and NVIDIA Dynamic Hand Gesture Datasets - which require temporal detection and classification of the performed hand gestures. ResNeXt-101 model, which is used as a classifier, achieves the state-of-the-art offline classification accuracy of 94.04% and 83.82% for depth modality on EgoGesture and NVIDIA benchmarks, respectively. In real-time detection and classification, we obtain considerable early detections while achieving performances close to offline operation. The codes and pretrained models used in this work are publicly available.

<img src="https://github.com/Timothy102/HandGestureRecognition/blob/main/palm.png" alt="drawing" width="400"/>


## Workflow

### Requirements

* TF Lite Interpreter
* A Virtual-Env or the conda enviroment is recommended. Use 


```bash
conda create -n myenv python
conda install -n myenv scipy 

```


or 

```python3
python3 -m pip install --upgrade pip
pip3 install --upgrade virtualenv
virtualenv -p python3 << name >>
``` 

Download the TensorFlow Lite Interpreter from the website. 

```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install python3-tflite-runtime
``` 

The inference part requires the TensorFlow Lite Interpreter alone. The neural network is already compressed in <tflite> format. Here's the Python code to initialize the interpreter.
  
  
``` python3
import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter(model_path = model_path)  
``` 
 
 ## Model 
  
<img src="https://github.com/Timothy102/HandGestureRecognition/blob/main/model.png" alt="drawing" width="400"/>

  
## Inference
  
Simply run the Makefile in your terminal.
  
```rb
make build  
```
 
It is comprised of taking the picture using raspistill and recieving the same image into a python script. 
Once done, the model will print the desired category. Happy codding! :)
  
 
  
## Hardware
  
A Raspberry Pi is a credit card-sized computer originally designed for education, inspired by the 1981 BBC Micro. Creator Eben Upton's goal was to create a low-cost device that would improve programming skills and hardware understanding at the pre-university level. But thanks to its small size and accessible price, it was quickly adopted by students and electronics enthusiasts for projects that require more than a basic microcontroller (such as Arduino devices). It is slower than a modern laptop or desktop computer but is still a complete Linux computer and can provide all the expected abilities that implies, at a low-power consumption level.

The Raspberry Pi is open hardware, with the exception of the primary chip on the Raspberry Pi, the Broadcomm SoC (System on a Chip), which runs many of the main components of the board–CPU, graphics, memory, the USB controller, etc.

The Raspberry Pi model used for this project is a Raspberry Pi 3 Model B which is the third generation Raspberry Pi. This model has a Quad-Core 64bit CPU, 1GB RAM, 4 x USB ports, 4 pole Stereo output and Composite video port, HDMI, Ethernet port, CSI Camera port, DSI display port, Micro SD port, Wifi and Bluetooth.
  
### Camera Module V2
The v2 Camera Module has a Sony IMX219 8-megapixel sensor. It supports 1080p30, 720p60 and VGA90 video modes, as well as still capture. The camera works with all models of Raspberry Pi 1, 2, and 3. It can be accessed through the MMAL and V4L APIs, and there are numerous third-party libraries built for it, including the Picamera Python library which is used in this project. It attaches via a 15cm ribbon cable to the CSI port on the Raspberry Pi.
  
## Architecture
  
### Python
  
 All of the code is written in the Python programming language with Tensorflow being the framework. 
  
### OpenCV
  
  OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library. OpenCV was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in the commercial products.

The library has more than 2500 optimized algorithms, which includes a comprehensive set of both classic and state-of-the-art computer vision and machine learning algorithms. These algorithms can be used to detect and recognize faces, identify objects, classify human actions in videos, track camera movements, track moving objects, extract 3D models of objects, produce 3D point clouds from stereo cameras, stitch images together to produce a high resolution image of an entire scene, find similar images from an image database, remove red eyes from images taken using flash, follow eye movements, recognize scenery and establish markers to overlay it with augmented reality, etc. OpenCV has more than 47 thousand people of user community and estimated number of downloads exceeding 14 million. The library is used extensively in companies, research groups and by governmental bodies.

It has C++, Python, Java and MATLAB interfaces and supports Windows, Linux, Android and Mac OS.
  
# Conclusion & Recommendations
  
  In this project, we have taken a raspberry pi and used it to run our gesture based flask web app. We have learned and gained a lot working with a Raspberry Pi and OpenCV. It was something different and challanging. It was interesting to work on the gesture recognition and to dive into machine learning. We did encounter some problems along the way but with help from stack overflow and other forums, we managed our way through to a working application. Majoity of the problems were related to the installation of openCV on the Raspberry Pi. Since the Raspberry Pi is such a small and lightweight computer and OpenCV is a vast and heavy use library, it wasn't an easy process. Following the OpenCV tutorial mentioned above, we had to change around with the settings on the Pi and be patient with the compile process. The next few problems encountered after that was to do with the merging of OpenCV and the Pi Camera. OpenCV is mostly used with usb cameras and laptop web cams and therefore we had to adjust the code around in order for it to work with the Pi Camera module. After that the application ran successfully. It took a lot of work and time but in the end, it was very rewarding to get everything working.
  
  
 # Contact
  
 Feel free to contact me with any proposals on LinkedIn: linkedin.com/in/tim-cvetko-32842a1a6/ or Mail: cvetko.tim@gmail.com :D
  
  
