# -*- coding: utf-8 -*-
"""Expert Level.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ok2qhYCSgehbi0pTt9DylZUQu0qRuLQV
"""

import cv2
import matplotlib.pyplot as plt
import glob
from google.colab import drive

"""## 1. Mount google drive & name it /mydrive/"""

drive.mount('/content/gdrive')

"""## 2. Clone and Build Darknet"""

!git clone https://github.com/AlexeyAB/darknet.git

!ls

# Commented out IPython magic to ensure Python compatibility.
# %cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!make

"""## 3. Get & edit YOLOv3 Configuration file
- Make a copy & rename the original yolov3 config file
- Update Config file, classes & filters
"""

# %cd darknet

# Make a copy & rename it to yolov3_custom.cfg
!cp cfg/yolov3.cfg cfg/yolov3_custom.cfg

# Change classes value to your number of objects
classes = 1
filters = (classes + 5) * 3
max_batches = classes * 2000

if max_batches < 6000:
  max_batches = 6000

# Edit classes & filters
!sed -i 's/batch=1/batch=64/' cfg/yolov3_custom.cfg
!sed -i 's/subdivisions=1/subdivisions=16/' cfg/yolov3_custom.cfg
!sed -i 's/max_batches = 500200/max_batches = $max_batches/' cfg/yolov3_custom.cfg
!sed -i '610 s@classes=80@classes=$classes@' cfg/yolov3_custom.cfg
!sed -i '696 s@classes=80@classes=$classes@' cfg/yolov3_custom.cfg
!sed -i '783 s@classes=80@classes=$classes@' cfg/yolov3_custom.cfg
!sed -i '603 s@filters=255@filters=$filters@' cfg/yolov3_custom.cfg
!sed -i '689 s@filters=255@filters=$filters@' cfg/yolov3_custom.cfg
!sed -i '776 s@filters=255@filters=$filters@' cfg/yolov3_custom.cfg

"""## 4. Extract and prepare dataset for training process
- Create a new directory in darknet/data/ directory & extract dataset to the new directory
- Create classes.names & training.data
- Create training.txt file

"""

!mkdir data/obj
!unzip /mydrive/custom_object_detection/Dataset.zip -d data/obj

# Add your object names to this list
custom_objects = ['Choco Pie']

objects = "\n".join(custom_objects)

!echo -e $objects > data/obj.names
!echo -e 'classes = $classes\ntrain = data/train.txt\nvalid = data/test.txt\nnames = data/obj.names\nbackup = /mydrive/custom_object_detection/' > data/obj.data

images_list = glob.glob("data/obj/*[jpg|png|jpeg]")
print(images_list)

# Create train.txt file
file = open("data/train.txt", "w") 
file.write("\n".join(images_list)) 
file.close()

"""## 5. Download YOLOv3 Pretrained weight"""

!wget https://pjreddie.com/media/files/darknet53.conv.74

"""## 6. Start Training!"""

!./darknet detector train data/obj.data cfg/yolov3_custom.cfg darknet53.conv.74 -dont_show

"""## 7. Test predict using trained weights"""

!./darknet detector test data/obj.data cfg/yolov3_custom.cfg /mydrive/custom_object_detection/yolov3_custom_last.weights -thresh 0.25

