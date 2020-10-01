# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 21:00:20 2020

@author: elija
"""
import tensorflow as tf
from keras.models import load_model
import mtcnn
import numpy as np
from PIL import Image
import cv2


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
          gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
#Somehow this fixed my problem hahah

path = r"C:\Users\elija\Documents\PassionProjects\Facial_Recognition\nerface\keras-facenet-master\model\facenet_keras.h5"
model = load_model(path)
#print(model.inputs)
#print(model.outputs)
#print(mtcnn.__version__)
#Managed to load the facenet model

#Before we can perform face recognition, need to detect
#faces
#Use the Multi-Task Cascaded Convolutional Neural Network
#Or mTCNN for face detection - finds and extracts fae

#Use mtcnn library to create a face dectector
#to extract face and for use with the FaceNet detector

filename = "test.jpg"
image = Image.open(filename)
image = image.convert('RGB')
pixels = np.asarray(image)

#Next to create the MTCNN face detector class
detector = mtcnn.MTCNN()
results = detector.detect_faces(pixels)
print(results)

x1, y1, width, height = results[0]['box']
x1, y1 = abs(x1), abs(y1)
x2, y2 = x1 + width, y1 + height
p1 = (x1,y1)
p2 = (x1,y2)
p3 = (x2,y2)
p4 = (x2,y1)

img_show = cv2.imread(filename)
cv2.line(img_show, p1, p2, (0, 0, 255))
cv2.line(img_show, p2, p3, (0, 0, 255))
cv2.line(img_show, p3, p4, (0, 0, 255))
cv2.line(img_show, p4, p1, (0, 0, 255))
cv2.imshow('bbox', img_show)
cv2.waitKey()
'''
face = pixels[y1:y2, x1:x2]
image = Image.fromarray(face)
image = image.resize((160,160))
face_array = np.asarray(image)
image.show()
'''