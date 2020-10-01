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
#the code for the facenet above isn't being used

detector = mtcnn.MTCNN()
n = 0
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 460)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 259)
while(cam.isOpened()):
    ret, frame = cam.read() 
    frame2= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(frame2)   
    image = image.convert('RGB')  
  
    pixels = np.asarray(image)
   # pixels = tf.convert_to_tensor(pixels, dtype = tf.int64)
    #Next to create the MTCNN face detector class
    tf.function(experimental_relax_shapes=True) 
    results = detector.detect_faces(pixels)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255,255,255)
    thickness = 2
    
    
    x1, y1, width, height = results[0]['box']
    results = None
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    p1 = (x1,y1)
    p2 = (x1,y2)
    p3 = (x2,y2)
    p4 = (x2,y1)
    cv2.line(frame, p1, p2, (0, 0, 255))
    cv2.line(frame, p2, p3, (0, 0, 255))
    cv2.line(frame, p3, p4, (0, 0, 255))
    cv2.line(frame, p4, p1, (0, 0, 255))
    cv2.putText(frame, "massive cunt", p1, font,
                     fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('Livefeed',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()