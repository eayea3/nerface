# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 13:48:19 2020

@author: elija
"""

from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
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


detector = mtcnn.MTCNN()
required_size= (160,160)




def main():
    #load faces
    data = load('5-faces.npz')
    testX_faces = data['arr_2'] #just for the pyplot plot
    
    #load face embeddings
    data = load('5-celebrity-faces-embeddings.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    
    #normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX) #must normalise the input embedded vectors
   #testX = in_encoder.transform(testX)
    
    #label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    #testy = out_encoder.transform(testy)
    
    #fit model - bit thtat actually trains the model
    SVMmodel = SVC(kernel='linear', probability = True)
    SVMmodel.fit(trainX, trainy)
    #Now the model is trained
    
    #facenet model for embedding
    model_path = r"C:\Users\elija\Documents\PassionProjects\Facial_Recognition\nerface\keras-facenet-master\model\facenet_keras.h5"
    facenet_model = load_model(model_path)
    
    
    
    #code to do with the webcam
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
        #results = extract_face(pixels)
        
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height #this is literally just for the bounding box
        face = pixels[y1:y2, x1:x2] #this is the face array of the extracted face
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)        
        print(face_array.shape) #160,160,3

        #face embedding - returns a vector
        embedding = get_embedding(facenet_model, face_array)
        embedding = np.asarray(embedding)
        print(embedding.shape) #128,)

        
        
        embedding_2d = list()
        embedding_2d.append(embedding)
        embedding_2d = np.asarray(embedding_2d)
        print(embedding_2d.shape)

        embedding_2d = in_encoder.transform(embedding_2d)
        print(embedding_2d[0].shape)

        
        #prediction for the face
        samples = expand_dims(embedding_2d[0], axis = 0)
        yhat_class = SVMmodel.predict(samples)
        yhat_prob = SVMmodel.predict_proba(samples)
        
        #get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index]*100
        predict_names = out_encoder.inverse_transform(yhat_class)
        #print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
        
        #Just plotting the bounding box and the label
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.9
        color = (150,0,0)
        thickness = 2
        p1 = (x1,y1)
        p2 = (x1,y2)
        p3 = (x2,y2)
        p4 = (x2,y1)
        cv2.line(frame, p1, p2, (0, 0, 255))
        cv2.line(frame, p2, p3, (0, 0, 255))
        cv2.line(frame, p3, p4, (0, 0, 255))
        cv2.line(frame, p4, p1, (0, 0, 255))
        if class_probability < (70):
                    cv2.putText(frame, ('Predicted: %s' % ("Unknown face")), p1, font,
                         fontScale, color, thickness, cv2.LINE_AA)
        else:
                    cv2.putText(frame, ('Predicted: %s (%.3f)' % (predict_names[0], class_probability)), p1, font,
                         fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Livefeed',frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()


def get_embedding(model, face_pixels):
    #scale pixel values
    face_pixels = face_pixels.astype('float32')
    #standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean)/ std
    #transform face into one sample/ turn it into a 1d vector
    samples = expand_dims(face_pixels, axis = 0)
    yhat = model.predict(samples)
    return yhat[0]




if __name__ == "__main__":
    main()