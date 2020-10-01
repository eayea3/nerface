# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:12:01 2020

@author: elija
"""

from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model

#get the face embedding - turn the array pictures
#into the correct format basically
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



def main():
    data = load('5-faces.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
    
    #now load the facnet model
    model_path = r"C:\Users\elija\Documents\PassionProjects\Facial_Recognition\nerface\keras-facenet-master\model\facenet_keras.h5"

    model = load_model(model_path)
    
    #convert each face in the train set to an embedding
    newTrainX = list()
    #how to initialise a list
    for face_pixels in trainX:
        print(face_pixels.shape)
        embedding = get_embedding(model, face_pixels)
        print(embedding.shape)
        newTrainX.append(embedding)
    newTrainX = asarray(newTrainX)
    print(newTrainX.shape)
    
    newTestX = list()
    for face_pixels in testX:
        embedding = get_embedding(model, face_pixels)
        newTestX.append(embedding)
    newTestX = asarray(newTestX)
    print(newTestX.shape)
    
    #save the arrays into the correct format
    savez_compressed('5-celebrity-faces-embeddings.npz', newTrainX, trainy, newTestX, testy)
    
    
if __name__ == "__main__":
    main()
    #outputs are:
    # (93, 128)
    #basically 93 inputs/face embeddings 
    #with a 128 element vector
    #likewise 25 validation examples (25, 128)
    