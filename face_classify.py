# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:54:24 2020

@author: elija
"""

#PERFORM FACE CLASSIFICATION
#always best to normalise the face embedding vectors, due to being compared to each other using a distance metric
#it is normalised to one

from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot

#load faces
data = load('5-faces.npz')
testX_faces = data['arr_2']

#load face embeddings
data = load('5-celebrity-faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
#trainX is new list
print("train X new list after loading")
print(trainX.shape)

#normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
print("test X new list after encoding")

testX = in_encoder.transform(testX)
print(testX.shape)

#print(trainX.shape)

#label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

#fit model - bit thtat actually trains the model
model = SVC(kernel='linear', probability = True)
model.fit(trainX, trainy)

#test model on a ramdom exmaple from the test dataset
selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection] #this is just so that imshow can display it
random_face_emb = testX[selection]
print('random selected face')
print(random_face_emb.shape)

random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])

#prediction for the face
samples = expand_dims(random_face_emb, axis = 0)
print(samples.shape)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)

#get name
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index]*100
predict_names = out_encoder.inverse_transform(yhat_class)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Ground truth: %s' % random_face_name[0])

#plot for fun
pyplot.imshow(random_face_pixels)
title = '%s (%.3f)' % (predict_names[0], class_probability)
pyplot.title(title)
pyplot.show()





