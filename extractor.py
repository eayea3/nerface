from os import listdir
import tensorflow as tf
from os.path import isdir
import mtcnn 
import numpy as np
from numpy import savez_compressed
from PIL import Image
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
          gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
detector  = mtcnn.MTCNN() #outside cus just makes more sense
required_size= (160,160)

def extract_face(filename):
    #function that literally extracts the face
    #from the photos
    image = Image.open(filename)
    image = image.convert('RGB')
    #must convert to array
    pixels = np.asarray(image)
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    
    #some sort of bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

def load_faces(directory):
    #makes a list for the faces to be stored into
    faces = list()
    for filename in listdir(directory):
        path = directory + filename
        face = extract_face(path)
        faces.append(face)
    return faces

def load_dataset(directory):
    X, y = list(), list()
    for subdir in listdir(directory):
        #for class name in the directory
        path = directory + '/' + subdir + '/'
        #Just a thing to see if everything works 
        #the wy it should
        #path_save = directory + '_imagecopy' + '/' + subdir + '/'
        print(path)
        if not isdir(path):
            continue
        faces = load_faces(path)
        #for counter, face in enumerate(faces):
        #    img = Image.fromarray(face, 'RGB')
        #    img.save(path_save + str(counter) + '.png')
        labels = [subdir for _ in range(len(faces))]
        print('>loaded %d examples for class: %s'%(len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
    #y has the labels
        
    return np.asarray(X), np.asarray(y)

train_path = r"C:\Users\elija\Documents\PassionProjects\Facial_Recognition\nerface\keras-facenet-master\train"
val_path = r"C:\Users\elija\Documents\PassionProjects\Facial_Recognition\nerface\keras-facenet-master\val"
#ath_save =  r"C:\Users\elija\Documents\PassionProjects\Facial_Recognition\nerface\keras-facenet-master\train_imagecopy"

#testX,
def main():
    trainX, trainy = load_dataset(train_path)
    print(trainX.shape, trainy.shape)
    testX, testy = load_dataset(val_path)
    print(testX, testy)
    savez_compressed('5-faces.npz', trainX, trainy, testX, testy)
    
    
if __name__ == "__main__":
    main()