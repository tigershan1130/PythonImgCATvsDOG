
import cv2
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.models import load_model
from keras.layers import *
from keras.optimizers import *
from random import shuffle
from matplotlib import pyplot
from numpy import load

TRAIN_DIR = 'D:\PythonApplication\ImageRecogCatVsDog\CatvsDog'
TEST_DIR = 'D:\PythonApplication\ImageRecogCatVsDog\CatvsDog'
IMG_SIZE = 122
LR = 1e-3


MODEL_NAME= 'dogsVsCats-{}-{}.model'.format(LR, '2conv-basic')

def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat': return 1.0
    #                             [no cat, very doggo]
    elif word_label == 'dog': return 0.0

def create_train_data():

    photos = []
    labels = []

    path = os.path.join(TRAIN_DIR, 'train')

    fileList = os.listdir(path)
    numFiles = len(fileList)
    currentCount = 0

    for img in fileList:
        try: 
            label =  label_img(img)
            img_data = cv2.imread(os.path.join(path, img))
            img_data = cv2.resize(img_data,(IMG_SIZE,IMG_SIZE))
            photos.append(img_data)
            labels.append(label)
            currentCount += 1
            print('Loading training data... {:.2} % '.format(currentCount/numFiles * 100))
        except Exception as e :
            pass     
    return photos, labels

def process_test_data():
    X = []
    Y = []

    path = os.path.join(TEST_DIR, 'test2')

    fileList = os.listdir(path)
    numFiles = len(fileList)
    currentCount = 0

    print(path)

    for img in fileList:
        wordlabel = img.split('.')
        img_data = cv2.imread(os.path.join(path,img))
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))

        print(img_data)
        print(wordlabel[0])

        X.append(img_data)
        Y.append(wordlabel[0])
        currentCount += 1
        print('Loading test data... ' + str(currentCount/numFiles * 100) + '%')
    return X, Y


def define_model():    
    model = Sequential()

    model.add(Conv2D(32, (3,3), input_shape = (IMG_SIZE, IMG_SIZE, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size= (2,2)))

    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size= (2,2)))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size= (2,2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
 
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

    model.summary()

    return model

########################## running data ######################

print("Loading training data...")
X, Y = create_train_data()
print("Done...loading data")

#for i in range(9):
#    #define subplot
#    pyplot.subplot(330 + 1 + i)
#    #define filename
#    pyplot.imshow(train_data[i][0])
#pyplot.show()
# If you have already created the dataset:
#train_data = np.load('train_data.npy')

#First split the data into two sets, 1000 for training, last 100 for val/test
X_train = X[:-5000]
X_val = X[-100:]

Y_train = Y[:-5000]
Y_val = Y[-100:]

batchSize = 32

nb_train_samples = len(X_train)
nb_validation_samples = len(X_val)


# this is the augmentation configuration we will use for training and validation
train_datagen = ImageDataGenerator(rescale = 1. / 255, shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
val_datagen = ImageDataGenerator(rescale = 1. / 255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

try:
    model = load_model('model_keras.h5')
    model.load_weights('model_weights.h5')
except:

    print("error loading training models")

    model = define_model()

    # prepare generators for training and validation sets
    train_generator = train_datagen.flow(np.array(X_train), Y_train, batch_size = batchSize)
    validation_generator = val_datagen.flow(np.array(X_val), Y_val, batch_size = batchSize)

    histroy = model.fit_generator(train_generator, steps_per_epoch = nb_train_samples//batchSize, epochs = 10, validation_data = validation_generator, validation_steps=nb_validation_samples//batchSize)


    model.save_weights('model_weights.h5')
    model.save('model_keras.h5')


print('Evaluating accuracy')
# evaluate model
validation_generator = val_datagen.flow(np.array(X_val), Y_val, batch_size = batchSize)
acc = model.evaluate_generator(validation_generator, len(np.array(X_val)))

print(str((acc[1] * 100.0)) + '%')

# test ....
X_test, Y_test = process_test_data()

test_datagen = ImageDataGenerator(rescale = 1. / 255)
test_generator = val_datagen.flow(np.array(X_test), batch_size = batchSize)


print("Predict test data")
prediction_probabilities = model.predict_generator(test_generator)


print(prediction_probabilities)

counter = range(0, len(prediction_probabilities))
aType = ""
solution = pd.DataFrame({"id":counter, "label":list(prediction_probabilities), "type":aType})
cols = solution['label']


for i in counter:

    valData = float(str(solution['label'][i]).lstrip('[').rstrip(']'))

    solution['label'][i] = valData
    solution['id'][i] = i + 1

    if valData > 0.5:
        solution['type'][i] = "cat"
    else:
        solution['type'][i] = "dog"
    
solution.to_csv("dogsVScats.csv", index = False)

