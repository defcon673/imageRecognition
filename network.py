# Create first network with Keras
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.optimizers import SGD
import keras
import numpy

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

weights_path = "sources/weights.npy"

def recognizeImage(dataset = 'sources/dataset.npy'):
# fix random seed for reproducibility
    seed = 1488
    numpy.random.seed(seed)

    # split into input (X) and output (Y) variables
    num_classes = 2 #number of labels

    # create model
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(32, 32, 3)))
    model.add(Convolution2D(4, 3, 3, activation='relu', name='conv0_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(4, 3, 3, activation='relu', name='conv0_2'))
    model.add(MaxPooling2D((2, 2), dim_ordering="th"))

    model.add(Convolution2D(8, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(8, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), dim_ordering="th"))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(16, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(16, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), dim_ordering="th"))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), dim_ordering="th"))

    model.add(Flatten(name="flatten"))
    model.add(Dense(256, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', name='dense_2'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, name='dense_3'))
    model.add(Activation("softmax", name="softmax"))
    # Compile model
    model.load_weights(weights_path)

    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())

    # Fit the model
    #model.fit(X, labels, nb_epoch=30, batch_size=32,  verbose=1)
    # calculate predictions
    predictions = model.predict(dataset)
    #results in predictions[0]
    result = [x[0] for x in predictions]
    return result
