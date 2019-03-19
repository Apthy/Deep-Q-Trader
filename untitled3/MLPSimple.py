from __future__ import print_function
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import adam
from keras.utils import np_utils
import csv
path = "GBPUSDFeatures.csv"
data = pd.read_csv(path)
np.random.seed(1671)

Num_epoch = 200
Batch_size = 128
Verbose = 1
Num_classes = 4
Optimiser = adam
N_hidden = 128
Validation_split = 0.2

(X_train, Y_train), (X_test, Y_test) = data
Reshaped = 784

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalise the data
X_test /= 255
X_train /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_test = np_utils.to_categorical(Y_test, Num_classes)
Y_train = np_utils.to_categorical(Y_train, Num_classes)
# 10 outputs with a final softmax
model = Sequential()
model.add(Dense(N_hidden, input_shape=(Reshaped,)))
model.add(Activation('relu'))
model.add(Dense(N_hidden))
model.add(Activation('relu'))
model.add(Dense(Num_classes))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=Optimiser, metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=Batch_size, epochs=Num_epoch, verbose=Verbose,
                    validation_split=Validation_split)
score = model.evaluate(X_test, Y_test, verbose=Verbose)
print("Test Score:", score[0])
print("test accuracy:", score[1])
