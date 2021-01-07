import cv2
import numpy as np
import os
from tensorflow import keras

from generator import generator

def fit_generator(model, generator_train, generator_val, EPOCHS, nb_batches_train, nb_batches_val):
    for e in range(EPOCHS):
        print('Epoch', e)
        batches = 0
        acc_train = 0
        for x_batch, y_batch in generator_train:
            history = model.fit(x_batch, y_batch, verbose=0)
            acc_train = acc_train + history.history['accuracy'][0]
            batches += 1
            if batches >= nb_batches_train:
                break
        print("---------------------------------------------------------")
        print('Epoch number:', e, 'Accuracy for training set:', acc_train/nb_batches_train)

        batches_val = 0
        acc_val = 0
        for x_batch, y_batch in generator_val:
            history = model.fit(x_batch, y_batch, verbose=0)
            acc_val = acc_val + history.history['accuracy'][0]
            batches_val += 1
            if batches_val >= nb_batches_val:
                break
        print("---------------------------------------------------------")
        print('Epoch number:', e, 'Accuracy for validation set:', acc_val/nb_batches_val)
