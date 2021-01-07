# import the necessary modules from the library
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D, Dropout

from generator import generator
from fit_generator import fit_generator
import parameters as param

TRAIN_DIR =  param.train_dir
VAL_DIR =  param.val_dir
INPUT = param.input
SEED = param.seed
BATCH_SIZE=param.batch_size
EPOCHS=param.epochs
INPUT_SHAPE = (INPUT,INPUT,3)


generator_train = generator(TRAIN_DIR, BATCH_SIZE, resize=INPUT)
generator_val = generator(VAL_DIR, BATCH_SIZE, resize=INPUT)

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=INPUT_SHAPE),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.5),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.5),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

nb_batches_train = len(os.listdir(TRAIN_DIR))//BATCH_SIZE
nb_batches_val = len(os.listdir(VAL_DIR))//BATCH_SIZE

fit_generator(model, generator_train, generator_val, EPOCHS, nb_batches_train, nb_batches_val)



# history = model.fit_generator(
#         generator_train,
#         steps_per_epoch=len(os.listdir(TRAIN_DIR))// BATCH_SIZE,
#         epochs=EPOCHS,
#         validation_data=generator_val,
#         validation_steps=len(os.listdir(VAL_DIR))// BATCH_SIZE)
#
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
#
# epochs_range = range(EPOCHS)
#
# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
# plt.show()

#
# model = Sequential()
# model.add(Conv2D(32, (3,3),padding='same',input_shape=INPUT_SHAPE,name='conv2d_1'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2),name='maxpool2d_1'))
# model.add(Conv2D(32, (3, 3),name='conv2d_2'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2),name='maxpool2d_2'))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2))
# model.add(Activation('sigmoid'))
