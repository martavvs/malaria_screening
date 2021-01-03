import cv2
import numpy as np
import os

def generator(data_dir, batch_size=32,resize=110):
    """
    Yields the next training batch.
    Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
    """
    while True: # Loop forever so the generator never terminates

        # Get the samples you'll use in this batch

        # Initialise X_train and y_train arrays for this batch
        X_train = np.zeros((batch_size, resize, resize, 3))
        y_train = np.zeros((batch_size, 1))
        counter = 0

        # For each example
        for png in os.listdir(data_dir):
            img_name = '{}/{}'.format(data_dir, png)
            if png[:4]=="Para":
                label = 0
            if png[:4]=="Unin":
                label = 1

            #preprocessing
            img =  cv2.imread(img_name)
            img = cv2.resize(img,(resize,resize))

            X_train[counter, :, :, :] = img[:, :, :]
            y_train[counter, :] = label

            counter += 1
            if counter==batch_size:
                counter = 0
                # The generator-y part: yield the next training batch
                yield X_train, y_train
