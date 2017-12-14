"""
This script augments the dataset,
going from 2600 to 21000 images
This allows the  dataset to be stored 
on GitHub, and easily downloaded.
It also saves it as h5 file
"""

import os
import cv2
import imutils
import numpy as np
from random import shuffle
import h5py
from pathlib import Path


imageSize = 128

def save():
    roman_coins = []
    files = os.listdir("dataset/")
    files = [file for file in files if file.endswith(".png")]
    shuffle(files)
    X = []
    for i, file in enumerate(files):
        if(file.startswith('class')):
            roman_coins += [i]
        print("Adding file {}/{}".format(i, len(files)))
        #img = cv2.imread("Rotated/" + file, -1)
        img = cv2.imread("dataset/" + file, -1)

        # Convert grayscale to RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Resize to desired size and normalize
        img = cv2.resize(img, (imageSize, imageSize))
        img = img.astype(float) / 255.
        X.append(img)

    # Convert to numpy array
    X = np.array(X)

    # 10% for testing
    testProportion = 0.1
    trainNb = int(X.shape[0] * (1 - testProportion))
    testNb = X.shape[0] - trainNb

    print("Splitting dataset in X_train({}) and X_test({})...".format(trainNb, testNb))
    # Split train test
    X_train = X[:trainNb]
    X_test = X[-testNb:]

    print(len(X_train))

    # Save at hdf5 format
    "Saving dataset in hdf5 format... - this may take a while"
    h5f = h5py.File("coins.h5", 'w')
    h5f.create_dataset('X_train', data=X_train)
    h5f.create_dataset('X_test', data=X_test)
    h5f.create_dataset('roman_coins', data=roman_coins)
    h5f.close()

file = Path("coins.h5")
if file.is_file():
    print("Hello")
else:
    save()

