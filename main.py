import os
import cv2
import imutils
import numpy as np
from random import shuffle
import h5py
from pathlib import Path
import tensorflow as tf
import math

imageSize = 64
roman_index = []
folders_list = ['Flipped', 'Rotated']
for folder in folders_list:
    if not os.path.exists(folder):
        os.makedirs(folder)
        
def flip():
    global roman_index
    files = [file for file in os.listdir("Images") if file.endswith(".png") or file.endswith(".jpg")]
    
    for i, file in enumerate(files):
        print("Flipping file {}/{}".format(i,len(files)))

        img = cv2.imread("Images/"+file,-1)
        rflip = cv2.flip(img,1)
        vflip = cv2.flip(img,0)
        rvflip = cv2.flip(rflip,0)

        cv2.imwrite("Flipped/"+file[:-4]+"_.png",img)
        cv2.imwrite("Flipped/"+file[:-4]+"_r.png",rflip)
        cv2.imwrite("Flipped/"+file[:-4]+"_v.png",vflip)
        cv2.imwrite("Flipped/"+file[:-4]+"_rv.png",rvflip)
# Generates 2 rotations of the images
def rotate():
    global roman_index
    files = [file for file in os.listdir("Flipped") if file.endswith(".png") or file.endswith(".jpg")]

    # 2 rotations * 4 flips = 8 possibilites
    for index, file in enumerate(files):
        print("Rotating file {}/{}".format(index,len(files)))  

        img = cv2.imread("Flipped/"+file,-1)
        cv2.imwrite("Rotated/"+file[:-4]+"_0.png",img)
        img = np.rot90(img)
        cv2.imwrite("Rotated/"+file[:-4]+"_90.png",img)

# Save h5 dataset
def save():
    global roman_index
    #Load, normalize and reshape
    files = os.listdir("Rotated/")
    files = [file for file in files if file.endswith(".png")]

    shuffle(files)
    X = []
    for i, file in enumerate(files):
        print("Adding file {}/{}".format(i,len(files)))
        
        if(file.startswith('class')):
            roman_index += [i]
            
        img = cv2.imread("Rotated/"+file, -1)

        # Convert grayscale to RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Resize to desired size and normalize
        img = cv2.resize(img,(imageSize,imageSize))
        img = img.astype(float)/255.
        X.append(img)

    # Convert to numpy array
    X = np.array(X)

    # 40% for testing
    testProportion = 0.4
    trainNb = int(X.shape[0]*(1-testProportion))
    testNb = X.shape[0] - trainNb

    print("Splitting dataset in X_train({}) and X_test({})...".format(trainNb,testNb))
    # Split train test
    X_train = X[:trainNb]
    X_test = X[-testNb:]

    #Save at hdf5 format
    print("Saving dataset in hdf5 format... - this may take a while")
    h5f = h5py.File("coins.h5", 'w')
    h5f.create_dataset('X_train', data=X_train)
    h5f.create_dataset('X_test', data=X_test)
    h5f.create_dataset('roman_coins', data=roman_index)
    h5f.close()

file = Path("coins.h5")

if not file.is_file():
    print("This script creates 21208k images from the base 2652")
    print ("Make sure the folders Rotated and Flipped exist")
    flip()
    rotate()
    save()
    
X_train = h5py.File('coins.h5','r')['X_train']
X_test = h5py.File('coins.h5','r')['X_test']
coins_index = h5py.File('coins.h5','r')['roman_coins']   

roman_coins = []
for i in coins_index:
    roman_coins += [i]
y_train = []
y_test = []
test_limit = len(X_train)
for i, data in enumerate(X_train):
    if i in roman_coins:
        y_train += [1]
    else:
        y_train += [0]

for i, data in enumerate(X_test):
    if i + test_limit in roman_coins:
        y_test += [1]
    else:
        y_test += [0]

y_train = np.array(y_train)
y_test = np.array(y_test)

x_train = np.ndarray(shape=X_train.shape, dtype=float)
x_test = np.ndarray(shape=X_test.shape, dtype=float)
for i, data in enumerate(X_train):
    x_train[i] = data
for i, data in enumerate(X_test):
    x_test[i] = data

# We define the characteristics of the input image 
height = 64
width = 64
channels = 3
n_inputs = height * width * 3
ntrain = len(X_train)
ntest  = len(X_test)

# We define the parameters of the layers according to 
# description previously presented

conv1_fmaps = 16       #32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 24     #64
conv2_ksize = 3
conv2_stride = 2
conv2_pad = "SAME"

pool3_fmaps = conv2_fmaps

n_fc1 = 64
n_outputs = 2


with tf.name_scope("inputs"):
    # Variable X is passed as a vector
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    # It is reshaped to the tensor according to image size an channels
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    # Class of each MNIST image
    y = tf.placeholder(tf.int32, shape=[None], name="y")

# The first layer is defined. Notice that the tensorflow function used
# is tf.layers.conv2d(). Also the parameters are those previously defined.


conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")

# The second layer is defined. Notice that the input of this layer is the output
# of the previous layer. You can check that conv1 has size 28x28 and conv2 has
# size 14x14 (Try to find out why)

conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")

# The maxpool layer is defined. Notice that the  tf.nn.max_pool() is used to define the layer.
# Also, maxpool is applied to each of the conv2_fmaps filters, and since the inputs have size
# 14x14, Stride=2 and Padding=Valid, after applying maxpool we have pool3_fmaps filters
# of size (7x7). That is the reason while the output is reshaped (flattened) to (pool3_fmaps * 7 * 7)

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 16 * 16])

# This is the full layer. From previous classes we already know function tf.layers.dense()
# used to define full layers. 
# The number of input and output units is the same (pool3_fmaps * 7 * 7)
with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")

# This is the output layer where the network produces a classification for each class
# The classification is used using the function softmax that we have studied in the previous lab
with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

# The loss function and optimizers are defined as in previous labs.    
with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

# We define two functions to evaluate the quality of the network as classifier
# Correct computes, for a batch of observations, how many were correctly classified.
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# We define a saver 
with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

n_epochs = 1 #10
IMAGE_SIZE = 64
NUM_CHANNELS = 3
BATCH_SIZE = 100
test_size = ntest

config = tf.ConfigProto(device_count = {'GPU': 0})
with  tf.Session(config=config) as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(ntrain // BATCH_SIZE):
            randidx = np.random.randint(ntrain, size=BATCH_SIZE)
            X_batch = x_train[randidx, :]
            y_batch = y_train[randidx]  
            X_batch = np.reshape(X_batch, (-1, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})


        correct_pred = 0
        for iteration in range(ntest // BATCH_SIZE):
            randidx = np.random.randint(ntest, size=BATCH_SIZE)
            X_batch = x_test[randidx, :]
            y_batch = y_test[randidx]  
            X_batch = np.reshape(X_batch, (-1, n_inputs))
#             correct_pred += np.sum(correct.eval(feed_dict={X: X_batch, y: y_batch}))
#             print(correct_pred / float(test_size))
        acc_test = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        print(epoch + 1, "Train accuracy:", acc_train, "Test accuracy:", acc_test)