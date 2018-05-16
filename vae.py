from __future__ import print_function
import numpy as np
from numpy import linalg as LA


from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Lambda, Input
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K
from keras import objectives
from keras.callbacks import EarlyStopping, Callback
from keras.layers import LeakyReLU
from keras import regularizers

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

import os

#import pylab

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 32))
    return z_mean + K.exp(.5*z_log_sigma) * epsilon

def vae_loss(inputs, finalLayer): #Loss function comprised of two parts, Cross_entropy, and divergense
    '''cross_entropy = objectives.binary_crossentropy(inputs,finalLayer)
    kl_divergence = -.5*K.mean(1+ z_log_sigma - K.square(z_mean) 
        - K.exp(z_log_sigma), axis=-1)
    return cross_entropy + kl_divergence
    '''

    reconstruction_loss = K.sum(K.square(finalLayer-inputs))
    kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.square(K.exp(z_log_sigma)), axis=-1)
    total_loss = K.mean(reconstruction_loss + kl_loss)    
    return total_loss
    

(x_train, _), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


import copy

for i in range(len(x_train)):
    x_train[i] = 1 - x_train[i]

for i in range(len(x_test)):
    x_test[i] = 1 - x_test[i]

for i in range(len(x_train_noisy)):
    x_train_noisy[i] = 1 - x_train_noisy[i]

for i in range(len(x_test_noisy)):
    x_test_noisy[i] = 1 - x_test_noisy[i]


#Define Encoder
encoderInputs = Input(shape=(784,)) #Formating the input layer

encoderSecondLayer = Dense(512, activation = 'relu')(encoderInputs)

encoded = Dense(128, activation = 'relu')(encoderSecondLayer)
z_mean = Dense(32)(encoded)
z_log_sigma = Dense(32)(encoded)

z = Lambda(sampling)([z_mean, z_log_sigma])


encoder = Model(encoderInputs, [z_mean, z_log_sigma,z])

#Decoder
decoderInputs = Input(shape=(32,))
decoderZeroLayer = Dense(128, activation = 'relu')(decoderInputs)

decoderSecondLayer = Dense(512, activation = 'relu')(decoderZeroLayer)

decoded = Dense(784, activation = 'sigmoid')(decoderSecondLayer)

decoder = Model(decoderInputs, decoded)

#Full Model
outputs = decoder(encoder(encoderInputs)[2]) 
vae_model = Model(encoderInputs, outputs) #Create the full model
vae_model.compile(optimizer='adam', loss=vae_loss) #Compile

callback = EarlyStopping(monitor='loss', 
patience=3, verbose=0, mode='auto')


vae_model.fit(x_train_noisy, x_train,
                epochs=50,
                batch_size= 96,
                shuffle=True,
                validation_data=(x_test_noisy, x_test),
                callbacks = [callback])

encoded_imgs = encoder.predict(x_test)[2]
decoded_imgs = decoder.predict(encoded_imgs)

import matplotlib.pyplot as plt




predicted = encoder.predict(x_train, 
            batch_size=96)[2] #Predict the outputs
        
max_coords = np.amax(predicted, axis = 0)
min_coords = np.amin(predicted, axis = 0)

rng = np.random.normal(0, 1, (10, 32)) 

generatedImgs = decoder.predict(rng)


x_test_encoded = encoder.predict(x_test, batch_size=96)[2]
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

n = 10  # how many digits we will display
plt.figure(figsize=(30, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2* n)
    plt.imshow(generatedImgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


