from shared_vae_class import shared_vae_class
from model_objects import model_parameters


import keras
from keras.datasets import mnist

import numpy as np

# Loading the MNIST Data
(x_train, _), (_, _) = mnist.load_data()

# Formating the MNIST Data
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

# Making Copies of the Data to creat Inverses
import copy

a_train = copy.copy(x_train)
for i in range(len(a_train)):
    a_train[i] = 1 - a_train[i]


# Define the model parameters: Batch size, Epochs, First layer size,
# second layer size, third layer size, encoded size, input size
model_parameters = model_parameters(128, 15, 512, 256, 128, 64, 784)

# Create the model with the parameters
shared_vae = shared_vae_class(model_parameters)

shared_vae.compile_model()

# Train the model with: left domain, right domain, and noise
shared_vae.train_model(x_train, a_train, .5)
shared_vae.generate(x_train, a_train)
