from shared_vae_class import shared_vae_class
from model_objects import model_parameters
import keras
import numpy as np
import ICVL
import MNIST
import os

from unpack_files import unpackFiles

# MNIST 28x28
# ICVL 60x80

dataSetInfo = MNIST.dataInfo()

if not os.path.exists(os.path.join('Data', 'Training', '{}_Training.pkl'.format(dataSetInfo.name))):
    unpackFiles(dataSetInfo.name)

(x_train, a_train, x_test, a_test) = dataSetInfo.load()


if not os.path.exists("Output/" + dataSetInfo.name):
    os.mkdir("Output/" + dataSetInfo.name)
# second layer size, third layer size, encoded size, input size
model_parameters = model_parameters(
    256, 1, 512, x_train.shape[1], 64, 48, 34, 512, a_train.shape[1], dataSetInfo)

# Create the model with the parameters
shared_vae = shared_vae_class(model_parameters)

shared_vae.compile_model()

# Train the model with: left domain, right domain, and noise
shared_vae.train_model(x_train, a_train, .5)
shared_vae.generate(x_train, a_train)
