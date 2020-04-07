"""
shared-latent-space/main_file.py

This file is the main file which should be run from command line

Usage: Run in comand line with no arguments:

        python main_file.py

Output:
    Represenation of the model's encoder saved to /Output/<dataset>
        with the parameters of the model in the file name.
    Visualiazations as defined in <dataset>'s corresponding implementation
        of DataSetInfoAbstractClass. These may also be saved to the
        /Output/<dataset> foled depending on the implementation.

Currently, <dataset> can be ICVL or MNIST as included with this repo.

Author: Chris Williams
Date: 5/22/18
"""


import os

import tensorflow.keras as keras
import numpy as np

# Local files
import ICVL
import MNIST

from unpack_files import unpackFiles
from shared_vae_class import shared_vae_class
from model_objects import model_parameters
# MNIST 28x28
# ICVL 60x80

dataSetInfo = MNIST.dataInfo()

# if not os.path.exists(os.path.join('Data', 'Training',
#                                    '{}_Training.pkl'.format(dataSetInfo.
#                                                             name))):
#     unpackFiles(dataSetInfo.name)

(x_train, a_train, x_test, a_test) = dataSetInfo.load()

if not os.path.exists(os.path.join('Output', dataSetInfo.name)):
    os.makedirs(os.path.join('Output', dataSetInfo.name))

print(x_train.shape, a_train.shape)
# second layer size, third layer size, encoded size, input size
model_parameters = model_parameters(
    batchSize=256, numEpochs=1,
    inputSizeLeft=x_train.shape[1],
    firstLayerSizeLeft=96,
    secondLayerSize=64,
    thirdLayerSize=32,
    encodedSize=16,
    inputSizeRight=a_train.shape[0],
    firstLayerSizeRight=1024,
    dataSetInfo=dataSetInfo)


# Create the model with the parameters
shared_vae = shared_vae_class(model_parameters)

shared_vae.compile_model()

# Train the model with: left domain, right domain, and noise
shared_vae.train_model(x_train, a_train, .5)
shared_vae.generate(x_train, a_train)
