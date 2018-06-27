"""
shared-latent-space/main_file.py

This file is the main file which should be run from command line. Please note
that currently two of the command line arugments are not used, but I have left
them so that the model can be easily changed.

Usage: Run in comand line with arguments:

        python main_file.py --data Cognoma --batchSize 60 --numEpochs 60
        --firstLayerSizeLeft 1024 --thirdLayerSize 0 --secondLayerSize 0
        --encodedSize 32 --firstLayerSizeRight 1024 --kappa .1 --noise 0.2
        --notes "example note"

Output:
    Represenation of the model's encoders saved to /Output/<dataset>
        with the number of the run in the name, so then every run is saved.
    The model parameters are saves to a file which uses the model run as its
        name, so that the parameters are linked to the number of the run
    Visualiazations as defined in <dataset>'s corresponding implementation
        of DataSetInfoAbstractClass. These may also be saved to the
        /Output/<dataset> folder depending on the implementation.

Currently, <dataset> can be ICVL, MNIST, or Cognoma as included with this repo.

Author: Chris Williams
Date: 6/25/18
"""


import os
import argparse

import keras
import numpy as np

# Local files
import ICVL
import MNIST
import Cognoma

from unpack_files import unpackFiles
from shared_vae_class import shared_vae_class
from model_objects import model_parameters


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="The name of the data file", type=str)
    parser.add_argument("--batchSize", help="Batch Size", type=int,
                        default=128)
    parser.add_argument("--numEpochs", help="Number of training Epochs",
                        type=int, default=60)
    parser.add_argument("--firstLayerSizeLeft", help="Left firstlayer size",
                        type=int)
    parser.add_argument("--secondLayerSize",
                        help="Second layer", type=int)
    parser.add_argument("--thirdLayerSize",
                        help="Third layer", type=int)
    parser.add_argument("--encodedSize",
                        help="Encdoded size", type=int)
    parser.add_argument("--firstLayerSizeRight",  help="Right firstlayer size",
                        type=int)
    parser.add_argument("--kappa",  help="Kappa for warmstart",
                        type=float, default=.1)
    parser.add_argument("--beta",  help="Beta for warmstart",
                        type=float, default=1)
    parser.add_argument("--noise",  help="Noise",
                        type=float, default=0)
    parser.add_argument("--dropout",  help="Dropout",
                        type=float, default=0)
    parser.add_argument("--notes",  help="Notes",
                        type=str, default="N/A")
    args = parser.parse_args()
    return args


args = get_args()

# Dictionary of generator types. The string is the command line argument     !!
data_dict = {
        'MNIST': MNIST.dataInfo(),
        'ICVL': ICVL.dataInfo(),
        'Cognoma': Cognoma.dataInfo()
}


dataSetInfo = data_dict[args.data]

if not os.path.exists(os.path.join('Data',
                                   '{}_Data'.format(dataSetInfo.name),
                                   'Training',
                                   '{}_Training.pkl'.format(dataSetInfo.
                                                            name))):
    unpackFiles(dataSetInfo.name)

(x_train, a_train, x_test, a_test) = dataSetInfo.load()

print("Finished Loading")
print x_train.shape
print a_train.shape

if not os.path.exists(os.path.join('Output', dataSetInfo.name)):
    os.mkdir(os.path.join('Output', dataSetInfo.name))


model_parameters = model_parameters(
    batchSize=args.batchSize, numEpochs=args.numEpochs,
    inputSizeLeft=x_train.shape[1],
    firstLayerSizeLeft=args.firstLayerSizeLeft,
    secondLayerSize=args.secondLayerSize,
    thirdLayerSize=args.thirdLayerSize,
    encodedSize=args.encodedSize,
    inputSizeRight=a_train.shape[1],
    firstLayerSizeRight=args.firstLayerSizeRight,
    kappa=args.kappa,
    beta=args.beta,
    noise=args.noise,
    notes=args.notes,
    dropout=args.dropout,
    dataSetInfo=dataSetInfo)

# Calculate which number run it is, so then the output will have a unique name
while os.path.exists(os.path.join('Output', dataSetInfo.name,
                                  "{}.html".format(model_parameters.outputNum)
                                  )):
    model_parameters.outputNum += 1

# Create the model with the parameters
shared_vae = shared_vae_class(model_parameters)

shared_vae.compile_model()

# Train the model with: left domain, right domain, and noise
shared_vae.train_model(x_train, a_train, x_test, a_test, 0)
shared_vae.generate(x_test, a_test)
