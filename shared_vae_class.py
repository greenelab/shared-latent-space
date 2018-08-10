"""
shared-latent-space/shared_vae_class.py

This class contains the model and the associated functions.
it is called by main_file.py. It uses many models to represent
the various parts of the shared latent space VAE and allow for
the training of certain parts in isolation. To train, it switches
between training the left part, and the right parts individually for n epochs.
In the generate() function, the class calls upon the
specific implementation of DataSetInfoAbstractClass for the
given dataset.


Author: Chris Williams
Date: 6/25/18
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly as py
import plotly.graph_objs as go

import keras
import scipy

from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Activation, Lambda, Input, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TerminateOnNaN, Callback
from keras.utils import plot_model
from keras.losses import binary_crossentropy, mse
from tensorflow import set_random_seed
from numpy.random import seed

# Local files
import ICVL
import MNIST

from model_objects import model_parameters, model

# Set the seed
seed(0)
set_random_seed(0)


class shared_vae_class(object):

    def __init__(self, model_parameters):
        """
        Takes in parameters of the model.

        Args:
            model_parameters (model_parameters): Parameters for the model.

        Returns: None
        """
        self.params = model_parameters

        # Variables for warm start
        self.betaRight = K.variable(model_parameters.beta)
        self.betaLeft = K.variable(model_parameters.beta)
        self.kappa = self.params.kappa

    def compile_model(self):
        """
        Compiles the Model.

        Args: None

        Returns: None
        """

        # the Sampling function for the VAE
        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(
                shape=(K.shape(z_mean)[0], self.params.encodedSize))
            return z_mean + K.exp(z_log_sigma) * epsilon

        # Loss function for the VAE
        # Loss function comprised of two parts, Cross_entropy, and
        # divergence
        def vae_loss(inputs, finalLayer):
            reconstruction_loss = K.sum(K.square(finalLayer - inputs))
            kl_loss = - 0.5 * K.sum(1 + z_log_sigmaFull - K.square(
                z_meanFull) - K.square(K.exp(z_log_sigmaFull)), axis=-1)
            total_loss = K.mean(reconstruction_loss + kl_loss)
            return total_loss

        # Loss function for the left VAE
        # Loss function comprised of two parts, Cross_entropy, and
        # divergence
        def left_vae_loss(inputs, finalLayer):
            reconstruction_loss = K.sum(K.square(finalLayer - inputs))
            kl_loss = - 0.5 * K.sum(1 + z_log_sigmaLeft - K.square(
                z_meanLeft) - K.square(K.exp(z_log_sigmaLeft)), axis=-1)
            total_loss = K.mean(reconstruction_loss + kl_loss)
            return total_loss

        # Loss function for the right VAE
        # Loss function comprised of two parts, Cross_entropy, and
        # divergence
        def right_vae_loss(inputs, finalLayer):
            # reconstruction_loss = K.sum(K.square(finalLayer - inputs))
            reconstruction_loss = self.params.inputSizeRight * binary_crossentropy(inputs, finalLayer)
            kl_loss = - 0.5 * K.sum(1 + z_log_sigmaRight - K.square(
                z_meanRight) - K.square(K.exp(z_log_sigmaRight)), axis=-1)
            total_loss = K.mean(reconstruction_loss + kl_loss)
            return total_loss

        # Define the Encoder with the Left and Right branches
        leftEncoderInput = Input(shape=(self.params.inputSizeLeft,))
        leftEncoderFirstLayer = Dense(
            self.params.firstLayerSizeLeft,
            activation='relu')(leftEncoderInput)
        leftEncoderSecondLayer = Dense(
            self.params.secondLayerSize,
            activation='relu')(leftEncoderFirstLayer)

        rightEncoderInput = Input(shape=(self.params.inputSizeRight,))
        rightEncoderFirstLayer = Dense(
            self.params.firstLayerSizeRight,
            activation='relu')(rightEncoderInput)
        rightEncoderSecondLayer = Dense(
            self.params.secondLayerSize,
            activation='relu')(rightEncoderFirstLayer)

        encoderMergeLayer = Dense(
            self.params.thirdLayerSize, activation='relu')
        leftMerge = encoderMergeLayer(leftEncoderSecondLayer)
        rightMerge = encoderMergeLayer(rightEncoderSecondLayer)

        # These different merge branches are used in different models
        mergedLayer = keras.layers.average([leftMerge, rightMerge])
        leftMergedLayer = keras.layers.average([leftMerge, leftMerge])
        rightMergedLayer = keras.layers.average(
            [rightMerge, rightMerge])

        z_mean = Dense(self.params.encodedSize)
        z_log_sigma = Dense(self.params.encodedSize)

        # These three sets are used in differen models
        z_meanLeft = z_mean(leftMergedLayer)
        z_log_sigmaLeft = z_log_sigma(leftMergedLayer)

        z_meanRight = z_mean(rightMergedLayer)
        z_log_sigmaRight = z_log_sigma(rightMergedLayer)

        z_meanFull = z_mean(mergedLayer)
        z_log_sigmaFull = z_log_sigma(mergedLayer)

        zLeft = Lambda(sampling)([z_meanLeft, z_log_sigmaLeft])
        zRight = Lambda(sampling)([z_meanRight, z_log_sigmaRight])
        zFull = Lambda(sampling)([z_meanFull, z_log_sigmaFull])

        # These are the three different models
        self.leftEncoder = Model(leftEncoderInput, zLeft)
        self.rightEncoder = Model(rightEncoderInput, zRight)
        self.fullEncoder = Model(
            [leftEncoderInput, rightEncoderInput], zFull)

        # Defining the Decoder with Left and Right Outputs
        decoderInputs = Input(shape=(self.params.encodedSize,))
        decoderFirstLayer = Dense(
            self.params.thirdLayerSize,
            activation='relu')(decoderInputs)

        leftDecoderSecondLayer = Dense(
            self.params.secondLayerSize,
            activation='relu')(decoderFirstLayer)
        leftDecoderThirdLayer = Dense(
            self.params.firstLayerSizeLeft,
            activation='relu')(leftDecoderSecondLayer)
        leftDecoderOutput = Dense(
            self.params.inputSizeLeft,
            activation='sigmoid')(leftDecoderThirdLayer)

        rightDecoderSecondLayer = Dense(
            self.params.secondLayerSize,
            activation='relu')(decoderFirstLayer)
        rightDecoderThirdLayer = Dense(
            self.params.firstLayerSizeRight,
            activation='relu')(rightDecoderSecondLayer)
        rightDecoderOutput = Dense(
            self.params.inputSizeRight,
            activation='sigmoid')(rightDecoderThirdLayer)

        # Three different Decoders
        self.fullDecoder = Model(
            decoderInputs, [leftDecoderOutput, rightDecoderOutput])
        self.leftDecoder = Model(decoderInputs, leftDecoderOutput)
        self.rightDecoder = Model(decoderInputs, rightDecoderOutput)
        # decoder.summary()

        # Left to Right transition
        outputs = self.fullDecoder(self.leftEncoder(leftEncoderInput))
        self.leftToRightModel = Model(leftEncoderInput, outputs)
        # leftToRightModel.summary()

        # Right to Left transition
        outputs = self.fullDecoder(self.rightEncoder(rightEncoderInput))
        self.rightToLeftModel = Model(rightEncoderInput, outputs)
        # rightToLeftModel.summary()

        # Full Model
        outputs = self.fullDecoder(self.fullEncoder(
            [leftEncoderInput, rightEncoderInput]))
        # Create the full model
        self.vae_model = Model(
            [leftEncoderInput, rightEncoderInput], outputs)
        lowLearnAdam = keras.optimizers.Adam(
            lr=0.0005, beta_1=0.9, beta_2=0.999,
            epsilon=None, decay=0.0, amsgrad=False)
        self.vae_model.compile(optimizer=lowLearnAdam,
                               loss=vae_loss)  # Compile
        # vae_model.summary()

        # Freeze all shared layers
        leftMerge.trainable = False
        rightMerge.trainable = False

        mergedLayer.trainable = False
        leftMergedLayer.trainable = False
        rightMergedLayer.trainable = False

        z_meanLeft.trainable = False
        z_log_sigmaLeft.trainable = False

        z_meanRight.trainable = False
        z_log_sigmaRight.trainable = False

        z_meanFull.trainable = False
        z_log_sigmaFull.trainable = False

        zLeft.trainable = False
        zRight.trainable = False
        zFull.trainable = False

        decoderFirstLayer.trainable = False

        # Left VAE model which can't train middle
        outputs = self.leftDecoder(self.leftEncoder(leftEncoderInput))
        self.leftModel = Model(leftEncoderInput, outputs)
        self.leftModel.compile(
            optimizer=lowLearnAdam, loss=left_vae_loss)

        # Right VAE model which can't train middle
        outputs = self.rightDecoder(self.rightEncoder(rightEncoderInput))
        self.rightModel = Model(rightEncoderInput, outputs)
        self.rightModel.compile(
            optimizer=lowLearnAdam, loss=right_vae_loss)

        # Make shared layers trainable
        leftMerge.trainable = True
        rightMerge.trainable = True

        mergedLayer.trainable = True
        leftMergedLayer.trainable = True
        rightMergedLayer.trainable = True

        z_meanLeft.trainable = True
        z_log_sigmaLeft.trainable = True

        z_meanRight.trainable = True
        z_log_sigmaRight.trainable = True

        z_meanFull.trainable = True
        z_log_sigmaFull.trainable = True

        zLeft.trainable = True
        zRight.trainable = True
        zFull.trainable = True

        decoderFirstLayer.trainable = True

        # Make separate layers frozen
        leftEncoderFirstLayer.trainable = False
        leftEncoderSecondLayer.trainable = False

        rightEncoderFirstLayer.trainable = False
        rightEncoderSecondLayer.trainable = False

        leftDecoderSecondLayer.trainable = False
        leftDecoderThirdLayer.trainable = False
        leftDecoderOutput.trainable = False

        rightDecoderSecondLayer.trainable = False
        rightDecoderThirdLayer.trainable = False
        rightDecoderOutput.trainable = False

        # Define center model
        outputs = self.fullDecoder(self.fullEncoder(
            [leftEncoderInput, rightEncoderInput]))
        # Create the center model
        self.centerModel = Model(
            [leftEncoderInput, rightEncoderInput], outputs)
        self.centerModel.compile(
            optimizer=lowLearnAdam, loss=vae_loss)  # Compile

        plot_model(self.fullEncoder,
                   to_file=os.path.join('Output',
                                        str(self.params.dataSetInfo.name),
                                        'sharedVaeFullEncoder{}_{}_{}_\
                                        {}_{}_{}_{}.png'
                                        .format(str(self.params.numEpochs),
                                                str(self.params.
                                                    firstLayerSizeLeft),
                                                str(self.params.inputSizeLeft),
                                                str(self.params.
                                                    secondLayerSize),
                                                str(self.params.encodedSize),
                                                str(self.params.
                                                    firstLayerSizeRight),
                                                str(self.params.
                                                    inputSizeRight))),
                   show_shapes=True)

    def train_model(self, leftDomain, rightDomain, leftDomainVal,
                    rightDomainVal, denoising):
        """
        Trains the model

        Args:
            leftDomain (array of floats): Left input.
            rightDomain (array of floats): Right input.
            leftDomainVal (array of floats): Left validation input.
            leftDomain (array of floats): Right validation input.
            denoising (float): Amount of noise to add before trainning.

        Returns: None
        """

        # If needs noise
        if (denoising > 0):
            noise_factor = denoising
            noise = (noise_factor * np.random.normal(loc=0.0, scale=1.0,
                                                     size=leftDomain.shape))
            leftDomain_noisy = leftDomain + noise
            leftDomain_noisy = np.clip(leftDomain_noisy, 0., 1.)
            leftDomain_noisy = leftDomain

            noise = (noise_factor * np.random.normal(loc=0.0, scale=1.0,
                                                     size=rightDomain.shape))
            rightDomain_noisy = rightDomain + noise
            rightDomain_noisy = np.clip(rightDomain_noisy, 0., 1.)
        else:
            leftDomain_noisy = leftDomain
            rightDomain_noisy = rightDomain

        # Set up arrays to hold the loss over training.
        left_vae_loss_data = []
        left_vae_val_loss_data = []
        left_callback = leftCallback(
            left_vae_loss_data, left_vae_val_loss_data,
            self.betaLeft, self.kappa)

        right_vae_loss_data = []
        right_vae_val_loss_data = []
        right_callback = rightCallback(
            right_vae_loss_data, right_vae_val_loss_data,
            self.betaRight, self.kappa)

        # Train the combined model
        self.vae_model.fit([leftDomain_noisy, rightDomain_noisy],
                           [leftDomain, rightDomain],
                           epochs=self.params.numEpochs,
                           batch_size=self.params.batchSize,
                           shuffle=True,
                           verbose=1)

        # Take turns training each part of the model separately
        for i in range(self.params.numEpochs):
            print("On EPOCH: " + repr(i + 1))
            self.centerModel.fit([leftDomain_noisy, rightDomain_noisy],
                                 [leftDomain, rightDomain],
                                 epochs=1,
                                 batch_size=self.params.batchSize,
                                 shuffle=True)
            self.leftModel.fit(leftDomain_noisy, leftDomain,
                               epochs=1,
                               batch_size=self.params.batchSize,
                               validation_data=(
                                             leftDomainVal,
                                             leftDomainVal),
                               callbacks=[left_callback,
                                          TerminateOnNaN()])
            self.rightModel.fit(rightDomain_noisy, rightDomain,
                                epochs=1,
                                batch_size=self.params.batchSize,
                                validation_data=(
                                             rightDomainVal,
                                             rightDomainVal),
                                callbacks=[
                                      right_callback,
                                      TerminateOnNaN()])


        '''
        self.fullModel.fit([leftDomain_noisy, rightDomain_noisy],
                           [leftDomain, rightDomain],
                           epochs=self.params.numEpochs,
                           batch_size=self.params.batchSize,
                           validation_data=([leftDomainVal,
                                             rightDomainVal],
                                            [leftDomainVal,
                                             rightDomainVal]),
                           callbacks=[left_callback,
                                      right_callback,
                                      TerminateOnNaN()])'''

        # Make graph of the training and validation loss
        sns.set_style("darkgrid")
        fig, ax = plt.subplots()

        data = np.vstack((left_vae_loss_data, right_vae_loss_data))

        output_df = pd.DataFrame(data, [self.params.dataSetInfo.leftDomainName
                                        + "Training Loss",
                                        self.params.dataSetInfo.rightDomainName
                                        + "Training Loss"
                                        ]).T
        output_df.plot(ax=ax)

        data = np.vstack((left_vae_val_loss_data, right_vae_val_loss_data))

        output_df = pd.DataFrame(data, [self.params.dataSetInfo.leftDomainName
                                        + "Validation Loss",
                                        self.params.dataSetInfo.rightDomainName
                                        + "Validation Loss"
                                        ]).T
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        output_df.plot(style='--', ax=ax)
        plt.savefig(os.path.join('Output', self.params.dataSetInfo.name,
                                 f'Loss_{str(self.params.outputNum)}.png'
                                 ))

        t_test_df = pd.DataFrame([str(self.params.numEpochs),
                                  str(self.params.firstLayerSizeLeft),
                                  str(self.params.inputSizeLeft),
                                  # str(self.params.secondLayerSize),
                                  # str(self.params.thirdLayerSize),
                                  str(self.params.encodedSize),
                                  str(self.params.firstLayerSizeRight),
                                  str(self.params.inputSizeRight),
                                  str(self.params.kappa),
                                  str(self.params.beta),
                                  str(self.params.noise),
                                  self.params.notes],
                                 index=['Epochs', 'First Layer Left',
                                        'Input Size Left',
                                        # 'Second Layer',
                                        # 'Third Layer',
                                        'Encoded Size',
                                        'First Layer Right',
                                        'inputSizeRight', 'Kappa',
                                        'Intial Beta Value',
                                        'Noise level', 'notes']).transpose()

        file = os.path.join('Output', self.params.dataSetInfo.name,
                            f'{str(self.params.outputNum)}')
        t_test_df.to_csv(file, sep='\t')

    def generate(self, leftDomain, rightDomain):
        """
        Generates from the model. Calls the visualize of the
        data specific implementation of DataSetInfoAbstractClass.

        Args:
            leftDomain (array of floats): Left input.
            rightDomain (array of floats): Right input.

        Returns: None
        """

        # Create generated data
        leftPredicted = self.leftEncoder.predict(leftDomain,
                                                 batch_size=self.
                                                 params.batchSize)
        rightPredicted = self.rightEncoder.predict(rightDomain,
                                                   batch_size=self.
                                                   params.batchSize)

        predicted = np.append(leftPredicted, rightPredicted)

        max_coords = np.amax(predicted, axis=0)
        min_coords = np.amin(predicted, axis=0)

        rng = np.random.uniform(
            max_coords, min_coords, (rightDomain.shape[0],
                                     self.params.encodedSize))

        (left_generatedImgs, right_generatedImgs) = (self.fullDecoder.
                                                     predict(rng))

        # Create Left to Right Transformation
        (left_decoded_imgs, leftToRightImgs) = self.leftToRightModel.predict(
            leftDomain)

        # Create Right to Left Transformation
        (rightToLeftImgs, right_decoded_imgs) = self.rightToLeftModel.predict(
            rightDomain)

        # Create the cycle images
        (leftToRightCycle, _) = self.rightToLeftModel.predict(leftToRightImgs)
        (_, rightToLeftCycle) = self.leftToRightModel.predict(rightToLeftImgs)

        modelHandle = model(self.leftEncoder,
                            self.rightEncoder,
                            self.leftDecoder,
                            self.rightDecoder,
                            self.leftToRightModel,
                            self.rightToLeftModel,
                            self.leftModel,
                            self.rightModel)

        # Visualize the Data if Applicable
        self.params.dataSetInfo.visualize(modelHandle,
                                          leftPredicted, rightPredicted,
                                          rightDomain,
                                          right_decoded_imgs, rightToLeftCycle,
                                          right_generatedImgs, leftToRightImgs,
                                          leftDomain, left_decoded_imgs,
                                          leftToRightCycle, left_generatedImgs,
                                          rightToLeftImgs, self.params)

        # Find the Difference in the cycles
        leftCycleDifference = (np.absolute(left_decoded_imgs)
                               - np.absolute(leftToRightCycle))
        rightCycleDifference = (np.absolute(right_decoded_imgs)
                                - np.absolute(rightToLeftCycle))

        # Print Average Cycle Differences
        print("Left Cycle Difference: " +
              repr(np.sum(leftCycleDifference) / leftDomain.shape[0]))
        print("Right Cycle Difference: " +
              repr(np.sum(rightCycleDifference) / leftDomain.shape[0]))

        # Create Noise
        leftRandomNoise = np.random.normal(
            loc=0.0, scale=1.0, size=leftDomain.shape)

        rightRandomNoise = np.random.normal(
            loc=0.0, scale=1.0, size=rightDomain.shape)

        # Create Left to Right Transformation Noise
        (left_decoded_imgs_noise,
         leftToRightImgsNoise) = self.leftToRightModel.predict(leftRandomNoise)

        # Create Right to Left Transformation Noise
        (rightToLeftImgsNoise,
         right_decoded_imgs_noise) = (self.rightToLeftModel.
                                      predict(rightRandomNoise))

        # Create the Noise cycle
        (leftToRightCycleNoise, _) = self.rightToLeftModel.predict(
            leftToRightImgsNoise)
        (_, rightToLeftCycleNoise) = self.leftToRightModel.predict(
            rightToLeftImgsNoise)

        leftCycleDifferenceNoise = (np.absolute(left_decoded_imgs_noise)
                                    - np.absolute(leftToRightCycleNoise))
        rightCycleDifferenceNoise = (np.absolute(right_decoded_imgs_noise)
                                     - np.absolute(rightToLeftCycleNoise))

        print("Left Cycle Noise Difference: " +
              repr(np.sum(leftCycleDifferenceNoise) / leftDomain.shape[0]))
        print("Right Cycle Noise Difference: " +
              repr(np.sum(rightCycleDifferenceNoise) / leftDomain.shape[0]))


class leftCallback(Callback):
    # For logging the loss throughout training

    def __init__(self, loss_data, val_loss, beta, kappa):
        self.loss_data = loss_data
        self.val_loss = val_loss
        self.beta = beta
        self.kappa = kappa

    def on_epoch_end(self, epoch, logs={}):
        self.loss_data.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        if K.get_value(self.beta) < 1:
            K.set_value(self.beta, K.get_value(self.beta) + self.kappa)
        else:
            K.set_value(self.beta, 1)


class rightCallback(Callback):
    # For logging the loss throughout training

    def __init__(self, loss_data, val_loss, beta, kappa):
        self.loss_data = loss_data
        self.val_loss = val_loss
        self.beta = beta
        self.kappa = kappa

    def on_epoch_end(self, epoch, logs={}):
        self.loss_data.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        if K.get_value(self.beta) < 1:
            K.set_value(self.beta, K.get_value(self.beta) + self.kappa)
        else:
            K.set_value(self.beta, 1)
