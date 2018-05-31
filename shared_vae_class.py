"""
shared-latent-space/shared_vae_class.py

This class contains the model and the associated functions.
it is called by main_file.py. It uses many models to represent
the various parts of the shared latent space VAE and allow for
the training of certain parts in isolation. To train, it trains
the model as a whole for n epochs, then switches between training
only the shared part of the model, the left part, and the right
parts individually for n epochs. This is achieved by freezing the
weights. In the generate() function, the class calls upon the
specific implementation of DataSetInfoAbstractClass for the
given dataset.


Author: Chris Williams
Date: 5/22/18
"""


import os

import numpy as np
import matplotlib.pyplot as plt
import keras

from keras.models import Model
from keras.layers import Dense, Activation, Lambda, Input
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import EarlyStopping, Callback
from keras.utils import plot_model

# Local files
import ICVL
import MNIST

from model_objects import model_parameters


class shared_vae_class(object):

    def __init__(self, model_parameters):
        """
        Takes in parameters of the model.

        Args:
            model_parameters (model_parameters): Parameters for the model.

        Returns: None
        """
        self.params = model_parameters

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
            reconstruction_loss = K.sum(K.square(finalLayer - inputs))
            kl_loss = - 0.5 * K.sum(1 + z_log_sigmaRight - K.square(
                z_meanRight) - K.square(K.exp(z_log_sigmaRight)), axis=-1)
            total_loss = K.mean(reconstruction_loss + kl_loss)
            return total_loss

        # Define the Encoder with the Left and Right branches
        leftEncoderInput = Input(shape=(self.params.inputSizeLeft,))
        leftEncoderFirstLayer = Dense(
            self.params.firstLayerSizeLeft,
            activation='relu')(leftEncoderInput)

        rightEncoderInput = Input(shape=(self.params.inputSizeRight,))
        rightEncoderFirstLayer = Dense(
            self.params.firstLayerSizeRight,
            activation='relu')(rightEncoderInput)

        encoderMergeLayer = Dense(
            self.params.thirdLayerSize, activation='relu')
        leftMerge = encoderMergeLayer(leftEncoderFirstLayer)
        rightMerge = encoderMergeLayer(rightEncoderFirstLayer)

        z_mean = Dense(self.params.encodedSize)
        z_log_sigma = Dense(self.params.encodedSize)

        # These three sets are used in differen models
        z_meanLeft = z_mean(leftMerge)
        z_log_sigmaLeft = z_log_sigma(leftMerge)

        z_meanRight = z_mean(rightMerge)
        z_log_sigmaRight = z_log_sigma(rightMerge)

        zLeft = Lambda(sampling)([z_meanLeft, z_log_sigmaLeft])
        zRight = Lambda(sampling)([z_meanRight, z_log_sigmaRight])

        # These are the three different models
        self.leftEncoder = Model(leftEncoderInput, zLeft)
        self.rightEncoder = Model(rightEncoderInput, zRight)

        # Defining the Decoder with Left and Right Outputs
        decoderInputs = Input(shape=(self.params.encodedSize,))
        decoderFirstLayer = Dense(
            self.params.thirdLayerSize,
            activation='relu')(decoderInputs)

        leftDecoderThirdLayer = Dense(
            self.params.firstLayerSizeLeft,
            activation='relu')(decoderFirstLayer)
        leftDecoderOutput = Dense(
            self.params.inputSizeLeft,
            activation='sigmoid')(leftDecoderThirdLayer)

        rightDecoderThirdLayer = Dense(
            self.params.firstLayerSizeRight,
            activation='relu')(decoderFirstLayer)
        rightDecoderOutput = Dense(
            self.params.inputSizeRight,
            activation='sigmoid')(rightDecoderThirdLayer)

        # Three different Decoders
        self.fullDecoder = Model(
            decoderInputs, [leftDecoderOutput, rightDecoderOutput])
        leftDecoder = Model(decoderInputs, leftDecoderOutput)
        rightDecoder = Model(decoderInputs, rightDecoderOutput)
        # decoder.summary()

        # Left to Right transition
        outputs = self.fullDecoder(self.leftEncoder(leftEncoderInput))
        self.leftToRightModel = Model(leftEncoderInput, outputs)
        # leftToRightModel.summary()

        # Right to Left transition
        outputs = self.fullDecoder(self.rightEncoder(rightEncoderInput))
        self.rightToLeftModel = Model(rightEncoderInput, outputs)
        # rightToLeftModel.summary()

        # TESTINGINGINGINGINGING
        outputs = rightDecoder(self.leftEncoder(leftEncoderInput))
        self.leftToRightTransitionModel = Model(leftEncoderInput, outputs)
        self.leftToRightTransitionModel.compile(optimizer='Adam',
                                                loss=left_vae_loss)

        # TESTINGINGINGINGINGING
        outputs = leftDecoder(self.rightEncoder(rightEncoderInput))
        self.rightToLeftTransitionModel = Model(rightEncoderInput, outputs)
        self.rightToLeftTransitionModel.compile(optimizer='Adam',
                                                loss=right_vae_loss)

        outputs = leftDecoder(self.leftEncoder(leftEncoderInput))
        self.leftModel = Model(leftEncoderInput, outputs)
        self.leftModel.compile(
            optimizer='Adam', loss=left_vae_loss)

        outputs = rightDecoder(self.rightEncoder(rightEncoderInput))
        self.rightModel = Model(rightEncoderInput, outputs)
        self.rightModel.compile(
            optimizer='Adam', loss=right_vae_loss)

    def train_model(self, leftDomain, rightDomain, denoising):
        """
        Trains the model

        Args:
            leftDomain (array of floats): Left input.
            rightDomain (array of floats): Right input.
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

        callback = EarlyStopping(monitor='loss',
                                 patience=3, verbose=0, mode='auto')

        left_vae_loss_data = []
        left_callback = custom_callback(left_vae_loss_data)

        right_vae_loss_data = []
        right_callback = custom_callback(right_vae_loss_data)

        for i in range(self.params.numEpochs):
            print("On EPOCH: " + repr(i + 1))
            self.leftModel.fit(leftDomain_noisy, leftDomain,
                               epochs=1,
                               batch_size=self.params.batchSize,
                               callback=[left_callback])
            self.rightModel.fit(rightDomain_noisy, rightDomain,
                                epochs=1,
                                batch_size=self.params.batchSize,
                                callback=[right_callback])

        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd

        sns.set_style("darkgrid")
        data = np.vstack((left_vae_loss_data, right_vae_loss_data))

        output_df = pd.DataFrame(data, ["Left to Right Loss", "Right to Left Loss"]).T
        output_df.plot()
        # plt.show()

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
        predicted = self.leftEncoder.predict(leftDomain,
                                             batch_size=self.params.batchSize)

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

        # Visualize the Data if Applicable
        self.params.dataSetInfo.visualize(rightDomain,
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

        # Create the Noise cycle images
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


class custom_callback(Callback):

    def __init__(self, loss_data):
        self.loss_data = loss_data

    def on_epoch_end(self, epoch, logs={}):
        self.loss_data.append(logs.get('loss'))
