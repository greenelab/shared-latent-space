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
import tensorflow.keras as keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Lambda, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

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
        leftEncoder = Model(leftEncoderInput, zLeft)
        rightEncoder = Model(rightEncoderInput, zRight)
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
        leftDeocder = Model(decoderInputs, leftDecoderOutput)
        rightDecoder = Model(decoderInputs, rightDecoderOutput)
        # decoder.summary()

        # Left to Right transition
        outputs = self.fullDecoder(leftEncoder(leftEncoderInput))
        self.leftToRightModel = Model(leftEncoderInput, outputs)
        # leftToRightModel.summary()

        # Right to Left transition
        outputs = self.fullDecoder(rightEncoder(rightEncoderInput))
        self.rightToLeftModel = Model(rightEncoderInput, outputs)
        # rightToLeftModel.summary()

        # Full Model
        outputs = self.fullDecoder(self.fullEncoder(
            [leftEncoderInput, rightEncoderInput]))
        # Create the full model
        self.vae_model = Model(
            [leftEncoderInput, rightEncoderInput], outputs)
        lowLearnAdam = keras.optimizers.Adam(
            lr=0.001, beta_1=0.9, beta_2=0.999,
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
        outputs = leftDeocder(leftEncoder(leftEncoderInput))
        self.leftModel = Model(leftEncoderInput, outputs)
        self.leftModel.compile(
            optimizer=lowLearnAdam, loss=left_vae_loss)

        # Right VAE model which can't train middle
        outputs = rightDecoder(rightEncoder(rightEncoderInput))
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

        # Train the combined model
        self.vae_model.fit([leftDomain_noisy, rightDomain_noisy],
                           [leftDomain, rightDomain],
                           epochs=self.params.numEpochs,
                           batch_size=self.params.batchSize,
                           shuffle=True,
                           callbacks=[callback],
                           verbose=1)

        # Take turns training each part of the model separately
        for i in range(self.params.numEpochs):
            print(("On EPOCH: " + repr(i + 1)))
            self.centerModel.fit([leftDomain_noisy, rightDomain_noisy],
                                 [leftDomain, rightDomain],
                                 epochs=1,
                                 batch_size=self.params.batchSize,
                                 shuffle=True,
                                 callbacks=[callback])
            self.leftModel.fit(leftDomain_noisy, leftDomain,
                               epochs=1,
                               batch_size=self.params.batchSize)
            self.rightModel.fit(rightDomain_noisy, rightDomain,
                                epochs=1,
                                batch_size=self.params.batchSize)

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
        predicted = self.fullEncoder.predict([leftDomain, rightDomain],
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

        randIndexes = np.random.randint(0, rightDomain.shape[0], (10,))

        # Visualize the Data if Applicable
        self.params.dataSetInfo.visualize(randIndexes, rightDomain,
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
        print(("Left Cycle Difference: " +
              repr(np.sum(leftCycleDifference) / leftDomain.shape[0])))
        print(("Right Cycle Difference: " +
              repr(np.sum(rightCycleDifference) / leftDomain.shape[0])))

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

        print(("Left Cycle Noise Difference: " +
              repr(np.sum(leftCycleDifferenceNoise) / leftDomain.shape[0])))
        print(("Right Cycle Noise Difference: " +
              repr(np.sum(rightCycleDifferenceNoise) / leftDomain.shape[0])))
