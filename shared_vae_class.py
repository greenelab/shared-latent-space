import numpy as np
from numpy import linalg as LA

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Lambda, Input
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K
from keras import objectives
from keras.callbacks import EarlyStopping, Callback
from keras.layers import LeakyReLU
from keras.utils import plot_model
import matplotlib.pyplot as plt

import os

from model_objects import model_parameters
import ICVL
import MNIST


class shared_vae_class(object):

    def __init__(self, model_parameters):
        self.params = model_parameters

    def compile_model(self):

        # the Sampling function for the VAE
        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(
                shape=(K.shape(z_mean)[0], self.params.encodedSize))
            return z_mean + K.exp(z_log_sigma) * epsilon

        # Loss function for the VAE
        # Loss function comprised of two parts, Cross_entropy, and
        # divergense
        def vae_loss(inputs, finalLayer):
            reconstruction_loss = K.sum(K.square(finalLayer - inputs))
            kl_loss = - 0.5 * K.sum(1 + z_log_sigmaFull - K.square(
                z_meanFull) - K.square(K.exp(z_log_sigmaFull)), axis=-1)
            total_loss = K.mean(reconstruction_loss + kl_loss)
            return total_loss

        # Loss function for the left VAE
        # Loss function comprised of two parts, Cross_entropy, and
        # divergense
        def left_vae_loss(inputs, finalLayer):
            reconstruction_loss = K.sum(K.square(finalLayer - inputs))
            kl_loss = - 0.5 * K.sum(1 + z_log_sigmaLeft - K.square(
                z_meanLeft) - K.square(K.exp(z_log_sigmaLeft)), axis=-1)
            total_loss = K.mean(reconstruction_loss + kl_loss)
            return total_loss

        # Loss function for the right VAE
        # Loss function comprised of two parts, Cross_entropy, and
        # divergense
        def right_vae_loss(inputs, finalLayer):
            reconstruction_loss = K.sum(K.square(finalLayer - inputs))
            kl_loss = - 0.5 * K.sum(1 + z_log_sigmaRight - K.square(
                z_meanRight) - K.square(K.exp(z_log_sigmaRight)), axis=-1)
            total_loss = K.mean(reconstruction_loss + kl_loss)
            return total_loss

        # Define the Encoder with the Left and Right branches
        leftEncoderInput = Input(shape=(self.params.inputSizeLeft,))
        leftEncoderFirstLayer = Dense(
            self.params.firstLayerSizeLeft, activation='relu')(leftEncoderInput)
        leftEncoderSecondLayer = Dense(
            self.params.secondLayerSize, activation='relu')(leftEncoderFirstLayer)

        rightEncoderInput = Input(shape=(self.params.inputSizeRight,))
        rightEncoderFirstLayer = Dense(
            self.params.firstLayerSizeRight, activation='relu')(rightEncoderInput)
        rightEncoderSecondLayer = Dense(
            self.params.secondLayerSize, activation='relu')(rightEncoderFirstLayer)

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
            self.params.thirdLayerSize, activation='relu')(decoderInputs)

        leftDecoderSecondLayer = Dense(
            self.params.secondLayerSize, activation='relu')(decoderFirstLayer)
        leftDecoderThirdLayer = Dense(
            self.params.firstLayerSizeLeft, activation='relu')(leftDecoderSecondLayer)
        leftDecoderOutput = Dense(
            self.params.inputSizeLeft, activation='sigmoid')(leftDecoderThirdLayer)

        rightDecoderSecondLayer = Dense(
            self.params.secondLayerSize, activation='relu')(decoderFirstLayer)
        rightDecoderThirdLayer = Dense(
            self.params.firstLayerSizeRight, activation='relu')(rightDecoderSecondLayer)
        rightDecoderOutput = Dense(
            self.params.inputSizeRight, activation='sigmoid')(rightDecoderThirdLayer)

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
            lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
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

        plot_model(self.fullEncoder, to_file="Output/" + str(self.params.dataSetInfo.name) + "/sharedVaeFullEncoder" +
                   str(self.params.numEpochs) + '_' +
                   str(self.params.firstLayerSizeLeft) + '_' +
                   str(self.params.inputSizeLeft) + '_'
                   + str(self.params.secondLayerSize) + '_' + str(self.params.thirdLayerSize) + '_' +
                   str(self.params.encodedSize) + '_' + str(self.params.firstLayerSizeRight) +
                   '_' + str(self.params.inputSizeRight)
                   + '.png', show_shapes=True)

    def train_model(self, leftDomain, rightDomain, denoising):
        # Train the Model

        # If needs noise
        if (denoising > 0):
            noise_factor = denoising
            # leftDomain_noisy = leftDomain + (noise_factor * \
            #    np.random.normal(loc=0.0, scale=1.0, size=leftDomain.shape))
            #leftDomain_noisy = np.clip(leftDomain_noisy, 0., 1.)
            leftDomain_noisy = leftDomain

            rightDomain_noisy = rightDomain + (noise_factor *
                                               np.random.normal(loc=0.0, scale=1.0, size=rightDomain.shape))
            rightDomain_noisy = np.clip(rightDomain_noisy, 0., 1.)
        else:
            leftDomain_noisy = leftDomain
            rightDomain_noisy = rightDomain

        callback = EarlyStopping(monitor='loss',
                                 patience=3, verbose=0, mode='auto')

        # Train the combined model
        self.vae_model.fit([leftDomain_noisy, rightDomain_noisy], [leftDomain, rightDomain],
                           epochs=self.params.numEpochs,
                           batch_size=self.params.batchSize,
                           shuffle=True,
                           callbacks=[callback],
                           verbose=1)

        # Take turns training each part of the model separately
        for i in range(self.params.numEpochs):
            print("On EPOCH: " + repr(i + 1))
            self.centerModel.fit([leftDomain_noisy, rightDomain_noisy], [leftDomain, rightDomain],
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
        # Create generated data
        predicted = self.fullEncoder.predict([leftDomain, rightDomain],
                                             batch_size=self.params.batchSize)  # Predict the outputs
        max_coords = np.amax(predicted, axis=0)
        min_coords = np.amin(predicted, axis=0)

        rng = np.random.uniform(
            max_coords, min_coords, (rightDomain.shape[0], self.params.encodedSize))

        (left_generatedImgs, right_generatedImgs) = self.fullDecoder.predict(rng)

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
        self.params.dataSetInfo.visualize(randIndexes, rightDomain, right_decoded_imgs, rightToLeftCycle, right_generatedImgs, leftToRightImgs,
                                          leftDomain, left_decoded_imgs, leftToRightCycle, left_generatedImgs, rightToLeftIgms, 60, 80, self.params)

        # Find the Difference in the cycles
        leftCycleDifference = left_decoded_imgs - leftToRightCycle
        rightCycleDifference = right_decoded_imgs - rightToLeftCycle

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
         right_decoded_imgs_noise) = self.rightToLeftModel.predict(rightRandomNoise)

        # Create the Noise cycle images
        (leftToRightCycleNoise, _) = self.rightToLeftModel.predict(
            leftToRightImgsNoise)
        (_, rightToLeftCycleNoise) = self.leftToRightModel.predict(
            rightToLeftImgsNoise)

        leftCycleDifferenceNoise = left_decoded_imgs_noise - leftToRightCycleNoise
        rightCycleDifferenceNoise = right_decoded_imgs_noise - rightToLeftCycleNoise

        print("Left Cycle Noise Difference: " +
              repr(np.sum(leftCycleDifferenceNoise) / leftDomain.shape[0]))
        print("Right Cycle Noise Difference: " +
              repr(np.sum(rightCycleDifferenceNoise) / leftDomain.shape[0]))
