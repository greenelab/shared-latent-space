from __future__ import print_function
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

import matplotlib.pyplot as plt

import os

from model_objects import model_parameters

class shared_vae_class(object):
    def __init__(self, model_parameters):
        self.params = model_parameters

    def compile_model(self):
        #the Sampling function for the VAE
        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.params.encodedSize))
            return z_mean + K.exp(z_log_sigma) * epsilon

        #Loss function for the VAE
        def vae_loss(inputs, finalLayer): #Loss function comprised of two parts, Cross_entropy, and divergense
            reconstruction_loss = K.sum(K.square(finalLayer-inputs))
            kl_loss = - 0.5 * K.sum(1 + z_log_sigmaFull - K.square(z_meanFull) - K.square(K.exp(z_log_sigmaFull)), axis=-1)
            total_loss = K.mean(reconstruction_loss + kl_loss)    
            return total_loss

        def left_vae_loss(inputs, finalLayer): #Loss function comprised of two parts, Cross_entropy, and divergense
            reconstruction_loss = K.sum(K.square(finalLayer-inputs))
            kl_loss = - 0.5 * K.sum(1 + z_log_sigmaLeft - K.square(z_meanLeft) - K.square(K.exp(z_log_sigmaLeft)), axis=-1)
            total_loss = K.mean(reconstruction_loss + kl_loss)    
            return total_loss

        def right_vae_loss(inputs, finalLayer): #Loss function comprised of two parts, Cross_entropy, and divergense
            reconstruction_loss = K.sum(K.square(finalLayer-inputs))
            kl_loss = - 0.5 * K.sum(1 + z_log_sigmaRight - K.square(z_meanRight) - K.square(K.exp(z_log_sigmaRight)), axis=-1)
            total_loss = K.mean(reconstruction_loss + kl_loss)    
            return total_loss

        #Define the Encoder with the Left and Right branches
        leftEncoderInput = Input(shape=(self.params.inputSize,))
        leftEncoderFirstLayer = Dense(self.params.firstLayerSize, activation = 'relu')(leftEncoderInput)
        leftEncoderSecondLayer = Dense(self.params.secondLayerSize, activation = 'relu')(leftEncoderFirstLayer)

        rightEncoderInput = Input(shape=(self.params.inputSize,))
        rightEncoderFirstLayer = Dense(self.params.firstLayerSize, activation = 'relu')(rightEncoderInput)
        rightEncoderSecondLayer = Dense(self.params.secondLayerSize, activation = 'relu')(rightEncoderFirstLayer)

        encoderMergeLayer = Dense(self.params.thirdLayerSize, activation = 'relu')
        leftMerge = encoderMergeLayer(leftEncoderSecondLayer)
        rightMerge = encoderMergeLayer(rightEncoderSecondLayer)

        mergedLayer = keras.layers.average([leftMerge,rightMerge])
        leftMergedLayer = keras.layers.average([leftMerge,leftMerge])
        rightMergedLayer = keras.layers.average([rightMerge,rightMerge])



        z_mean = Dense(self.params.encodedSize)
        z_log_sigma = Dense(self.params.encodedSize)

        z_meanLeft = z_mean(leftMergedLayer)
        z_log_sigmaLeft = z_log_sigma(leftMergedLayer)

        z_meanRight= z_mean(rightMergedLayer)
        z_log_sigmaRight= z_log_sigma(rightMergedLayer)

        z_meanFull= z_mean(mergedLayer)
        z_log_sigmaFull= z_log_sigma(mergedLayer)

        zLeft = Lambda(sampling)([z_meanLeft, z_log_sigmaLeft])
        zRight = Lambda(sampling)([z_meanRight, z_log_sigmaRight])
        zFull = Lambda(sampling)([z_meanFull, z_log_sigmaFull])


        leftEncoder = Model(leftEncoderInput, zLeft)
        rightEncoder = Model(rightEncoderInput, zRight)
        self.fullEncoder = Model([leftEncoderInput, rightEncoderInput], zFull)

        #Encoder Definition
        encoder = Model([leftEncoderInput, rightEncoderInput], zFull)


        #Defining the Decoder with Left and Right Outputs
        decoderInputs = Input(shape=(self.params.encodedSize,))
        decoderFirstLayer = Dense(self.params.thirdLayerSize, activation = 'relu')(decoderInputs)

        leftDecoderSecondLayer = Dense(self.params.secondLayerSize, activation = 'relu')(decoderFirstLayer)
        leftDecoderThirdLayer = Dense(self.params.firstLayerSize, activation = 'relu')(leftDecoderSecondLayer)
        leftDecoderOutput = Dense(self.params.inputSize, activation = 'sigmoid')(leftDecoderThirdLayer)

        rightDecoderSecondLayer = Dense(self.params.secondLayerSize, activation = 'relu')(decoderFirstLayer)
        rightDecoderThirdLayer = Dense(self.params.firstLayerSize, activation = 'relu')(rightDecoderSecondLayer)
        rightDecoderOutput = Dense(self.params.inputSize, activation = 'sigmoid')(rightDecoderThirdLayer)

        #Decoder Definition
        self.decoder = Model(decoderInputs, [leftDecoderOutput, rightDecoderOutput])

        leftDeocder = Model(decoderInputs, leftDecoderOutput)
        rightDecoder = Model(decoderInputs, rightDecoderOutput)
        #decoder.summary()


        #Left to Right transition
        outputs = self.decoder(leftEncoder(leftEncoderInput))
        self.leftToRightModel = Model(leftEncoderInput, outputs)
        #leftToRightModel.summary()

        #Right to Left transition
        outputs = self.decoder(rightEncoder(rightEncoderInput))
        self.rightToLeftModel = Model(rightEncoderInput, outputs)
        #rightToLeftModel.summary()

        #Full Model
        outputs = self.decoder(encoder([leftEncoderInput, rightEncoderInput]))
        self.vae_model = Model([leftEncoderInput, rightEncoderInput], outputs) #Create the full model
        self.vae_model.compile(optimizer='adam', loss=vae_loss) #Compile
        #vae_model.summary()
        





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


        #Left VAE model
        outputs = leftDeocder(leftEncoder(leftEncoderInput))
        self.leftModel = Model(leftEncoderInput, outputs)
        self.leftModel.compile(optimizer='adam', loss=left_vae_loss)


        #Right VAE model
        outputs = rightDecoder(rightEncoder(rightEncoderInput))
        self.rightModel = Model(rightEncoderInput, outputs)
        self.rightModel.compile(optimizer='adam', loss=right_vae_loss)



        #Certain Part
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

        outputs = self.decoder(encoder([leftEncoderInput, rightEncoderInput]))
        self.centerModel = Model([leftEncoderInput, rightEncoderInput], outputs) #Create the full model
        self.centerModel.compile(optimizer='adam', loss=vae_loss) #Compile
        #vae_model.summary()

    def train_model(self, leftDomain, rightDomain, denoising):
        #Train the Model


        #Creating Noise in the Data
        x_train = leftDomain
        

        a_train = rightDomain
        
        if (denoising > 0):
            noise_factor = denoising
            x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
            x_train_noisy = np.clip(x_train_noisy, 0., 1.)

            a_train_noisy = a_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=np.shape(a_train)) 
            a_train_noisy = np.clip(a_train_noisy, 0., 1.)


        callback = EarlyStopping(monitor='loss', 
        patience=3, verbose=0, mode='auto')


        self.vae_model.fit([x_train_noisy, a_train_noisy], [x_train, a_train],
                            epochs=self.params.numEpochs,
                            batch_size= self.params.batchSize,
                            shuffle=True,
                            callbacks = [callback],
                            verbose = 1)

        for i in range(self.params.numEpochs):
            print("On EPOCH: " +  repr(i + 1))
            self.centerModel.fit([x_train_noisy, a_train_noisy], [x_train, a_train],
                            epochs=1,
                            batch_size= self.params.batchSize,
                            shuffle=True,
                            callbacks = [callback])
            self.leftModel.fit(x_train_noisy, x_train,
                            epochs=1,
                            batch_size= self.params.batchSize)
            self.rightModel.fit(a_train_noisy, a_train,
                            epochs=1,
                            batch_size= self.params.batchSize)

    def generate(self, leftDomain, rightDomain):
        #Create generated data
        predicted = self.fullEncoder.predict([leftDomain, rightDomain], 
                    batch_size=self.params.batchSize) #Predict the outputs
                
        max_coords = np.amax(predicted, axis = 0)
        min_coords = np.amin(predicted, axis = 0)


        rng = np.random.uniform(max_coords, min_coords, (params.batchSize, self.params.encodedSize)) 

        (left_generatedImgs, right_generatedImgs) = self.decoder.predict(rng)


        #Create Left to Right Transformation
        (left_decoded_imgs,leftToRightImgs) = self.leftToRightModel.predict(leftDomain)

        #Create Right to Left Transformation
        (rightToLeftImgs,right_decoded_imgs) = self.rightToLeftModel.predict(rightDomain)


        #Create the cycle images
        (leftToRightCycle, _ ) = self.rightToLeftModel.predict(leftToRightImgs)
        (_, rightToLeftCycle ) = self.leftToRightModel.predict(rightToLeftImgs)



        #Display the Original, Reconstructed, and Generated data for both the regular and invserve MNIST data
        n = 10  # how many digits we will display
        plt.figure(figsize=(120, 4))
        for i in range(n):
            # display reg original
            ax = plt.subplot(12, n, i + 1)
            plt.imshow(leftDomain[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reg reconstruction
            ax = plt.subplot(12, n, i + 1 + n)
            plt.imshow(left_decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display left to right transformed
            ax = plt.subplot(12, n, i + 1 + 2* n)
            plt.imshow(leftToRightImgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display left to right transformed cycled through
            ax = plt.subplot(12, n, i + 1 + 3* n)
            plt.imshow(leftToRightCycle[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reg generated
            ax = plt.subplot(12, n, i + 1 + 4* n)
            plt.imshow(left_generatedImgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display inv original
            ax = plt.subplot(12, n, i + 1 + 5*n)
            plt.imshow(rightDomain[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display inv reconstruction
            ax = plt.subplot(12, n, i + 1 + 6*n)
            plt.imshow(right_decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display right to left transformed
            ax = plt.subplot(12, n, i + 1 + 7* n)
            plt.imshow(rightToLeftImgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display right to left transformed
            ax = plt.subplot(12, n, i + 1 + 8* n)
            plt.imshow(rightToLeftCycle[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display inv generated
            ax = plt.subplot(12, n, i + 1 + 9* n)
            plt.imshow(right_generatedImgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()