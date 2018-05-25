"""
shared-latent-space/MNIST.py

This class is an implementation of the abstract class
DataSetInfoAbstractClass. It contains the specific
implementations for how to load and visualize the MNIST
data set. It saves the visualizations in the /Output/MNIST
folder with various parameters of the model in the name
of the file.


Author: Chris Williams
Date: 5/22/18
"""

import os
import cPickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy import misc
from keras.datasets import mnist

# Local Files
from model_objects import model_parameters
from DataSetInfoAbstractClass import dataSetInfoAbstract


class dataInfo(dataSetInfoAbstract):

    def __init__(self):
        """
        Defines the object's name and image size

        Args: None

        Returns: None
        """
        self.name = "MNIST"
        self.training_file = os.path.join('Data', 'MNIST_Data', 'Testing',
                                          'MNIST_Testing.pkl')
        self.testing_file = os.path.join('Data', 'MNIST_Data', 'Testing',
                                         'MNIST_Testing.pkl')
        self.Xdim = 28
        self.Ydim = 28

    def load(self):
        """
        Loads the testing and training data from pickle files.

        Args: None

        Returns: (Float array, Float array, Float array, Float array)
                    The left training data, left testing data,
                    right training data, right testing data
        """

        # Loading the MNIST Data
        with open(self.training_file, "rb") as fp:
            (x_train, a_train) = cPickle.load(fp)
        with open(self.testing_file, "rb") as fp:
            (x_test, a_test) = cPickle.load(fp)
        return (x_train, a_train, x_test, a_test)

    def visualize(self, rightDomain, right_decoded_imgs,
                  rightToLeftCycle, right_generatedImgs, leftToRightImgs,
                  leftDomain, left_decoded_imgs, leftToRightCycle,
                  left_generatedImgs, rightToLeftImgs, params, n=10):
        """
        Visualizes all of the data passed to it.

        Args:
            rightDomain (array of floats): Right input.
            right_decoded_imgs (array of floats): Right input
                                                  encoded and decoded.
            rightToLeftCycle (array of floats): Right input
                                               encoded and decoded as left,
                                               then encoded and decoded as
                                               right.
            right_generatedImgs (array of floats): Random encoded points
                                                    decoded as right.
            leftToRightImgs (array of floats): Left input encoded and decoded
                                               as right.
            leftDomain (array of floats): Left input.
            left_decoded_imgs (array of floats): Left input
                                                  encoded and decoded.
            leftToRightCycle (array of floats): Left input
                                               encoded and decoded as right,
                                               then encoded and decoded as
                                               left.
            left_generatedImgs (array of floats): Random encoded points
                                                    decoded as left.
            rightToLeftImgs (array of floats): Right input encoded and decoded
                                               as left.
            params (model_parameters): Parameters of the model.
            n (int): Defaults to 10, number of visualizations.

        Returns: None
        """
        randIndexes = np.random.randint(0, rightDomain.shape[0], (n,))

        # Display the Original, Reconstructed, Transformed, Cycle, and
        # Generated data for both the regular and invserve MNIST data
        plt.figure()
        for i in range(n):
            # display reg original
            ax = plt.subplot(12, n, i + 1)
            plt.imshow(leftDomain[randIndexes[i]].reshape(self.Xdim,
                                                          self.Ydim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Left Truth")

            # display reg reconstruction
            ax = plt.subplot(12, n, i + 1 + n)
            plt.imshow(left_decoded_imgs[randIndexes[i]].reshape(self.Xdim,
                                                                 self.Ydim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Left Reconstructed")

            # display left to right transformed
            ax = plt.subplot(12, n, i + 1 + 2 * n)
            plt.imshow(leftToRightImgs[randIndexes[i]].reshape(self.Xdim,
                                                               self.Ydim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Left to Right Transform")

            # display left to right transformed cycled through
            ax = plt.subplot(12, n, i + 1 + 3 * n)
            plt.imshow(leftToRightCycle[randIndexes[i]].reshape(self.Xdim,
                                                                self.Ydim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Left Cycle")

            # display reg generated
            ax = plt.subplot(12, n, i + 1 + 4 * n)
            plt.imshow(left_generatedImgs[randIndexes[i]].reshape(self.Xdim,
                                                                  self.Ydim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Left Generated")

            # display inv original
            ax = plt.subplot(12, n, i + 1 + 5 * n)
            plt.imshow(rightDomain[randIndexes[i]].reshape(self.Xdim,
                                                           self.Ydim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Right Truth")

            # display inv reconstruction
            ax = plt.subplot(12, n, i + 1 + 6 * n)
            plt.imshow(right_decoded_imgs[randIndexes[i]].reshape(self.Xdim,
                                                                  self.Ydim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Right Reconstructed")

            # display right to left transformed
            ax = plt.subplot(12, n, i + 1 + 7 * n)
            plt.imshow(rightToLeftImgs[randIndexes[i]].reshape(self.Xdim,
                                                               self.Ydim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Right to Left Transform")

            # display right to left transformed cycled through
            ax = plt.subplot(12, n, i + 1 + 8 * n)
            plt.imshow(rightToLeftCycle[randIndexes[i]].reshape(self.Xdim,
                                                                self.Ydim))

            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Right Cycle")

            # display inv generated
            ax = plt.subplot(12, n, i + 1 + 9 * n)
            plt.imshow(right_generatedImgs[randIndexes[i]].reshape(self.Xdim,
                                                                   self.Ydim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Right Generated")
        plt.savefig(os.path.join('Output', 'MNIST',
                                 'Output_{}_{}_{}_{}_{}_{}_{}_{}.png'.
                                 format(str(params.numEpochs),
                                        str(params.firstLayerSizeLeft),
                                        str(params.inputSizeLeft),
                                        str(params.secondLayerSize),
                                        str(params.thirdLayerSize),
                                        str(params.encodedSize),
                                        str(params.firstLayerSizeRight),
                                        str(params.inputSizeRight))))
        plt.show()
        return
