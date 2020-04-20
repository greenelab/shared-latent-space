"""
shared-latent-space/ICVL.py

This class is an implementation of the abstract class
DataSetInfoAbstractClass. It contains the specific
implementations for how to load and visualize the ICVL
data set. It saves the visualizations in the /Output/ICVL
folder with various parameters of the model in the name
of the file.


Author: Chris Williams
Date: 5/22/18
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Local files
from model_objects import model_parameters
from DataSetInfoAbstractClass import dataSetInfoAbstract
import pickle


class dataInfo(dataSetInfoAbstract):

    def __init__(self):
        """
        Defines the object's name, file locations and image size

        Args: None

        Returns: None
        """
        self.name = "ICVL"
        self.training_file = os.path.join('Data', 'ICVL_Data', 'Training',
                                          'ICVL_Training.pkl')
        self.testing_file = os.path.join('Data', 'ICVL_Data', 'Testing',
                                         'ICVL_Testing.pkl')
        self.Xdim = 60
        self.Ydim = 80

    def load(self):
        """
        Loads the testing and training data from pickle files.

        Args: None

        Returns: (Float array, Float array, Float array, Float array)
                    The left training data, left testing data,
                    right training data, right testing data
        """

        with open(self.training_file, "rb") as fp:
            (x_train, a_train) = pickle.load(fp)

        with open(self.testing_file, "rb") as fp:
            (x_test, a_test) = pickle.load(fp)
        return (x_train, a_train, x_test, a_test)

    # This file plots the lines for the various fingers. This is all hardcoded
    # from the files
    def draw_hands(self, Xs, Ys):
        """
        Plots a series of lines between predetermined spots in the
        X and Y arrays.

        Args:
            Xs (Float array): Array of x coords
            Ys (Float array): Array of y coords

        Returns: None
        """

        plt.scatter(Xs, Ys)

        plt.gca().invert_yaxis()

        # These indices represent the various joints and colors
        idxs = [(0, 1, 'k'), (1, 2, 'g'), (2, 3, 'g'), (3, 16, 'g'),
                (0, 4, 'r'), (4, 5, 'r'), (5, 6, 'r'), (6, 17, 'r'),
                (0, 7, 'm'), (7, 8, 'm'), (8, 9, 'm'), (9, 18, 'm'),
                (0, 10, 'y'), (10, 11, 'y'), (11, 12, 'y'), (12, 19, 'y'),
                (0, 13, 'b'), (13, 14, 'b'), (14, 15, 'b'), (15, 20, 'b')]

        for idx_tuple in idxs:
            i, j, c = idx_tuple
            linesStart = np.array([Xs[i], Xs[j]])
            linesEnd = np.array([Ys[i], Ys[j]])
            plt.plot(linesStart, linesEnd, c)

    def visualize(self, randIndexes, rightDomain, right_decoded_imgs,
                  rightToLeftCycle,
                  right_generatedImgs, leftToRightImgs,
                  leftDomain, left_decoded_imgs, leftToRightCycle,
                  left_generatedImgs, rightToLeftImgs, params, n=10):
        """
        Visualizes all of the data passed to it.

        Args:
            randIndexes (array of ints): Random points to portray,
                                         but same for each set of data
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

        plt.figure(figsize=(120, 40))
        for i in range(n):

            # display original Depth Map
            ax = plt.subplot(5, n, i + 1)
            plt.imshow(rightDomain[randIndexes[i]].reshape(self.Xdim,
                                                           self.Ydim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Right Truth")

            # display depth map reconstruction
            ax = plt.subplot(5, n, i + 1 + n)
            plt.imshow(right_decoded_imgs[randIndexes[i]].reshape(self.Xdim,
                                                                  self.Ydim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Right Reconstructed")

            # display right to left transformed cycled through
            ax = plt.subplot(5, n, i + 1 + 2 * n)
            plt.imshow(rightToLeftCycle[randIndexes[i]].reshape(self.Xdim,
                                                                self.Ydim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Right Cycle")

            # display depth generated
            ax = plt.subplot(5, n, i + 1 + 3 * n)
            plt.imshow(right_generatedImgs[randIndexes[i]].reshape(self.Xdim,
                                                                   self.Ydim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Right Generated")

            # display left to right transformed
            ax = plt.subplot(5, n, i + 1 + 4 * n)
            plt.imshow(leftToRightImgs[randIndexes[i]].reshape(self.Xdim,
                                                               self.Ydim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Left to Right Transform")
        # save the output
        plt.savefig(os.path.join('Output', 'ICVL',
                                 'Right_{}_{}_{}_{}_{}_{}_{}_{}.png'.
                                 format(str(params.numEpochs),
                                        str(params.firstLayerSizeLeft),
                                        str(params.inputSizeLeft),
                                        str(params.secondLayerSize),
                                        str(params.thirdLayerSize),
                                        str(params.encodedSize),
                                        str(params.firstLayerSizeRight),
                                        str(params.inputSizeRight))))

        plt.figure(figsize=(120, 40))
        for i in range(n):
            # Display the knucle map
            Xs = np.array(leftDomain[randIndexes[i]][0::3])
            Ys = np.array(leftDomain[randIndexes[i]][1::3])
            ax = plt.subplot(5, n, i + 1)
            self.draw_hands(Xs, Ys)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Left Truth")

            # display knuckle map reconstruction
            ax = plt.subplot(5, n, i + 1 + n)
            Xs = np.array(left_decoded_imgs[randIndexes[i]][0::3])
            Ys = np.array(left_decoded_imgs[randIndexes[i]][1::3])
            self.draw_hands(Xs, Ys)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Left Reconstructed")

            # display right to left transformed cycled through
            ax = plt.subplot(5, n, i + 1 + 2 * n)
            Xs = np.array(leftToRightCycle[randIndexes[i]][0::3])
            Ys = np.array(leftToRightCycle[randIndexes[i]][1::3])
            self.draw_hands(Xs, Ys)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Left Cycle")

            # display generated knuckle map
            ax = plt.subplot(5, n, i + 1 + 3 * n)
            Xs = np.array(left_generatedImgs[randIndexes[i]][0::3])
            Ys = np.array(left_generatedImgs[randIndexes[i]][1::3])
            self.draw_hands(Xs, Ys)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Left Generated")

            # display left to right transformed
            ax = plt.subplot(5, n, i + 1 + 4 * n)
            Xs = np.array(rightToLeftImgs[randIndexes[i]][0::3])
            Ys = np.array(rightToLeftImgs[randIndexes[i]][1::3])
            self.draw_hands(Xs, Ys)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Right to Left Transformed")
        # Save the Output
        plt.savefig(os.path.join('Output', 'ICVL',
                                 'Left_{}_{}_{}_{}_{}_{}_{}_{}.png'.
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
