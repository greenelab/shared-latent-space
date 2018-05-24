"""
shared-latent-space/Cognoma.py

This class is an implementation of the abstract class
DataSetInfoAbstractClass. It contains the specific
implementations for how to load and visualize the Cognoma
data set. It saves the visualizations in the /Output/Cognoma
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
import pandas as pd
from scipy import misc

# Local files
from DataSetInfoAbstractClass import dataSetInfoAbstract
from model_objects import model_parameters


class dataInfo(dataSetInfoAbstract):

    def __init__(self):
        """
        Defines the object's name, file locations and image size

        Args: None

        Returns: None
        """
        self.name = "Cognoma"
        self.training_file = os.path.join('Data', 'Cognoma_Data', 'Training',
                                          'Cognoma_Training.pkl')
        self.testing_file = os.path.join('Data', 'Cognoma_Data', 'Testing',
                                         'Cognoma_Testing.pkl')

    def load(self):
        """
        Loads the testing and training data from pickle files.

        Args: None

        Returns: (Float array, Float array, Float array, Float array)
                    The left training data, left testing data,
                    right training data, right testing data
        """
        with open(self.training_file, "rb") as fp:
            (x_train, a_train) = cPickle.load(fp)

        x_test = x_train
        a_test = a_train
        return (x_train, a_train, x_test, a_test)

    def visualize(self, randIndexes, rightDomain, right_decoded_imgs,
                  rightToLeftCycle,
                  right_generatedImgs, leftToRightImgs,
                  leftDomain, left_decoded_imgs, leftToRightCycle,
                  left_generatedImgs, rightToLeftImgs, params, n=100):
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
            n (int): Defaults to 10, this doesn't apply to this visualization

        Returns: None
        """
        # Make matrix of data for cluster map for the Right Domain
        num_examples = n
        random = np.random.randint(0, num_examples,
                                   size=num_examples)
        X = np.append(rightDomain[random, :],
                      right_decoded_imgs[random, :], axis=0)
        y = np.append(np.zeros(num_examples),
                      np.ones(num_examples), axis=0)

        # Make the column labels for the cluster map
        index = 0
        for x in range(1, params.inputSizeRight):
            index = np.append(index, x)

        cluster = pd.DataFrame(X.transpose(), index=index, columns=y)

        # Buils the color labels for cluster map
        label_colors = np.repeat(sns.dark_palette("blue", 1, reverse=True),
                                 num_examples, axis=0)

        label_colors = np.append(label_colors,
                                 np.repeat(sns.dark_palette("green", 1,
                                                            reverse=True,),
                                           num_examples, axis=0),
                                 axis=0)

        # Build cluster map and hide the legend
        g = sns.clustermap(cluster.astype('float64'), row_cluster=False,
                           col_colors=label_colors,
                           xticklabels=False, yticklabels=False)
        # To re-enable colorbar, comment-out the following line
        g.fig.suptitle('Right Domain (Mutation), Original and Reconstructed Clustermap')
        g.cax.set_visible(False)

        # For the Left Domain
        random = np.random.randint(0, num_examples,
                                   size=num_examples)
        X = np.append(leftDomain[random, :],
                      left_decoded_imgs[random, :], axis=0)
        y = np.append(np.zeros(num_examples),
                      np.ones(num_examples), axis=0)

        # Make the column labels for the cluster map
        index = 0
        for x in range(1, params.inputSizeLeft):
            index = np.append(index, x)

        cluster = pd.DataFrame(X.transpose(), index=index, columns=y)

        # Buils the color labels for cluster map
        label_colors = np.repeat(sns.dark_palette("blue", 1, reverse=True),
                                 num_examples, axis=0)

        label_colors = np.append(label_colors,
                                 np.repeat(sns.dark_palette("green", 1,
                                                            reverse=True,),
                                           num_examples, axis=0),
                                 axis=0)

        # Build cluster map and hide the legend
        g = sns.clustermap(cluster.astype('float64'), row_cluster=False,
                           col_colors=label_colors,
                           xticklabels=False, yticklabels=False)
        # To re-enable colorbar, comment-out the following line
        g.cax.set_visible(False)
        g.fig.suptitle('Left Domain (Expression), Original and Reconstructed Clustermap')
        plt.show()
        return
