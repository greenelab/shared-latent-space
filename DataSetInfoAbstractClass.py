"""
shared-latent-space/DataSetInfoAbstractClass.py

This class defines an abstract class which is specific to a given
type of data. Each data set should have its own implementation of
this data set which defines both the functions load() and visualize().
These functions are then used in shared_vae_class.py

Author: Chris Williams
Date: 5/22/18
"""

from abc import ABCMeta, abstractmethod


class dataSetInfoAbstract(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def load(self):
        """
        Loads the testing and training data from pickle files.
        This needs to return the specified data.

        Args: None

        Returns: (Float array, Float array, Float array, Float array)
                    This should return left training data, left testing data,
                    right training data, right testing data
        """
        return

    @abstractmethod
    def visualize(self, randIndexes, rightDomain, right_decoded_imgs,
                  rightToLeftCycle, right_generatedImgs, leftToRightImgs,
                  leftDomain, left_decoded_imgs, leftToRightCycle,
                  left_generatedImgs, rightToLeftImgs, params, n):
        """
        Visualizes all of the data passed to it. This does not need to return
        anything.

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
            n (int): Number of visualizations, should be given a default
                     in implementation.

        Returns: None
        """
        return
