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
import plotly as py
import scipy
from scipy import misc
from sklearn import preprocessing
import plotly.graph_objs as go
# Local files
from DataSetInfoAbstractClass import dataSetInfoAbstract
from model_objects import model_parameters


class dataInfo(dataSetInfoAbstract):

    def __init__(self):
        """
        Defines the object's name, file locations, image size and domain names

        Args: None

        Returns: None
        """
        self.name = "Cognoma"
        self.training_file = os.path.join('Data', 'Cognoma_Data', 'Training',
                                          'Cognoma_Training.pkl')
        self.testing_file = os.path.join('Data', 'Cognoma_Data', 'Testing',
                                         'Cognoma_Testing.pkl')
        self.rightXDim = 40
        self.rightYDim = 36
        self.leftXDim = 100
        self.leftYDim = 80
        self.rightDomainName = "Mutation"
        self.leftDomainName = "Expression"

    def load(self):
        """
        Loads the testing and training data from pickle files.

        Args: None

        Returns: (Float array, Float array, Float array, Float array)
                    The left training data, left testing data,
                    right training data, right testing data
        """
        with open(self.training_file, "rb") as fp:
            (x_temp, a_temp) = cPickle.load(fp)

        # a_temp = preprocessing.scale(a_temp)

        # a_temp = preprocessing.normalize(a_temp, norm='l2')

        np.random.shuffle(x_temp)
        np.random.shuffle(a_temp)

        length = x_temp.shape[0]

        x_train = x_temp[:int(length * .9)]
        x_test = x_temp[int(length * .9):]

        a_train = a_temp[:int(length * .9)]
        a_test = a_temp[int(length * .9):]
        return (x_train, a_train, x_test, a_test)

    def visualize(self, modelHandle,
                  leftPredicted, rightPredicted, rightDomain, right_decoded_imgs,
                  rightToLeftCycle,
                  right_generatedImgs, leftToRightImgs,
                  leftDomain, left_decoded_imgs, leftToRightCycle,
                  left_generatedImgs, rightToLeftImgs, params, n=100):
        """
        Visualizes all of the data passed to it.

        Args:
            leftPredicted (array of floats): The latent space predictions
            rightPredicted (array of floats): The latent space predictions
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
        TP53Wildtype = np.array(rightDomain[0, :])
        TP53WildtypeExp = np.array(leftDomain[0, :])

        for x in range(1, rightDomain.shape[0]):
            if rightDomain[x, 485] == 0:
                TP53Wildtype = np.vstack((TP53Wildtype, rightDomain[x, :]))
                TP53WildtypeExp = np.vstack(
                    (TP53WildtypeExp, leftDomain[x, :]))

        TP53Not = np.array(rightDomain[0, :])
        TP53NotExp = np.array(leftDomain[0, :])

        for x in range(1, rightDomain.shape[0]):
            if rightDomain[x, 485] == 1:
                TP53Not = np.vstack((TP53Not, rightDomain[x, :]))
                TP53NotExp = np.vstack((TP53NotExp, leftDomain[x, :]))

        randIndexes = np.random.randint(
            0, min(TP53Wildtype.shape[0], TP53Not.shape[0]) - 1, (100,))
        TP53Wildtype = TP53Wildtype[1:]
        TP53WildtypeExp = TP53WildtypeExp[1:]
        TP53Wildtype = TP53Wildtype[randIndexes]
        TP53WildtypeExp = TP53WildtypeExp[randIndexes]

        TP53Not = TP53Not[1:]
        TP53NotExp = TP53NotExp[1:]
        TP53Not = TP53Not[randIndexes]
        TP53NotExp = TP53NotExp[randIndexes]

        (TP53WildTypeToExp, _) = modelHandle.rightToLeftModel.predict(TP53Wildtype)

        TP53WildTypeToExpMSE = np.square(np.subtract(
            TP53WildtypeExp, TP53WildTypeToExp)).mean(axis=0)

        import copy

        TP53Induced = copy.copy(TP53Wildtype)
        TP53InducedExp = copy.copy(TP53WildtypeExp)

        TP53Induced[:, 485] = 1

        (TP53InducedToExp, _) = modelHandle.rightToLeftModel.predict(TP53Induced)

        TP53InducedToExpMSE = np.square(np.subtract(
            TP53InducedExp, TP53InducedToExp)).mean(axis=0)

        ttest_results = scipy.stats.ttest_ind(
            TP53WildTypeToExpMSE, TP53InducedToExpMSE)
        t_stat = ttest_results.statistic
        p_val = ttest_results.pvalue

        print t_stat
        print p_val

        table_data = dict(values=[[t_stat], [p_val]])
        table_labels = dict(values=['Stat', 'PVal'])
        table = [go.Table(cells=table_data, header=table_labels)]

        py.offline.plot(table, filename=os.path.join('Output', params.dataSetInfo.name,
                                                     't_test_{}'.format(str(params.outputNum))))

        TP53NotInduced = copy.copy(TP53Not)
        TP53NotInduced[:, 485] = 0

        (TP53NotInducedToExp, _) = modelHandle.rightToLeftModel.predict(TP53NotInduced)

        SyntheticPVals = np.array([])
        for x in range(0, leftDomain.shape[0]):
            ttest_results = scipy.stats.ttest_ind(
                TP53WildtypeExp[:, x], TP53InducedToExp[:, x])
            SyntheticPVals = np.append(SyntheticPVals, ttest_results.statistic)

        RealPVals = np.array([])
        for x in range(0, leftDomain.shape[0]):
            ttest_results = scipy.stats.ttest_ind(
                TP53NotExp[:, x], TP53WildtypeExp[:, x])
            RealPVals = np.append(RealPVals, ttest_results.statistic)

        SyntheticNotPVals = np.array([])
        for x in range(0, leftDomain.shape[0]):
            ttest_results = scipy.stats.ttest_ind(
                TP53NotExp[:, x], TP53NotInducedToExp[:, x])
            SyntheticNotPVals = np.append(
                SyntheticNotPVals, ttest_results.statistic)

        plt.figure()
        plt.title("Scatter Plot of the t-test difference between real TP53 wildtype vs. real TP53 mutated and real vs. synthetic TP53 data")
        plt.scatter(RealPVals, SyntheticPVals, c='r',
                    label="Real TP53 Wildtype vs. Synthetic TP53 Mutated")
        plt.scatter(RealPVals, SyntheticNotPVals, c='b',
                    label="Real TP53 Mutated vs. Synthetic TP53 Wildtype")
        plt.xlabel("T-test stat of real TP53 Wildtype vs. real mutated")
        plt.ylabel("T-test stat of real and synthetic data")
        # plt.yscale('log')
        # plt.xscale('log')
        plt.legend()
        plt.savefig(os.path.join('Output', params.dataSetInfo.name,
                                 'Scatter_{}.png'.
                                 format(str(params.outputNum))))

        TP53WildtypeLatent = modelHandle.rightEncoder.predict(TP53Wildtype)
        TP53NotLatent = modelHandle.rightEncoder.predict(TP53Not)

        LantentSpacePerc = (TP53WildtypeLatent - TP53NotLatent) / TP53NotLatent

        LantentSpacePerc = LantentSpacePerc.mean(axis=0)

        plt.figure()
        plt.bar(np.arange(LantentSpacePerc.shape[0]), LantentSpacePerc)
        plt.title(
            "Percentage difference between TP53 Wildtype and mutated for each latent space node, Mutation")
        plt.xticks(np.arange(LantentSpacePerc.shape[0]), np.arange(
            LantentSpacePerc.shape[0]))
        plt.savefig(os.path.join('Output', params.dataSetInfo.name,
                                 'TP53MutLatentSpaceComparrison_{}.png'.
                                 format(str(params.outputNum))))
        # plt.show()

        maxInd = np.argmax(LantentSpacePerc)
        print TP53WildtypeLatent.mean(axis=0)[maxInd]
        print TP53NotLatent.mean(axis=0)[maxInd]

        TP53WildtypeExpLatent = modelHandle.leftEncoder.predict(TP53WildtypeExp)
        TP53NotExpLatent = modelHandle.leftEncoder.predict(TP53NotExp)

        LantentSpacePerc = (TP53WildtypeExpLatent - TP53NotExpLatent) / TP53NotExpLatent

        LantentSpacePerc = LantentSpacePerc.mean(axis=0)

        plt.figure()
        plt.bar(np.arange(LantentSpacePerc.shape[0]), LantentSpacePerc)
        plt.title(
            "Percentage difference between TP53 Wildtype and mutated for each latent space node, Expression")
        plt.xticks(np.arange(LantentSpacePerc.shape[0]), np.arange(
            LantentSpacePerc.shape[0]))
        plt.savefig(os.path.join('Output', params.dataSetInfo.name,
                                 'TP53ExpLatentSpaceComparrison_{}.png'.
                                 format(str(params.outputNum))))
        # plt.show()

        import cPickle
        file = os.path.join('Data', 'Cognoma_Data', 'Training',
                            'cancerLabels.csv')
        with open(file, "rb") as fp:
            labels = cPickle.load(fp)

        import umap as up
        n_sne = n

        umap = up.UMAP(n_neighbors=5,
                       min_dist=0.1,
                       metric='correlation')
        randIndexes = np.random.randint(0, leftDomain.shape[0], (n_sne,))
        umap_results = umap.fit_transform(leftPredicted[randIndexes, :])

        plt.figure()
        plt.title(params.dataSetInfo.leftDomainName + " Latent Space")

        for g in np.unique(labels[randIndexes]):
            i = np.where(labels[randIndexes] == g)
            plt.scatter(umap_results[i, 0], umap_results[i, 1], label=g)
        plt.legend()
        plt.savefig(os.path.join('Output', 'Cognoma',
                                 'LeftUMap_{}.png'.
                                 format(str(params.outputNum))))

        umap = up.UMAP(n_neighbors=5,
                       min_dist=0.1,
                       metric='correlation')
        randIndexes = np.random.randint(0, rightDomain.shape[0], (n_sne,))
        umap_results = umap.fit_transform(rightPredicted[randIndexes, :])

        plt.figure()
        plt.title(params.dataSetInfo.rightDomainName + " Latent Space")
        for g in np.unique(labels[randIndexes]):
            i = np.where(labels[randIndexes] == g)
            plt.scatter(umap_results[i, 0], umap_results[i, 1], label=g)
        plt.legend()
        plt.savefig(os.path.join('Output', 'Cognoma',
                                 'RightUMap_{}.png'.
                                 format(str(params.outputNum))))

        rightDomainFloat = rightDomain.astype('float64')

        randIndexes = np.random.randint(0, rightDomain.shape[0], (10,))

        plt.figure(figsize=(120, 40))
        for i in range(10):

            # display original Depth Map
            ax = plt.subplot(6, 10, i + 1)
            plt.imshow(rightDomainFloat[randIndexes[i]].
                       reshape(self.rightXDim,
                               self.rightYDim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Right Domain (Mutated) Truth")

            # display depth map reconstruction
            ax = plt.subplot(6, 10, i + 1 + 10)
            plt.imshow(right_decoded_imgs[randIndexes[i]]
                       .reshape(self.rightXDim,
                                self.rightYDim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Right Domain (Mutated) Reconstructed")

            # display depth map reconstruction
            ax = plt.subplot(6, 10, i + 1 + 2 * 10)
            plt.imshow(rightToLeftImgs[randIndexes[i]]
                       .reshape(self.leftXDim,
                                self.leftYDim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Right Domain (Mutated) translated into Left")

            # display right to left transformed cycled through
            ax = plt.subplot(6, 10, i + 1 + 3 * 10)
            plt.imshow(leftDomain[randIndexes[i]].reshape(self.leftXDim,
                                                          self.leftYDim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Left Domain (Expression) Truth")

            # display depth generated
            ax = plt.subplot(6, 10, i + 1 + 4 * 10)
            plt.imshow(left_decoded_imgs[randIndexes[i]]
                       .reshape(self.leftXDim,
                                self.leftYDim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Left Domain (Expression) Reconstructed")

            # display depth generated
            ax = plt.subplot(6, 10, i + 1 + 5 * 10)
            plt.imshow(leftToRightImgs[randIndexes[i]]
                       .reshape(self.rightXDim,
                                self.rightYDim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Left Domain (Expression) translated into Right")

        plt.savefig(os.path.join('Output', 'Cognoma',
                                 'Visualized_{}.png'.
                                 format(str(params.outputNum))))

        # Make matrix of data for cluster map for the Right Domain
        num_examples = n
        random = np.random.randint(0, num_examples,
                                   size=num_examples)
        X = np.append(rightDomain[random, :],
                      right_decoded_imgs[random, :],
                      axis=0)
        X = np.append(X,
                      leftToRightImgs[random, :],
                      axis=0)
        y = np.append(np.zeros(num_examples),
                      np.ones(num_examples),
                      axis=0)
        y = np.append(y,
                      np.full(num_examples, 2),
                      axis=0)

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

        label_colors = np.append(label_colors,
                                 np.repeat(sns.dark_palette("purple", 1,
                                                            reverse=True,),
                                           num_examples, axis=0),
                                 axis=0)

        # Build cluster map and hide the legend
        g = sns.clustermap(cluster.astype('float64'), row_cluster=False,
                           col_colors=label_colors,
                           xticklabels=False, yticklabels=False)
        # To re-enable colorbar, comment-out the following line
        g.fig.suptitle('Right Domain (Mutated), Original (Blue),'
                       ' Reconstructed (Green), and Translated (Purple)'
                       ' Clustermap.')
        g.cax.set_visible(False)

        plt.savefig(os.path.join('Output', 'Cognoma',
                                 'RightHeatmap_{}.png'.
                                 format(str(params.outputNum))))

        # For the Left Domain
        random = np.random.randint(0, num_examples,
                                   size=num_examples)
        X = np.append(leftDomain[random, :],
                      left_decoded_imgs[random, :], axis=0)
        X = np.append(X,
                      rightToLeftImgs[random, :],
                      axis=0)
        y = np.append(np.zeros(num_examples),
                      np.ones(num_examples), axis=0)
        y = np.append(y,
                      np.full(num_examples, 2),
                      axis=0)

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
        label_colors = np.append(label_colors,
                                 np.repeat(sns.dark_palette("purple", 1,
                                                            reverse=True,),
                                           num_examples, axis=0),
                                 axis=0)

        # Build cluster map and hide the legend
        g = sns.clustermap(cluster.astype('float64'), row_cluster=False,
                           col_colors=label_colors,
                           xticklabels=False, yticklabels=False)
        # To re-enable colorbar, comment-out the following line
        g.cax.set_visible(False)
        g.fig.suptitle('Left Domain (Expression), Original (Blue),'
                       ' Reconstructed (Green), and Translated (Purple)'
                       ' Clustermap')

        plt.savefig(os.path.join('Output', 'Cognoma',
                                 'LeftHeatmap_{}.png'.
                                 format(str(params.outputNum))))
        # plt.show()
        return
