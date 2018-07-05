"""
shared-latent-space/Cognoma.py

This class is an implementation of the abstract class
DataSetInfoAbstractClass. It contains the specific
implementations for how to load and visualize the Cognoma
data set. It saves the visualizations in the /Output/Cognoma
folder with various parameters of the model in the name
of the file.


Author: Chris Williams
Date: 6/26/18
"""
import os
import cPickle
import copy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import plotly as py
import umap as up
import plotly.graph_objs as go
import sklearn as sk
import scipy
from scipy import misc
from sklearn import preprocessing

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
                                          'Cognoma_Training_Large.pkl')
        self.rightXDim = 45
        self.rightYDim = 123
        self.leftXDim = 100
        self.leftYDim = 80
        self.rightDomainName = "Mutation"
        self.leftDomainName = "Expression"

        # For Cognoma evaluation, which mutation to induce and compare
        self.mutationID = 485

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

        x_temp, a_temp = sk.utils.shuffle(x_temp, a_temp)

        length = x_temp.shape[0]

        x_train = x_temp[:int(length * .9)]
        x_test = x_temp[int(length * .9):]

        a_train = a_temp[:int(length * .9)]
        a_test = a_temp[int(length * .9):]
        return (x_train, a_train, x_test, a_test)

    def visualize(self, modelHandle,
                  leftPredicted, rightPredicted, rightDomain,
                  right_decoded_imgs,
                  rightToLeftCycle,
                  right_generatedImgs, leftToRightImgs,
                  leftDomain, left_decoded_imgs, leftToRightCycle,
                  left_generatedImgs, rightToLeftImgs, params, n=100):
        """
        Visualizes all of the data passed to it.

        Args:
            modelHandle (model): Holds all the components of the model
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
            n (int): Defaults to 100

        Returns: None
        """

        #######################################################################
        # For calculating the t-test between the MSE of real and generated data

        # Find all TP53 wildtype examples and their gene expressions
        SNPpresent = np.array(rightDomain[0, :])
        SNPpresentExp = np.array(leftDomain[0, :])

        for x in range(rightDomain.shape[0]):
            if rightDomain[x, self.mutationID] == 0:
                SNPpresent = np.vstack((SNPpresent, rightDomain[x, :]))
                SNPpresentExp = np.vstack(
                    (SNPpresentExp, leftDomain[x, :]))

        # Find all TP53 mutated examples and their gene expressions
        SNPabsent = np.array(rightDomain[0, :])
        SNPabsentExp = np.array(leftDomain[0, :])

        for x in range(rightDomain.shape[0]):
            if rightDomain[x, self.mutationID] == 1:
                SNPabsent = np.vstack((SNPabsent, rightDomain[x, :]))
                SNPabsentExp = np.vstack((SNPabsentExp, leftDomain[x, :]))

        # Pick a random grouping of these examples
        randIndexes = np.random.randint(
            0, min(SNPpresent.shape[0], SNPabsent.shape[0]) - 1,
            (n,))
        SNPpresent = SNPpresent[1:]
        SNPpresentExp = SNPpresentExp[1:]
        SNPpresent = SNPpresent[randIndexes]
        SNPpresentExp = SNPpresentExp[randIndexes]

        SNPabsent = SNPabsent[1:]
        SNPabsentExp = SNPabsentExp[1:]
        SNPabsent = SNPabsent[randIndexes]
        SNPabsentExp = SNPabsentExp[randIndexes]

        # Predict based on the mutations
        (SNPpresentToExp, _) = modelHandle.rightToLeftModel.predict(
                                                              SNPpresent)

        # Find the Mean Squared Error
        SNPpresentToExpMSE = np.square(np.subtract(
            SNPpresentExp, SNPpresentToExp)).mean(axis=0)

        # Create arrays for induced samples
        InducedMutation = copy.copy(SNPpresent)
        InducedMutationExp = copy.copy(SNPpresentExp)

        # Induce a TP53 mutation
        InducedMutation[:, self.mutationID] = 1

        # Predict based on this mutation
        (InducedMutationToExp, _) = modelHandle.rightToLeftModel.predict(
                                                             InducedMutation)

        # Find the Mean Squared Error
        InducedMutationToExpMSE = np.square(np.subtract(
            InducedMutationExp, InducedMutationToExp)).mean(axis=0)

        # Run a t test on the MSE's
        ttest_results = scipy.stats.ttest_ind(
            SNPpresentToExpMSE, InducedMutationToExpMSE)
        t_stat = ttest_results.statistic
        p_val = ttest_results.pvalue

        # Print a table with the t test results
        table_data = dict(values=[[t_stat], [p_val]])
        table_labels = dict(values=['Stat', 'PVal'])
        table = [go.Table(cells=table_data, header=table_labels)]

        py.offline.plot(table, filename=os.path.join('Output',
                                                     params.dataSetInfo.name,
                                                     't_test_{}'.format(
                                                      str(params.outputNum))))
        #######################################################################
        # For Scatter Plot of the differentially expressed genes between
        # real TP53 wildtype vs. real TP53 mutated and
        # real vs. synthetic TP53 data

        # Induce wiltype TP53
        InducedSNPpresent = copy.copy(SNPabsent)
        InducedSNPpresent[:, self.mutationID] = 0

        (InducedSNPpresentToExp, _) = modelHandle.rightToLeftModel.predict(
                                                            InducedSNPpresent)

        # Create y points for the linear regressions
        xPoints = np.repeat(0, n)
        xPoints = np.append(xPoints, np.repeat(1, n))

        # Perform t-tests for all four situations of real and synthetic data
        SyntheticEffect = np.array([])
        SyntheticPVal = np.array([])
        for x in range(leftDomain.shape[1]):
            regressResults = scipy.stats.linregress(
                  xPoints,
                  np.append(SNPpresentExp[:, x], InducedMutationToExp[:, x]))
            SyntheticEffect = np.append(SyntheticEffect, regressResults.slope)
            SyntheticPVal = np.append(SyntheticPVal, regressResults.pvalue)

        RealEffect = np.array([])
        RealPVal = np.array([])
        for x in range(leftDomain.shape[1]):
            regressResults = scipy.stats.linregress(
                  xPoints,
                  np.append(SNPabsentExp[:, x], SNPpresentExp[:, x]))
            RealEffect = np.append(RealEffect, regressResults.slope)
            RealPVal = np.append(RealPVal, regressResults.pvalue)

        SyntheticNotEffect = np.array([])
        SyntheticNotPVal = np.array([])
        for x in range(leftDomain.shape[1]):
            regressResults = scipy.stats.linregress(
                  xPoints,
                  np.append(SNPabsentExp[:, x], InducedSNPpresentToExp[:, x]))
            SyntheticNotEffect = np.append(
                SyntheticNotEffect, regressResults.slope)
            SyntheticNotPVal = np.append(
                SyntheticNotPVal, regressResults.pvalue)

        AllSyntheticEffect = np.array([])
        AllSyntheticPVal = np.array([])
        for x in range(leftDomain.shape[1]):
            regressResults = scipy.stats.linregress(
                  xPoints,
                  np.append(InducedMutationToExp[:, x],
                            InducedSNPpresentToExp[:, x]))
            AllSyntheticEffect = np.append(
                AllSyntheticEffect, regressResults.slope)
            AllSyntheticPVal = np.append(
                AllSyntheticPVal, regressResults.pvalue)

        # Make a scatter plot of the difference between three of the different
        # expression
        # Shows the differnece between the transformed domains and the actual
        # difference in the real data.
        plt.figure()
        plt.title("Scatter Plot of the differentially expressed genes between"
                  "real TP53 wildtype vs. real TP53 mutated and real vs."
                  " synthetic TP53 data")
        plt.scatter(RealEffect, SyntheticEffect, c='r', alpha=0.03,
                    label="Real TP53 Wildtype vs. Synthetic TP53 Mutated")
        plt.scatter(RealEffect, SyntheticNotEffect, c='b', alpha=0.03,
                    label="Real TP53 Mutated vs. Synthetic TP53 Wildtype")
        plt.xlabel("Effect of real TP53 Wildtype vs. real mutated")
        plt.ylabel("Effect of real and synthetic data")
        lgd = plt.legend(ncol=1,
                         bbox_to_anchor=(1.03, 1),
                         loc=2,
                         borderaxespad=0.,
                         fontsize=10)
        plt.savefig(os.path.join('Output', params.dataSetInfo.name,
                                 'Scatter_{}.png'.
                                 format(str(params.outputNum))),
                    bbox_extra_artists=(lgd,),
                    bbox_inches='tight')

        #######################################################################
        # For creating bar graphs of latent space expression between
        # TP53 Wiltype and TP53 mutated

        # Find the difference in the nodes between TP53 wildtyupe and mutated
        # in the right model
        SNPpresentLatent = modelHandle.rightEncoder.predict(SNPpresent)
        SNPabsentLatent = modelHandle.rightEncoder.predict(SNPabsent)

        # Turn this into a percentage difference
        LantentSpacePerc = (abs(SNPpresentLatent - SNPabsentLatent)
                            / abs((SNPabsentLatent + SNPabsentLatent)/2))

        LantentSpacePerc = LantentSpacePerc.mean(axis=0)

        # Create a bar graph showing which nodes are most different
        plt.figure()
        plt.bar(np.arange(LantentSpacePerc.shape[0]), LantentSpacePerc)
        title = plt.title("Percentage difference between TP53 Wildtype and"
                          " mutated for each latent space node, Mutation")
        plt.xticks(np.arange(LantentSpacePerc.shape[0]), np.arange(
            LantentSpacePerc.shape[0]))
        plt.xlabel("Latent Space Nodes")
        plt.ylabel("Latent Space Percentage Difference")
        plt.savefig(os.path.join('Output', params.dataSetInfo.name,
                                 'TP53MutLatentSpaceComparrison_{}.png'.
                                 format(str(params.outputNum))),
                    bbox_extra_artists=(title,),
                    bbox_inches='tight')

        # Repeat the same process of latent space analysis for the left model
        SNPpresentExpLatent = modelHandle.leftEncoder.predict(SNPpresentExp)
        SNPabsentExpLatent = modelHandle.leftEncoder.predict(SNPabsentExp)

        LantentSpacePerc = abs(SNPpresentExpLatent - SNPabsentExpLatent)
        LantentSpacePerc = LantentSpacePerc / abs((SNPpresentExpLatent
                                                   + SNPabsentExpLatent)/2)
        LantentSpacePerc = LantentSpacePerc.mean(axis=0)

        plt.figure()
        plt.bar(np.arange(LantentSpacePerc.shape[0]), LantentSpacePerc)
        title = plt.title("Percentage difference between TP53 Wildtype and"
                          " mutated for each latent space node, Expression")
        plt.xticks(np.arange(LantentSpacePerc.shape[0]), np.arange(
            LantentSpacePerc.shape[0]))
        plt.xlabel("Latent Space Nodes")
        plt.ylabel("Latent Space Percentage Difference")
        plt.savefig(os.path.join('Output', params.dataSetInfo.name,
                                 'TP53ExpLatentSpaceComparrison_{}.png'.
                                 format(str(params.outputNum))),
                    bbox_extra_artists=(title,),
                    bbox_inches='tight')

        #######################################################################
        # Make a latent space of percentage difference between the latente
        # space nodes between the wildtype and mutated

        SNPpresentLatentMean = SNPpresentLatent.mean(axis=0)
        SNPabsentLatentMean = SNPabsentLatent.mean(axis=0)
        SNPpresentExpLatentMean = SNPpresentExpLatent.mean(axis=0)
        SNPabsentExpLatentMean = SNPabsentExpLatent.mean(axis=0)

        # Make a scatter plot of the average present and absent values per node
        plt.figure()
        plt.title("Scatter Plot of the latent space expression of wiltype vs."
                  "mutated")
        plt.scatter(SNPabsentLatentMean, SNPpresentLatentMean, c='r',
                    alpha=0.2,
                    label="Mutation latent space")
        plt.scatter(SNPabsentExpLatentMean, SNPpresentExpLatentMean, c='b',
                    alpha=0.2,
                    label="Expression latent space")
        plt.ylabel("Latent space expression of wiltype")
        plt.xlabel("Latent space expression of mutated")
        lgd = plt.legend(ncol=1,
                         bbox_to_anchor=(1.03, 1),
                         loc=2,
                         borderaxespad=0.,
                         fontsize=10)
        plt.savefig(os.path.join('Output', params.dataSetInfo.name,
                                 'LatentSpaceScatter_{}.png'.
                                 format(str(params.outputNum))),
                    bbox_extra_artists=(lgd,),
                    bbox_inches='tight')

        #######################################################################
        # Create volcano plots for all four of the differential regressions

        plt.figure()
        plt.title("Volcano Plot of the effect and PVal between"
                  " between real TP53"
                  " wildtype vs. real TP53 mutated")
        plt.scatter(RealEffect, -np.log10(RealPVal), c='r', alpha=0.03)
        plt.xlabel("Effect size of real TP53 Wildtype vs. real mutated")
        plt.ylabel("negative log 10 PVal")
        plt.savefig(os.path.join('Output', params.dataSetInfo.name,
                                 'VolcanoReal_{}.png'.
                                 format(str(params.outputNum))),
                    bbox_inches='tight')

        plt.figure()
        plt.title("Volcano Plot of the effect and PVal between real TP53"
                  " wildtype vs. synthetic TP53 mutated")
        plt.scatter(SyntheticEffect, -np.log10(SyntheticPVal), c='r',
                    alpha=0.03)
        plt.xlabel("Effect size of real TP53 Wildtype vs. synthetic mutated")
        plt.ylabel("Negative log 10 PVal")
        plt.savefig(os.path.join('Output', params.dataSetInfo.name,
                                 'VolcanoSynth_{}.png'.
                                 format(str(params.outputNum))),
                    bbox_inches='tight')

        plt.figure()
        plt.title("Volcano Plot of the effect and PVal between synthetic TP53"
                  " wildtype vs. real TP53 mutated")
        plt.scatter(SyntheticNotEffect, -np.log10(SyntheticNotPVal), c='r',
                    alpha=0.03)
        plt.xlabel("Effect size of synthetic TP53 Wildtype vs. real mutated")
        plt.ylabel("Negative log 10 PVal")
        plt.savefig(os.path.join('Output', params.dataSetInfo.name,
                                 'VolcanoSynthNot_{}.png'.
                                 format(str(params.outputNum))),
                    bbox_inches='tight')

        plt.figure()
        plt.title("Volcano Plot of the effect and PVal between synthetic TP53"
                  " wildtype vs. synthetic TP53 mutated")
        plt.scatter(AllSyntheticEffect, -np.log10(AllSyntheticPVal), c='r',
                    alpha=0.03)
        plt.xlabel("Effect size of synthetic TP53 Wildtype vs."
                   " synthetic mutated")
        plt.ylabel("Negative log 10 PVal")
        plt.savefig(os.path.join('Output', params.dataSetInfo.name,
                                 'VolcanoAllSynth_{}.png'.
                                 format(str(params.outputNum))),
                    bbox_inches='tight')
        plt.show()

        #######################################################################
        # Create UMaps of the latent spaces for each model

        # Load the canver labels of the data
        file = os.path.join('Data', 'Cognoma_Data', 'Training',
                            'cancerLabels.csv')
        with open(file, "rb") as fp:
            labels = cPickle.load(fp)

        # Create a UMap of the latent space with the labels in left model
        umap = up.UMAP(n_neighbors=5,
                       min_dist=0.1,
                       metric='correlation')
        randIndexes = np.random.randint(0, leftDomain.shape[0], (n,))
        umap_results = umap.fit_transform(leftPredicted[randIndexes, :])

        plt.figure()
        title = plt.title(params.dataSetInfo.leftDomainName + " Latent Space")

        for g in np.unique(labels[randIndexes]):
            i = np.where(labels[randIndexes] == g)
            plt.scatter(umap_results[i, 0], umap_results[i, 1], label=g)
        plt.legend()
        plt.savefig(os.path.join('Output', 'Cognoma',
                                 'LeftUMap_{}.png'.
                                 format(str(params.outputNum))),
                    bbox_extra_artists=(title,),
                    bbox_inches='tight')

        # Create a UMap of the latent space with the labels in right model
        umap = up.UMAP(n_neighbors=5,
                       min_dist=0.1,
                       metric='correlation')
        randIndexes = np.random.randint(0, rightDomain.shape[0], (n,))
        umap_results = umap.fit_transform(rightPredicted[randIndexes, :])

        plt.figure()
        title = plt.title(params.dataSetInfo.rightDomainName + " Latent Space")
        for g in np.unique(labels[randIndexes]):
            i = np.where(labels[randIndexes] == g)
            plt.scatter(umap_results[i, 0], umap_results[i, 1], label=g)
        plt.legend()
        plt.savefig(os.path.join('Output', 'Cognoma',
                                 'RightUMap_{}.png'.
                                 format(str(params.outputNum))),
                    bbox_extra_artists=(title,),
                    bbox_inches='tight')

        #######################################################################
        # Visualize as images

        # Make the right domain into floats
        rightDomainFloat = rightDomain.astype('float64')
        # Pick a random number of examples to display
        randIndexes = np.random.randint(0, rightDomain.shape[0], (10,))

        # Show the randomly selected samples as images
        plt.figure(figsize=(120, 40))
        for i in range(10):

            # display original mutated
            ax = plt.subplot(6, 10, i + 1)
            plt.imshow(rightDomainFloat[randIndexes[i]].
                       reshape(self.rightXDim,
                               self.rightYDim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Right Domain (Mutated) Truth")

            # display mutated reconstruction
            ax = plt.subplot(6, 10, i + 1 + 10)
            plt.imshow(right_decoded_imgs[randIndexes[i]]
                       .reshape(self.rightXDim,
                                self.rightYDim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Right Domain (Mutated) Reconstructed")

            # display mutated translated into expression
            ax = plt.subplot(6, 10, i + 1 + 2 * 10)
            plt.imshow(rightToLeftImgs[randIndexes[i]]
                       .reshape(self.leftXDim,
                                self.leftYDim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Right Domain (Mutated) translated into Left")

            # display original expression
            ax = plt.subplot(6, 10, i + 1 + 3 * 10)
            plt.imshow(leftDomain[randIndexes[i]].reshape(self.leftXDim,
                                                          self.leftYDim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Left Domain (Expression) Truth")

            # display expression reconstruction
            ax = plt.subplot(6, 10, i + 1 + 4 * 10)
            plt.imshow(left_decoded_imgs[randIndexes[i]]
                       .reshape(self.leftXDim,
                                self.leftYDim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Left Domain (Expression) Reconstructed")

            # display expression translated into mutated
            ax = plt.subplot(6, 10, i + 1 + 5 * 10)
            plt.imshow(leftToRightImgs[randIndexes[i]]
                       .reshape(self.rightXDim,
                                self.rightYDim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if (i == 0):
                ax.set_title("Left Domain (Expression) translated into Right")

        # Save visualization
        plt.savefig(os.path.join('Output', 'Cognoma',
                                 'Visualized_{}.png'.
                                 format(str(params.outputNum))))

        #######################################################################
        # Create Heirical cluster maps for left and right domains comparing
        # real, reconstructed, and transformed data

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

        # Make the column labels for the cluster map for right domain
        index = 0
        for x in range(1, params.inputSizeRight):
            index = np.append(index, x)

        cluster = pd.DataFrame(X.transpose(), index=index, columns=y)

        # Build the color labels for cluster map
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
        title = g.fig.suptitle('Right Domain (Mutated), Original (Blue),'
                               ' Reconstructed (Green),'
                               'and Translated (Purple)'
                               ' Clustermap.')
        g.cax.set_visible(False)

        plt.savefig(os.path.join('Output', 'Cognoma',
                                 'RightHeatmap_{}.png'.
                                 format(str(params.outputNum))),
                    bbox_extra_artists=(title,),
                    bbox_inches='tight')

        # Cluster map for the left domain
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
        title = g.fig.suptitle('Left Domain (Expression), Original (Blue),'
                               ' Reconstructed (Green),'
                               ' and Translated (Purple)'
                               ' Clustermap')

        plt.savefig(os.path.join('Output', 'Cognoma',
                                 'LeftHeatmap_{}.png'.
                                 format(str(params.outputNum))),
                    bbox_extra_artists=(title,),
                    bbox_inches='tight')

        return
