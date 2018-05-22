import matplotlib.pyplot as plt
import numpy as np
from model_objects import model_parameters
from scipy import misc
from keras.datasets import mnist
from DataSetInfoAbstractClass import dataSetInfoAbstract
import cPickle


class dataInfo(dataSetInfoAbstract):

    def __init__(self):
        self.name = "MNIST"
        self.Xdim = 28
        self.Ydim = 28

    def load(self):
        # Loading the MNIST Data
        with open("Data/MNIST_Data/Training/MNIST_Training.pkl", "rb") as openfileobject:
            (x_train, a_train) = cPickle.load(openfileobject)

            with open("Data/MNIST_Data/Testing/MNIST_Testing.pkl", "rb") as openfileobject:
                (x_test, a_test) = cPickle.load(openfileobject)
        return (x_train, a_train, x_test, a_test)

    def visualize(self, randIndexes, rightDomain, right_decoded_imgs, rightToLeftCycle, right_generatedImgs, leftToRightImgs,
                  leftDomain, left_decoded_imgs, leftToRightCycle, left_generatedImgs, rightToLeftImgs, params):
        # Display the Original, Reconstructed, Transformed, Cycle, and
        # Generated data for both the regular and invserve MNIST data
        n = 10  # how many digits we will display
        plt.figure()
        for i in range(n):
            # display reg original
            ax = plt.subplot(12, n, i + 1)
            plt.imshow(leftDomain[randIndexes[i]].reshape(self.Xdim, self.Ydim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reg reconstruction
            ax = plt.subplot(12, n, i + 1 + n)
            plt.imshow(left_decoded_imgs[randIndexes[i]].reshape(self.Xdim, self.Ydim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display left to right transformed
            ax = plt.subplot(12, n, i + 1 + 2 * n)
            plt.imshow(leftToRightImgs[randIndexes[i]].reshape(self.Xdim, self.Ydim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display left to right transformed cycled through
            ax = plt.subplot(12, n, i + 1 + 3 * n)
            plt.imshow(leftToRightCycle[randIndexes[i]].reshape(self.Xdim, self.Ydim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reg generated
            ax = plt.subplot(12, n, i + 1 + 4 * n)
            plt.imshow(left_generatedImgs[randIndexes[i]].reshape(self.Xdim, self.Ydim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display inv original
            ax = plt.subplot(12, n, i + 1 + 5 * n)
            plt.imshow(rightDomain[randIndexes[i]].reshape(self.Xdim, self.Ydim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display inv reconstruction
            ax = plt.subplot(12, n, i + 1 + 6 * n)
            plt.imshow(right_decoded_imgs[randIndexes[i]].reshape(self.Xdim, self.Ydim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display right to left transformed
            ax = plt.subplot(12, n, i + 1 + 7 * n)
            plt.imshow(rightToLeftImgs[randIndexes[i]].reshape(self.Xdim, self.Ydim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display right to left transformed cycled through
            ax = plt.subplot(12, n, i + 1 + 8 * n)
            plt.imshow(rightToLeftCycle[randIndexes[i]].reshape(self.Xdim, self.Ydim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display inv generated
            ax = plt.subplot(12, n, i + 1 + 9 * n)
            plt.imshow(right_generatedImgs[randIndexes[i]].reshape(self.Xdim, self.Ydim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig('Output/MNIST/Output_' + str(params.numEpochs) + '_' + str(params.firstLayerSizeLeft) + '_' + str(params.inputSizeLeft) + '_'
                    + str(params.secondLayerSize) + '_' + str(params.thirdLayerSize) + '_' +
                    str(params.encodedSize) + '_' + str(params.firstLayerSizeRight) +
                    '_' + str(params.inputSizeRight)
                    + '.png')
        plt.show()
        return
