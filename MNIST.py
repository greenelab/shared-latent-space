import matplotlib.pyplot as plt
import numpy as np
from model_objects import model_parameters
from scipy import misc
from keras.datasets import mnist
from DataSetInfoAbstractClass import dataSetInfoAbstract
import seaborn as sns

class dataInfo(dataSetInfoAbstract):

    def __init__(self):
        self.name = "MNIST"

    def load(self):
        # Loading the MNIST Data
        (x_train, _), (x_test, _) = mnist.load_data()

        # Formating the MNIST Data
        x_train = x_train.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

        # Formating the MNIST Data
        x_test = x_test.astype('float32') / 255.
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

        # Making Copies of the Data to creat Inverses
        import copy

        a_train = copy.copy(x_train)
        for i in range(len(a_train)):
            a_train[i] = 1 - a_train[i]

        a_test = copy.copy(x_test)
        for i in range(len(a_test)):
            a_test[i] = 1 - a_test[i]

        return (x_train, a_train, x_test, a_test)

    def visualize(self, randIndexes, rightDomain, right_decoded_imgs, rightToLeftCycle, right_generatedImgs, leftToRightImgs,
                  leftDomain, left_decoded_imgs, leftToRightCycle, left_generatedImgs, rightToLeftImgs, Xdim, YDim, params):
        # Display the Original, Reconstructed, Transformed, Cycle, and
        # Generated data for both the regular and invserve MNIST data
        n = 10  # how many digits we will display
        plt.figure()
        for i in range(n):
            # display reg original
            ax = plt.subplot(12, n, i + 1)
            plt.imshow(leftDomain[randIndexes[i]].reshape(Xdim, YDim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reg reconstruction
            ax = plt.subplot(12, n, i + 1 + n)
            plt.imshow(left_decoded_imgs[randIndexes[i]].reshape(Xdim, YDim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display left to right transformed
            ax = plt.subplot(12, n, i + 1 + 2 * n)
            plt.imshow(leftToRightImgs[randIndexes[i]].reshape(Xdim, YDim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display left to right transformed cycled through
            ax = plt.subplot(12, n, i + 1 + 3 * n)
            plt.imshow(leftToRightCycle[randIndexes[i]].reshape(Xdim, YDim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reg generated
            ax = plt.subplot(12, n, i + 1 + 4 * n)
            plt.imshow(left_generatedImgs[randIndexes[i]].reshape(Xdim, YDim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display inv original
            ax = plt.subplot(12, n, i + 1 + 5 * n)
            plt.imshow(rightDomain[randIndexes[i]].reshape(Xdim, YDim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display inv reconstruction
            ax = plt.subplot(12, n, i + 1 + 6 * n)
            plt.imshow(right_decoded_imgs[randIndexes[i]].reshape(Xdim, YDim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display right to left transformed
            ax = plt.subplot(12, n, i + 1 + 7 * n)
            plt.imshow(rightToLeftImgs[randIndexes[i]].reshape(Xdim, YDim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display right to left transformed cycled through
            ax = plt.subplot(12, n, i + 1 + 8 * n)
            plt.imshow(rightToLeftCycle[randIndexes[i]].reshape(Xdim, YDim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display inv generated
            ax = plt.subplot(12, n, i + 1 + 9 * n)
            plt.imshow(right_generatedImgs[randIndexes[i]].reshape(Xdim, YDim))
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
