import matplotlib.pyplot as plt
import numpy as np
from model_objects import model_parameters
from skimage.transform import resize
from scipy import misc
from DataSetInfoAbstractClass import dataSetInfoAbstract


class dataInfo(dataSetInfoAbstract):

    def __init__(self):
        self.name = "ICVL"

    def load(self):

        with open("Data/ICVL_Data/Training/Annotation_Training.csv", "r") as openfileobject:
            temp = [openfileobject.readline().strip().split(',')]
            for line in openfileobject:
                temp.append(line.strip().split(','))

        x_train = np.array(temp).astype('float32')
        print x_train.shape

        x_train = np.array(x_train).astype('float32')
        x_train = x_train + abs(np.min(x_train))
        x_train = x_train / np.max(x_train)

        import os
        output = []
        for filename in sorted(os.listdir('Data/ICVL_Data/Training/depth/')):
            filename = 'Data/ICVL_Data/Training/depth/' + filename
            temp = misc.imread(filename)
            temp = resize(temp, (temp.shape[0] / 4, temp.shape[1] / 4))
            temp = np.array(temp)
            temp = temp.flatten()

            output.append(temp)

        a_train = np.array(output).astype('float32') / np.max(output)
        print a_train.shape

        x_test = x_train
        a_test = a_train

        return (x_test, a_train, x_test, a_test)

    def draw_hands(self, Xs, Ys):
        plt.scatter(Xs, Ys)

        plt.gca().invert_yaxis()

        linesStart = np.array([Xs[0], Xs[1]])
        linesEnd = np.array([Ys[0], Ys[1]])
        plt.plot(linesStart, linesEnd)

        linesStart = np.array([Xs[1], Xs[2]])
        linesEnd = np.array([Ys[1], Ys[2]])
        plt.plot(linesStart, linesEnd, 'g')

        linesStart = np.array([Xs[2], Xs[3]])
        linesEnd = np.array([Ys[2], Ys[3]])
        plt.plot(linesStart, linesEnd, 'g')

        linesStart = np.array([Xs[3], Xs[16]])
        linesEnd = np.array([Ys[3], Ys[16]])
        plt.plot(linesStart, linesEnd, 'g')

        linesStart = np.array([Xs[0], Xs[4]])
        linesEnd = np.array([Ys[0], Ys[4]])
        plt.plot(linesStart, linesEnd, 'r')

        linesStart = np.array([Xs[4], Xs[5]])
        linesEnd = np.array([Ys[4], Ys[5]])
        plt.plot(linesStart, linesEnd, 'r')

        linesStart = np.array([Xs[5], Xs[6]])
        linesEnd = np.array([Ys[5], Ys[6]])
        plt.plot(linesStart, linesEnd, 'r')

        linesStart = np.array([Xs[6], Xs[17]])
        linesEnd = np.array([Ys[6], Ys[17]])
        plt.plot(linesStart, linesEnd, 'r')

        linesStart = np.array([Xs[0], Xs[7]])
        linesEnd = np.array([Ys[0], Ys[7]])
        plt.plot(linesStart, linesEnd, 'm')

        linesStart = np.array([Xs[7], Xs[8]])
        linesEnd = np.array([Ys[7], Ys[8]])
        plt.plot(linesStart, linesEnd, 'm')

        linesStart = np.array([Xs[8], Xs[9]])
        linesEnd = np.array([Ys[8], Ys[9]])
        plt.plot(linesStart, linesEnd, 'm')

        linesStart = np.array([Xs[9], Xs[18]])
        linesEnd = np.array([Ys[9], Ys[18]])
        plt.plot(linesStart, linesEnd, 'm')

        linesStart = np.array([Xs[0], Xs[10]])
        linesEnd = np.array([Ys[0], Ys[10]])
        plt.plot(linesStart, linesEnd, 'y')

        linesStart = np.array([Xs[10], Xs[11]])
        linesEnd = np.array([Ys[10], Ys[11]])
        plt.plot(linesStart, linesEnd, 'y')

        linesStart = np.array([Xs[11], Xs[12]])
        linesEnd = np.array([Ys[11], Ys[12]])
        plt.plot(linesStart, linesEnd, 'y')

        linesStart = np.array([Xs[12], Xs[19]])
        linesEnd = np.array([Ys[12], Ys[19]])
        plt.plot(linesStart, linesEnd, 'y')

        linesStart = np.array([Xs[0], Xs[13]])
        linesEnd = np.array([Ys[0], Ys[13]])
        plt.plot(linesStart, linesEnd, 'b')

        linesStart = np.array([Xs[13], Xs[14]])
        linesEnd = np.array([Ys[13], Ys[14]])
        plt.plot(linesStart, linesEnd, 'b')

        linesStart = np.array([Xs[14], Xs[15]])
        linesEnd = np.array([Ys[14], Ys[15]])
        plt.plot(linesStart, linesEnd, 'b')

        linesStart = np.array([Xs[15], Xs[20]])
        linesEnd = np.array([Ys[15], Ys[20]])
        plt.plot(linesStart, linesEnd, 'b')

    def visualize(self, randIndexes, rightDomain, right_decoded_imgs, rightToLeftCycle, right_generatedImgs, leftToRightImgs,
                  leftDomain, left_decoded_imgs, leftToRightCycle, left_generatedImgs, rightToLeftImgs, Xdim, YDim, params):
        n = 10  # how many digits we will display
        plt.figure(figsize=(120, 40))
        for i in range(n):

            # display inv original
            ax = plt.subplot(5, n, i + 1)
            plt.imshow(rightDomain[randIndexes[i]].reshape(Xdim, YDim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display inv reconstruction
            ax = plt.subplot(5, n, i + 1 + n)
            plt.imshow(right_decoded_imgs[randIndexes[i]].reshape(Xdim, YDim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display right to left transformed cycled through
            ax = plt.subplot(5, n, i + 1 + 2 * n)
            plt.imshow(rightToLeftCycle[randIndexes[i]].reshape(Xdim, YDim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display inv generated
            ax = plt.subplot(5, n, i + 1 + 3 * n)
            plt.imshow(right_generatedImgs[randIndexes[i]].reshape(Xdim, YDim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display left to right transformed
            ax = plt.subplot(5, n, i + 1 + 4 * n)
            plt.imshow(leftToRightImgs[randIndexes[i]].reshape(Xdim, YDim))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig('Output/ICVL/Right_' + str(params.numEpochs) + '_' + str(params.firstLayerSizeLeft) + '_' + str(params.inputSizeLeft) + '_'
                    + str(params.thirdLayerSize) + '_' + str(params.secondLayerSize) + '_' +
                    str(params.encodedSize) + '_' + str(params.firstLayerSizeRight) +
                    '_' + str(params.inputSizeRight)
                    + '.png')
        n = 10  # how many digits we will display
        plt.figure(figsize=(120, 40))
        for i in range(n):
            Xs = np.array(leftDomain[randIndexes[i]][0::3])
            Ys = np.array(leftDomain[randIndexes[i]][1::3])
            ax = plt.subplot(5, n, i + 1)
            self.draw_hands(Xs, Ys)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display inv reconstruction
            ax = plt.subplot(5, n, i + 1 + n)
            Xs = np.array(left_decoded_imgs[randIndexes[i]][0::3])
            Ys = np.array(left_decoded_imgs[randIndexes[i]][1::3])
            self.draw_hands(Xs, Ys)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display right to left transformed cycled through
            ax = plt.subplot(5, n, i + 1 + 2 * n)
            Xs = np.array(leftToRightCycle[randIndexes[i]][0::3])
            Ys = np.array(leftToRightCycle[randIndexes[i]][1::3])
            self.draw_hands(Xs, Ys)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display inv generated
            ax = plt.subplot(5, n, i + 1 + 3 * n)
            Xs = np.array(left_generatedImgs[randIndexes[i]][0::3])
            Ys = np.array(left_generatedImgs[randIndexes[i]][1::3])
            self.draw_hands(Xs, Ys)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display left to right transformed
            ax = plt.subplot(5, n, i + 1 + 4 * n)
            Xs = np.array(rightToLeftImgs[randIndexes[i]][0::3])
            Ys = np.array(rightToLeftImgs[randIndexes[i]][1::3])
            self.draw_hands(Xs, Ys)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig('Output/ICVL/Left' + str(params.numEpochs) + '_' + str(params.firstLayerSizeLeft) + '_' + str(params.inputSizeLeft) + '_'
                    + str(params.thirdLayerSize) + '_' + str(params.secondLayerSize) + '_' +
                    str(params.encodedSize) + '_' + str(params.firstLayerSizeRight) +
                    '_' + str(params.inputSizeRight)
                    + '.png')
        plt.show()
        return
