from abc import ABCMeta, abstractmethod


class dataSetInfoAbstract(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def load(self):
        return

    @abstractmethod
    def visualize(self, randIndexes, rightDomain, right_decoded_imgs, rightToLeftCycle, right_generatedImgs, leftToRightImgs,
                  leftDomain, left_decoded_imgs, leftToRightCycle, left_generatedImgs, rightToLeftImgs, Xdim, YDim, params):
        return
