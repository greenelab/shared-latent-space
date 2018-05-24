"""
shared-latent-space/model_objects.py

This class is used to store the parameters of the model
and information about the dataset. It is passed between
various functions.

Author: Chris Williams
Date: 5/22/18
"""


class model_parameters(object):

    def __init__(self, batchSize, numEpochs, inputSizeLeft,
                 firstLayerSizeLeft, secondLayerSize, thirdLayerSize,
                 encodedSize, inputSizeRight, firstLayerSizeRight,
                 dataSetInfo):
        """
        Takes in parameters of the model and data set info.

        Args:
            batchSize (int): The size of batches.
            numEpochs (int): Number of trainning epochs.
            inputSizeLeft (int): Size of the left domain.
            firstLayerSizeLeft (int): Size of the first layer of the left.
            secondLayerSize (int): Size of the second layer (Shared).
            thirdLayerSize (int): Size of the third layer (Shared).
            encodedSize (int): Size of latent space (Shared).
            inputSizeRight (int): Size of the right domain.
            firstLayerSizeRight (int): Size of the first layer of the right.
            dataSetInfo (DataSetInfoAbstractClass): Data specific functions.

        Returns: None
        """
        self.batchSize = batchSize
        self.numEpochs = numEpochs
        self.inputSizeLeft = inputSizeLeft
        self.firstLayerSizeLeft = firstLayerSizeLeft

        self.secondLayerSize = secondLayerSize
        self.thirdLayerSize = thirdLayerSize
        self.encodedSize = encodedSize

        self.inputSizeRight = inputSizeRight
        self.firstLayerSizeRight = firstLayerSizeRight

        self.dataSetInfo = dataSetInfo
