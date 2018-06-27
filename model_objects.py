"""
shared-latent-space/model_objects.py

model_parameters class is used to store the parameters of the model
and information about the dataset. It is passed between
various functions.

model class is used to store the various components of the model for
visulaization inside the specific visualization functions.

Author: Chris Williams
Date: 6/25/18
"""


class model_parameters(object):

    def __init__(self, batchSize, numEpochs, inputSizeLeft,
                 firstLayerSizeLeft,
                 secondLayerSize,
                 thirdLayerSize,
                 encodedSize, inputSizeRight, firstLayerSizeRight, kappa,
                 beta, noise, dropout, notes, dataSetInfo):
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
            kappa (float): The kappa variable for warm start.
            beta (float): Initial Value for Beta variable for warm start.
            noise (float): the amount of noise for input.
            dropoout (float): Amount of dropout to be applied
            notes (string): Any special notes about the run.
            dataSetInfo (dataSetInfoAbstract): Data set info

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

        self.kappa = kappa
        self.beta = beta
        self.noise = noise
        self.dropout = dropout

        self.notes = notes

        self.dataSetInfo = dataSetInfo

        # Stored for generating unqiue names for the current run
        self.outputNum = 0


class model(object):

    def __init__(self, leftEncoder, rightEncoder,
                 leftDecoder, rightDecoder,
                 leftToRightModel, rightToLeftModel,
                 leftModel, rightModel):
        """
        Hold the various compenents of the model.

        Args:
            leftEncoder (Keras model): Left Encoder.
            rightEncoder (Keras model): Right Encoder.
            leftDecoder (Keras model): Left Decoder.
            rightDecoder (Keras model): Right Decoder.
            leftToRightModel (Keras model): Left to Right Model.
            rightToLeftModel (Keras model): Right to Left Model.
            leftModel (Keras model): Left VAE Model.
            rightModel (Keras model): Right VAE Model.

        Returns: None
        """
        self.leftEncoder = leftEncoder
        self.rightEncoder = rightEncoder
        self.leftDecoder = leftDecoder
        self.rightDecoder = rightDecoder
        self.leftToRightModel = leftToRightModel
        self.rightToLeftModel = rightToLeftModel
        self.leftModel = leftModel
        self.rightModel = rightModel
