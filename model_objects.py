class model_parameters(object):

    def __init__(self, batchSize, numEpochs, firstLayerSizeLeft,
                 inputSizeLeft, secondLayerSize, thirdLayerSize,
                 encodedSize, firstLayerSizeRight,
                 inputSizeRight, dataSetInfo):
        self.batchSize = batchSize
        self.numEpochs = numEpochs
        self.firstLayerSizeLeft = firstLayerSizeLeft
        self.inputSizeLeft = inputSizeLeft

        self.secondLayerSize = secondLayerSize
        self.thirdLayerSize = thirdLayerSize
        self.encodedSize = encodedSize

        self.firstLayerSizeRight = firstLayerSizeRight
        self.inputSizeRight = inputSizeRight

        self.dataSetInfo = dataSetInfo
