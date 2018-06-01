import os


for firstLayerSizeLeft in range(512, 1025, 128):
    for secondLayerSize in range(512, 513, 1):
        for thirdLayerSize in range(128, 513, 128):
            for encodedSize in range(128, 257, 128):
                for firstLayerSizeRight in range(512, 1025, 128):
                    cmd = "python main_file.py --data Cognoma --batchSize 60 --numEpochs 50 --firstLayerSizeLeft {} --secondLayerSize {} --thirdLayerSize {} --encodedSize {} --firstLayerSizeRight {}".format(str(firstLayerSizeLeft),
                                                                                                                                         str(secondLayerSize),
                                                                                                                                         str(thirdLayerSize),
                                                                                                                                         str(encodedSize),
                                                                                                                                         str(firstLayerSizeRight))
                    # print cmd
                    os.system(cmd)
