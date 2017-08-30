import os
import numpy as np
import MI_Calculate as mi


# Colon Tumor Dataset
def getColonTumorDataset():
    print("Starting to load Colon Tumor Dataset")
    dir = os.path.abspath(os.path.dirname(__file__) + "\\Dataset\\colon")
    fileTrain = dir + "\\colonTumor.data"
    atribut = dir + "\\colonTumor.names"

    listData = []
    listClass = []
    listAtributeNames = []

    # Load Train Data & Label
    listData, listClass = loadDataFromFile(fileTrain)

    # Load Atribute Names
    listAtributeNames = loadAtributesData(atribut, 2)

    # Convert label to binary
    listLabel = []
    for i in listClass:
        if i == 'positive':
            listLabel.append(0)
        else:
            listLabel.append(1)

    listData = np.asarray(listData)
    listLabel = np.asarray(listLabel)

    return listData, listLabel, listAtributeNames

# Ovarian Cancer Dataset
def getOvarianDataset():
    print("Starting to load Ovarian Cancer Dataset")
    dir = os.path.abspath(os.path.dirname(__file__) + "\\Dataset\\ovarian")
    fileTrain = dir + "\\ovarian_61902.data"
    atribut = dir + "\\ovarian_61902.names"

    listData = []
    listClass = []
    listAtributeNames = []

    # Load Train Data & Label
    listData, listClass = loadDataFromFile(fileTrain)

    # Load Atribute Names
    listAtributeNames = loadAtributesData(atribut, 2)

    # Convert label to binary
    listLabel = []
    for i in listClass:
        if i == 'Cancer':
            listLabel.append(0)
        else:
            listLabel.append(1)

    listData = np.asarray(listData)
    listLabel = np.asarray(listLabel)

    return listData, listLabel, listAtributeNames

def loadDataFromFile(fileDir):
    f = open(fileDir, 'r')

    listSample = []
    listLabel = []

    # Load Sample & Label
    for i, word in enumerate(f):
        temp = word.split(',')
        cls = temp[-1].replace('\n', '')
        temp.remove(temp[-1])
        if temp != []:
            temp = [float(i) for i in temp]
            listSample.append(temp)
            listLabel.append(cls)
    f.close()
    return listSample, listLabel


def loadAtributesData(fileDir, startPoint):
    f = open(fileDir, 'r')

    listAtributeNames = []

    for j, word in enumerate(f):
        if j >= startPoint:
            word = word.replace('\n', '')
            word = word.rstrip()
            if word != '':
                listAtributeNames.append(word)
    f.close()
    return listAtributeNames


def minmaxNormalization(train, nmax, nmin):
    # Min Max Normalization
    tp_train = np.transpose(train)
    newmax = nmax
    newmin = nmin
    listNormTrain = []

    for item in range(0, tp_train.shape[0]):
        maxData = np.amax(tp_train[item])
        minData = np.amin(tp_train[item])
        trainNorm = []
        # Data Train
        for i in tp_train[item]:
            newdataTrain = (i - minData) * (newmax - newmin) / (maxData - minData) + newmin
            #            newdataTrain = round(newdataTrain,2)
            newdataTrain = float(format(newdataTrain, '.3f'))
            trainNorm.append(newdataTrain)

        listNormTrain.append(trainNorm)

    listNormTrain = np.asarray(listNormTrain)
    listNormTrain = np.transpose(listNormTrain)

    return listNormTrain


def MIselection(dataset, threshold):
    print('Load Pure Data ...')
    strLoad = 'Preprocessed Dataset/Data Pure/Data_' + dataset + '_Full.npy'
    strSave = 'Preprocessed Dataset/With MI/' + dataset + '/MI_' + str(threshold) + '_' + dataset + '_Full.npy'
    temp = np.load(strLoad)
    listData, listLabel, listAtributeNames = np.asarray(temp[0]), \
                                            np.asarray(temp[1]), \
                                            np.asarray(temp[2])
    print('Mutual Information Process ...')
    mi_array = mi.computeMIall(listData, listLabel)
    train_x_tranpose = np.transpose(listData)
    listMIselected = []
    listMIValue = []
    for i in range(0, len(mi_array)):
        if mi_array[i] >= threshold:
            listMIselected.append(i)
            listMIValue.append(mi_array[i])

    train_x_tranpose = train_x_tranpose[listMIselected]
    train_x_MI = np.transpose(train_x_tranpose)
    print('Normalizing Data ...')
    datasetMI = minmaxNormalization(train_x_MI, 0.1, 0.9)

    temp = []
    temp.append(datasetMI)
    temp.append(listLabel)
    temp.append(listAtributeNames)
    np.save(strSave, temp)

    return mi_array, listMIselected, listMIValue, datasetMI

mi_array, listMIselected, listMIValue, datasetMI = MIselection('Ovarian',0.1)
#strLoad = 'Preprocessed Dataset/Data Pure/Data_Colon_Full.npy'
#temp = np.load(strLoad)
#listData, listLabel, listAtributeNames = np.asarray(temp[0]), \
#                                            np.asarray(temp[1]), \
#                                            np.asarray(temp[2]), \

#listData, listLabel, listAtributeNames = getOvarianDataset()