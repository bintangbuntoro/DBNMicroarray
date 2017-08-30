import os

import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest


# from sklearn.feature_selection import mutual_info_classif


# Breast Cancer Dataset
def getBreastCancerDataset():
    print("Starting to load Breast Cancer Dataset")
    dir = os.path.abspath(os.path.dirname(__file__) + "\\Dataset\\breast")
    fileTrain = dir + "\\breastCancer_train.data"
    fileTest = dir + "\\breastCancer_test.data"
    atribut = dir + "\\breastCancer.names"

    # Load Train Data & Label
    listTrainData, listTrainClass = loadDataFromFile(fileTrain)

    # Load Test Data & Label
    listTestData, listTestClass = loadDataFromFile(fileTest)

    # Load Atribute Names
    listAtributeNames = loadAtributesData(atribut, 2)

    # Convert label to binary
    listBinaryTrainLabel = []
    for i in listTrainClass:
        if i == 'relapse':
            listBinaryTrainLabel.append(0)
        else:
            listBinaryTrainLabel.append(1)
    listBinaryTestLabel = []
    for i in listTestClass:
        if i == 'relapse':
            listBinaryTestLabel.append(0)
        else:
            listBinaryTestLabel.append(1)

    # Storing Training Data
    print('=== Load data training ===')
    train_x = np.asarray(listTrainData)
    train_y = np.asarray(listBinaryTrainLabel)
    print('Train Data Cleansing ...')
    train_x = cleansingData(train_x, 100)

    # Storing Test Data
    print('=== Load data testing ===')
    test_x = np.asarray(listTestData)
    test_y = np.asarray(listBinaryTestLabel)
    print('Test Data Cleansing ...')
    test_x = cleansingData(test_x, 100)

    return train_x, test_x, train_y, test_y, listAtributeNames


# Colon Tumor Dataset
def getColonTumorDataset():
    print("Starting to load Colon Tumor Dataset")
    dir = os.path.abspath(os.path.dirname(__file__) + "\\Dataset\\colon")
    fileTrain = dir + "\\colonTumor.data"
    atribut = dir + "\\colonTumor.names"

    listData = []
    listClass = []
    listAtributeNames = []
    listTrainData = []
    listTestData = []
    train_y = []
    test_y = []

    # Load Train Data & Label
    listData, listClass = loadDataFromFile(fileTrain)

    # Load Atribute Names
    listAtributeNames = loadAtributesData(atribut, 2)

    # Convert label to binary
    listBinaryTrainLabel = []
    for i in listClass:
        if i == 'positive':
            listBinaryTrainLabel.append(0)
        else:
            listBinaryTrainLabel.append(1)

    msk = np.random.rand(len(listData)) < 0.8
    for item in range(0, len(msk)):
        if msk[item] == True:
            listTrainData.append(listData[item])
            train_y.append(listBinaryTrainLabel[item])
        else:
            listTestData.append(listData[item])
            test_y.append(listBinaryTrainLabel[item])

    # Storing Training Data

    print('Load data training')
    train_x = np.asarray(listTrainData)

    # Storing Test Data
    print('Load data testing')
    test_x = np.asarray(listTestData)

    return train_x, test_x, train_y, test_y, listAtributeNames


# Leukemia Dataset
def getLeukemiaDataset():
    print("Starting to load Leukemia MLL Dataset")
    dir = os.path.abspath(os.path.dirname(__file__) + "\\Dataset\\leukemia")
    fileTrain = dir + "\\AMLALL_train.data"
    fileTest = dir + "\\AMLALL_test.data"
    atribut = dir + "\\AMLALL.names"

    listTrainData = []
    listTrainClass = []
    listTestData = []
    listTestClass = []
    listAtributeNames = []

    # Load Train Data & Label
    listTrainData, listTrainClass = loadDataFromFile(fileTrain)

    # Load Test Data & Label
    listTestData, listTestClass = loadDataFromFile(fileTest)

    # Load Atribute Names
    listAtributeNames = loadAtributesData(atribut, 4)

    # Convert label to binary
    listBinaryTrainLabel = []
    for i in listTrainClass:
        if i == 'ALL':
            listBinaryTrainLabel.append(0)
        elif i == 'AML':
            listBinaryTrainLabel.append(1)
    listBinaryTestLabel = []
    for i in listTestClass:
        if i == 'ALL':
            listBinaryTestLabel.append(0)
        elif i == 'AML':
            listBinaryTestLabel.append(1)

    # Storing Training Data

    print('Load data training')
    train_x = np.asarray(listTrainData)
    train_y = np.asarray(listBinaryTrainLabel)

    # Storing Test Data
    print('Load data testing')
    test_x = np.asarray(listTestData)
    test_y = np.asarray(listBinaryTestLabel)

    return train_x, test_x, train_y, test_y, listAtributeNames


# Lung Cancer Dataset
def getLungDataset():
    print("Starting to load Lung Cancer Dataset")
    dir = os.path.abspath(os.path.dirname(__file__) + "\\Dataset\\lung")
    fileTrain = dir + "\\lungCancer_train.data"
    fileTest = dir + "\\lungCancer_test.data"
    atribut = dir + "\\lungCancer_train.names"

    listTrainData = []
    listTrainClass = []
    listTestData = []
    listTestClass = []
    listAtributeNames = []

    # Load Train Data & Label
    listTrainData, listTrainClass = loadDataFromFile(fileTrain)

    # Load Test Data & Label
    listTestData, listTestClass = loadDataFromFile(fileTest)

    # Load Atribute Names
    listAtributeNames = loadAtributesData(atribut, 2)

    # Convert label to binary
    listBinaryTrainLabel = []
    for i in listTrainClass:
        if i == 'Mesothelioma':
            listBinaryTrainLabel.append(0)
        else:
            listBinaryTrainLabel.append(1)
    listBinaryTestLabel = []
    for i in listTestClass:
        if i == 'Mesothelioma':
            listBinaryTestLabel.append(0)
        else:
            listBinaryTestLabel.append(1)

    # Storing Training Data
    print('Load data training')
    train_x = np.asarray(listTrainData)
    train_y = np.asarray(listBinaryTrainLabel)

    # Storing Test Data
    print('Load data testing')
    test_x = np.asarray(listTestData)
    test_y = np.asarray(listBinaryTestLabel)

    return train_x, test_x, train_y, test_y, listAtributeNames


# Mesothelioma

# Ovarian Cancer Dataset
def getOvarianDataset():
    print("Starting to load Ovarian Cancer Dataset")
    dir = os.path.abspath(os.path.dirname(__file__) + "\\Dataset\\ovarian")
    fileTrain = dir + "\\ovarian_61902.data"
    atribut = dir + "\\ovarian_61902.names"

    listData = []
    listClass = []
    listAtributeNames = []
    listTrainData = []
    listTestData = []
    train_y = []
    test_y = []

    # Load Train Data & Label
    listData, listClass = loadDataFromFile(fileTrain)

    # Load Atribute Names
    listAtributeNames = loadAtributesData(atribut, 2)

    # Convert label to binary
    listBinaryTrainLabel = []
    for i in listClass:
        if i == 'Cancer':
            listBinaryTrainLabel.append(0)
        else:
            listBinaryTrainLabel.append(1)

    msk = np.random.rand(len(listData)) < 0.8
    for item in range(0, len(msk)):
        if msk[item] == True:
            listTrainData.append(listData[item])
            train_y.append(listBinaryTrainLabel[item])
        else:
            listTestData.append(listData[item])
            test_y.append(listBinaryTrainLabel[item])

    # Storing Training Data

    print('Load data training')
    train_x = np.asarray(listTrainData)

    # Storing Test Data
    print('Load data testing')
    test_x = np.asarray(listTestData)

    return train_x, test_x, train_y, test_y, listAtributeNames


def cleansingData(arr, parClean):
    idxMisVal = []
    idxMisValAll = []
    temp = arr
    for j in range(0, arr.shape[1]):
        if np.all(arr[:, j] == parClean):
            idxMisValAll.append(j)
        elif np.any(arr[:, j] == parClean):
            idxMisVal.append(j)

    for i in idxMisVal:
        totData = 0
        denominator = 0
        for j in arr[:, i]:
            if j != parClean:
                totData = totData + j
                denominator = denominator + 1
        mn = totData / denominator
        for k in range(0, arr.shape[0]):
            if arr[k, i] == parClean:
                arr[k, i] = mn
        totData = 0
        denominator = 0
        mn = 0

    for j in range(0, len(idxMisValAll)):
        temp = np.delete(temp, np.s_[idxMisValAll[j]], 1)
        idxMisValAll[:] = [x - 1 for x in idxMisValAll]

    arr = temp

    return arr


def minmaxNormalization(train, test, nmax, nmin):
    # Min Max Normalization
    tp_train = np.transpose(train)
    tp_test = np.transpose(test)
    newmax = nmax
    newmin = nmin
    listNormTrain = []
    listNormTest = []

    for item in range(0, tp_train.shape[0]):
        maxData = np.amax(tp_train[item])
        minData = np.amin(tp_train[item])
        trainNorm = []
        testNorm = []
        # Data Train
        for i in tp_train[item]:
            newdataTrain = (i - minData) * (newmax - newmin) / (maxData - minData) + newmin
            #            newdataTrain = round(newdataTrain,2)
            newdataTrain = float(format(newdataTrain, '.3f'))
            trainNorm.append(newdataTrain)
        # Data Test
        for j in tp_test[item]:
            newdataTest = (j - minData) * (newmax - newmin) / (maxData - minData) + newmin
            #            newdataTest = round(newdataTest,2)
            newdataTest = float(format(newdataTest, '.3f'))
            testNorm.append(newdataTest)

        listNormTrain.append(trainNorm)
        listNormTest.append(testNorm)

    listNormTrain = np.asarray(listNormTrain)
    listNormTrain = np.transpose(listNormTrain)
    listNormTest = np.asarray(listNormTest)
    listNormTest = np.transpose(listNormTest)

    return listNormTrain, listNormTest


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


# def MutualInformation(x, y, dimention):
#    fsx = SelectKBest(mutual_info_classif, k=dimention).fit_transform(x, y)
#
#    ttrain_x = np.transpose(x)
#    tfstrain_x = np.asarray(np.transpose(fsx))
#    temp = []
#    for i in range(0, ttrain_x.shape[0]):
#        for j in range(0, tfstrain_x.shape[0]):
#            if np.array_equal(ttrain_x[i], tfstrain_x[j]):
#                temp.append(i)
#    return fsx, temp


# def getDataWithMI(dataset, dimention):
#    if dataset is 'breast':
#        train_x, test_x, train_y, test_y, listAtributeNames = getBreastCancerDataset()
#        print('MI on train data ...')
#        train_x, MItrain = MutualInformation(train_x, train_y, dimention)
#        print('MI on test data ...')
#        temp = []
#        ttest_x = np.transpose(test_x)
#        for i in MItrain:
#            temp.append(ttest_x[i])
#        test_x = np.transpose(np.asarray(temp))
#        print('Normalizing Data ...')
#        train_x, test_x = minmaxNormalization(train_x, test_x, 0.1, 0.9)
#        return train_x, test_x, train_y, test_y, listAtributeNames, MItrain
#    elif dataset is 'leukemia':
#        train_x, test_x, train_y, test_y, listAtributeNames = getLeukemiaDataset()
#        print('MI on train data ...')
#        train_x, MItrain = MutualInformation(train_x, train_y, dimention)
#        print('MI on test data ...')
#        temp = []
#        ttest_x = np.transpose(test_x)
#        for i in MItrain:
#            temp.append(ttest_x[i])
#        test_x = np.transpose(np.asarray(temp))
#        print('Normalizing Data ...')
#        train_x, test_x = minmaxNormalization(train_x, test_x, 0.1, 0.9)
#        return train_x, test_x, train_y, test_y, listAtributeNames, MItrain
#    elif dataset is 'lung':
#        train_x, test_x, train_y, test_y, listAtributeNames = getLungDataset()
#        print('MI on train data ...')
#        train_x, MItrain = MutualInformation(train_x, train_y, dimention)
#        print('MI on test data ...')
#        temp = []
#        ttest_x = np.transpose(test_x)
#        for i in MItrain:
#            temp.append(ttest_x[i])
#        test_x = np.transpose(np.asarray(temp))
#        print('Normalizing Data ...')
#        train_x, test_x = minmaxNormalization(train_x, test_x, 0.1, 0.9)
#        return train_x, test_x, train_y, test_y, listAtributeNames, MItrain
#    elif dataset is 'colon':
#        train_x, test_x, train_y, test_y, listAtributeNames = getColonTumorDataset()
#        print('MI on train data ...')
#        train_x, MItrain = MutualInformation(train_x, train_y, dimention)
#        print('MI on test data ...')
#        temp = []
#        ttest_x = np.transpose(test_x)
#        for i in MItrain:
#            temp.append(ttest_x[i])
#        test_x = np.transpose(np.asarray(temp))
#        print('Normalizing Data ...')
#        train_x, test_x = minmaxNormalization(train_x, test_x, 0.1, 0.9)
#        return train_x, test_x, train_y, test_y, listAtributeNames, MItrain
#    elif dataset is 'ovarian':
#        train_x, test_x, train_y, test_y, listAtributeNames = getOvarianDataset()
#        print('MI on train data ...')
#        train_x, MItrain = MutualInformation(train_x, train_y, dimention)
#        print('MI on test data ...')
#        temp = []
#        ttest_x = np.transpose(test_x)
#        for i in MItrain:
#            temp.append(ttest_x[i])
#        test_x = np.transpose(np.asarray(temp))
#        print('Normalizing Data ...')
#        train_x, test_x = minmaxNormalization(train_x, test_x, 0.1, 0.9)
#        return train_x, test_x, train_y, test_y, listAtributeNames, MItrain


def getDataNormalize(dataset):
    if dataset is 'breast':
        train_x, test_x, train_y, test_y, listAtributeNames = getBreastCancerDataset()
        print('Normalizing Data ...')
        train_x, test_x = minmaxNormalization(train_x, test_x, 0.1, 0.9)
        return train_x, test_x, train_y, test_y, listAtributeNames
    elif dataset is 'leukemia':
        train_x, test_x, train_y, test_y, listAtributeNames = getLeukemiaDataset()
        print('Normalizing Data ...')
        train_x, test_x = minmaxNormalization(train_x, test_x, 0.1, 0.9)
        return train_x, test_x, train_y, test_y, listAtributeNames
    elif dataset is 'lung':
        train_x, test_x, train_y, test_y, listAtributeNames = getLungDataset()
        print('Normalizing Data ...')
        train_x, test_x = minmaxNormalization(train_x, test_x, 0.1, 0.9)
        return train_x, test_x, train_y, test_y, listAtributeNames
    elif dataset is 'colon':
        train_x, test_x, train_y, test_y, listAtributeNames = getColonTumorDataset()
        print('Normalizing Data ...')
        train_x, test_x = minmaxNormalization(train_x, test_x, 0.1, 0.9)
        return train_x, test_x, train_y, test_y, listAtributeNames
    elif dataset is 'ovarian':
        train_x, test_x, train_y, test_y, listAtributeNames = getOvarianDataset()
        print('Normalizing Data ...')
        train_x, test_x = minmaxNormalization(train_x, test_x, 0.1, 0.9)
        return train_x, test_x, train_y, test_y, listAtributeNames
