import numpy as np
import DBNTesting as test
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


def leukemia():
    temp = np.load('Preprocessed Dataset/With MI/Leukemia/MI_0.1_Leukemia.npy')
    train_x, test_x, train_y, test_y, listAtributeNames = np.asarray(temp[0]), \
                                                          np.asarray(temp[1]), \
                                                          np.asarray(temp[2]), \
                                                          np.asarray(temp[3]), \
                                                          np.asarray(temp[4])

    temp2 = np.load('Model/Skenario 4/Leukemia/S4-50-10 MI 0.1 LR 0.01.npy')
    listWeight, listBias, lr, epochRBM, epochFT, \
    arsitektur, msePreTraining, mseFineTuning = np.asarray(temp2[0]), \
                                                np.asarray(temp2[1]), \
                                                np.asarray(temp2[2]), \
                                                np.asarray(temp2[3]), \
                                                np.asarray(temp2[4]), \
                                                np.asarray(temp2[5]), \
                                                np.asarray(temp2[6]), \
                                                np.asarray(temp2[7])

    # Testing
    Y_pred = list(test.predict(test_x, test_y, listWeight, listBias))
    Y_pred = np.asarray(Y_pred)
    # print('Done.\nAccuracy: %f' % accuracy_score(test_y, Y_pred))
    print("Y_test = ", test_y)
    print("Y_pred = ", Y_pred)
    accuracy = accuracy_score(test_y, Y_pred)
    f1 = f1_score(test_y, Y_pred, average='macro')
    print("Akurasi = ", accuracy)
    print("F1-Score = ", f1)

    return listWeight, listBias, lr, epochRBM, epochFT, arsitektur, msePreTraining, mseFineTuning, train_x, test_x, train_y, test_y, listAtributeNames

listWeight, listBias, lr, epochRBM, epochFT, arsitektur, msePreTraining, mseFineTuning, train_x, test_x, train_y, test_y, listAtributeNames = leukemia()
#Y_test = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
#Y_pred = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 1 1 1 1 1 0 1 1 1"
#Y_test = Y_test.split()
#Y_pred = Y_pred.split()
#accuracy = accuracy_score(Y_test, Y_pred)
#f1 = f1_score(Y_test, Y_pred, average='macro')
#print("Akurasi = ", accuracy)
#print("F1-Score = ", f1)
