import numpy as np
import DBNTesting as test
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

K = 2
listAkurasi = []
listF1score = []
for i in range(1,K+1):
    temp = np.load('Model/Skenario 4/Ovarian/S4-200-40 MI 0.1 LR 0.01 K-'+str(i)+'.npy')
    train_x, train_y, test_x, test_y, listWeight, listBias, lr, epochRBM, epochFT, \
    arsitektur, msePreTraining, mseFineTuning = np.asarray(temp[0]), \
                                                np.asarray(temp[1]), \
                                                np.asarray(temp[2]), \
                                                np.asarray(temp[3]), \
                                                np.asarray(temp[4]), \
                                                np.asarray(temp[5]), \
                                                np.asarray(temp[6]), \
                                                np.asarray(temp[7]), \
                                                np.asarray(temp[8]), \
                                                np.asarray(temp[9]), \
                                                np.asarray(temp[10]), \
                                                np.asarray(temp[11])

    # Testing
    Y_pred = list(test.predict(test_x, test_y, listWeight, listBias))
    Y_pred = np.asarray(Y_pred)
    print("Y_test = ", test_y)
    print("Y_pred = ", Y_pred)
    accuracy = accuracy_score(test_y, Y_pred)
    f1 = f1_score(test_y, Y_pred, average='macro')
    print("Akurasi = ", accuracy)
    print('F1-Macro: %f' % f1)
    print (classification_report(test_y, Y_pred))
    print("Learning Rate ", lr)
    listAkurasi.append(accuracy)
    listF1score.append(f1)

print()
print("Performansi Colon Menggunakan K-Fold")
print("Mean Accuracy = ",np.mean(listAkurasi))
print("Mean F1-Score = ",np.mean(listF1score))
