import numpy as np
from sklearn.metrics.classification import accuracy_score
from DBNClassification import SupervisedDBNClassification as DBN
import timeit
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

print('Load Colon Dataset ...')
temp = np.load('Preprocessed Dataset/Without MI/Data_minmax_Colon_Full.npy')
# temp = np.load('Preprocessed Dataset/With MI/Colon/MI_0.1_Colon_Full.npy')
# temp = np.load('Preprocessed Dataset/Data Pure/Data_Colon_Full.npy')
listData, listLabel, listAtributeNames = np.asarray(temp[0]), \
                                         np.asarray(temp[1]), \
                                         np.asarray(temp[2])

print('Training Phase ...')
# Training
start_time = timeit.default_timer()
dbn = DBN(hidden_layers_structure=[200,40],
          learning_rate_rbm=0.01,
          learning_rate=0.01,
          n_epochs_rbm=20000,  # 20000
          n_iter_backprop=1000,  # 1000
          batch_size=20,
          activation_function='sigmoid',
          dropout_p=0.2)

# K-Fold
kf = KFold(n_splits=2)
kf.get_n_splits(listData)
k = 0
for train_index, test_index in kf.split(listData):
    print("TRAIN:", train_index, "TEST:", test_index)
    train_x, test_x = listData[train_index], listData[test_index]
    train_y, test_y = listLabel[train_index], listLabel[test_index]

    dbn.fit(train_x, train_y)
    end_time = timeit.default_timer()
    print("Running Time : ", (end_time - start_time) / 60.)

    print('Testing Phase ...')
    # Testing
    Y_pred = list(dbn.predict(test_x))
    Y_pred = np.asarray(Y_pred)
    print('Done.\nAccuracy: %f' % accuracy_score(test_y, Y_pred))
    print('F1-Macro: %f' % f1_score(test_y, Y_pred, average='micro'))
    print("Y_test = ", test_y)
    print("Y_pred = ", Y_pred)

    print("Save Model ...")
    listWeight, listBias = dbn.getModel()
    lr = dbn.learning_rate
    epochRBM = dbn.unsupervised_dbn.n_epochs_rbm
    epochFT = dbn.n_iter_backprop
    arsitektur = dbn.unsupervised_dbn.hidden_layers_structure
    msePreTraining = dbn.unsupervised_dbn.getMseRBM()
    mseFineTuning = dbn.listMseFT
    temp = []
    temp.append(train_x)
    temp.append(train_y)
    temp.append(test_x)
    temp.append(test_y)
    temp.append(listWeight)
    temp.append(listBias)
    temp.append(lr)
    temp.append(epochRBM)
    temp.append(epochFT)
    temp.append(arsitektur)
    temp.append(msePreTraining)
    temp.append(mseFineTuning)
    location = 'Model/Skenario 4/Colon/S4-200-40 MI ALL LR 0.01 K-'+str(k+1)+'.npy'
    k = k+1
    np.save(location, temp)
    print("SAVED : ", location)
