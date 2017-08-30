import numpy as np
from sklearn.metrics.classification import accuracy_score
from DBNClassification import SupervisedDBNClassification as DBN
import timeit

print('Load Leukemia Dataset ...')
# temp = np.load('Preprocessed Dataset/With MI/Lung/MI_0.2_Lung.npy')
temp = np.load('Preprocessed Dataset/Without MI/Data_minmax_Lung.npy')
train_x, test_x, train_y, test_y, listAtributeNames = np.asarray(temp[0]), \
                                                      np.asarray(temp[1]), \
                                                      np.asarray(temp[2]), \
                                                      np.asarray(temp[3]), \
                                                      np.asarray(temp[4])
print('Shape Train : ', train_x.shape)
print('Shape Test : ', test_x.shape)
print('')
print('Training Phase ...')
# Training
start_time = timeit.default_timer()
dbn = DBN(hidden_layers_structure=[200,40],
          learning_rate_rbm=0.04,
          learning_rate=0.04,
          n_epochs_rbm=1000,
          n_iter_backprop=500,
          batch_size=20,
          activation_function='sigmoid',
          dropout_p=0.2)

print('')
dbn.fit(train_x, train_y)
end_time = timeit.default_timer()
print("Running Time : ", (end_time - start_time) / 60.)

print('Testing Phase ...')
# Testing
Y_pred = list(dbn.predict(test_x))
Y_pred = np.asarray(Y_pred)
print('Accuracy: %f' % accuracy_score(test_y, Y_pred))
print("Y_test = ", test_y)
print("Y_pred = ", Y_pred)
print('')
print('Shape Train : ', train_x.shape)
print('Shape Test : ', test_x.shape)
print("Save Model ...")
listWeight, listBias = dbn.getModel()
lr = dbn.learning_rate
epochRBM = dbn.unsupervised_dbn.n_epochs_rbm
epochFT = dbn.n_iter_backprop
arsitektur = dbn.unsupervised_dbn.hidden_layers_structure
msePreTraining = dbn.unsupervised_dbn.getMseRBM()
mseFineTuning = dbn.listMseFT
temp = []
temp.append(listWeight)
temp.append(listBias)
temp.append(lr)
temp.append(epochRBM)
temp.append(epochFT)
temp.append(arsitektur)
temp.append(msePreTraining)
temp.append(mseFineTuning)
location = 'Model/Skenario 1/Lung/S1-200-40 MI ALL LR 0.04.npy'
np.save(location, temp)
print("SAVED : ",location)
