import numpy as np
from loadDataset import minmaxNormalization
import MI_Calculate as mi

def MIselection(dataset,threshold):
    print('Load Pure Data ...')
    strLoad = 'Preprocessed Dataset/Data Pure/Data_'+dataset+'.npy'
    strSave = 'Preprocessed Dataset/With MI/'+dataset+'/MI_'+str(threshold)+'_'+dataset
#    temp = np.load('Preprocessed Dataset/Data Pure/Data_Ovarian.npy')
    temp = np.load(strLoad)
    train_x, test_x, train_y, test_y, listAtributeNames= np.asarray(temp[0]), \
                                                                   np.asarray(temp[1]), \
                                                                   np.asarray(temp[2]), \
                                                                   np.asarray(temp[3]), \
                                                                   np.asarray(temp[4])
    print('Mutual Information Process ...')
    mi_array = mi.computeMIall(train_x,train_y)
    train_x_tranpose = np.transpose(train_x)
    test_x_tranpose = np.transpose(test_x)
    listMIselected = []
    listMIValue = []
    for i in range(0,len(mi_array)):
        if mi_array[i] >= threshold:
            listMIselected.append(i)
            listMIValue.append(mi_array[i])
            
    train_x_tranpose = train_x_tranpose[listMIselected]
    test_x_tranpose = test_x_tranpose[listMIselected]
    train_x_MI = np.transpose(train_x_tranpose)
    test_x_MI = np.transpose(test_x_tranpose)
    print('Normalizing Data ...')
    train_x, test_x = minmaxNormalization(train_x_MI, test_x_MI, 0.1, 0.9)
    
    temp = []
    temp.append(train_x)
    temp.append(test_x)
    temp.append(train_y)
    temp.append(test_y)
    temp.append(listAtributeNames)
    np.save(strSave, temp)
    
    return mi_array, listMIselected,listMIValue
    
mi_array, listMIselected,listMIValue = MIselection('Ovarian',0.05)

