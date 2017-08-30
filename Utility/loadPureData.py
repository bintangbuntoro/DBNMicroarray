# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 02:20:15 2017

@author: BintangBuntoro
"""
import numpy as np
import loadDataset as ld

# train_x, test_x, train_y, test_y, listAtributeNames = ld.getLeukemiaDataset()

train_x, test_x, train_y, test_y, listAtributeNames = ld.getColonTumorDataset()

temp = []
temp.append(train_x)
temp.append(test_x)
temp.append(train_y)
temp.append(test_y)
temp.append(listAtributeNames)
np.save('Preprocessed Dataset/Data Pure/Data_Colon', temp)

#temp = np.load('Data/MI_minmax_Leukemia_3000.npy')
#train_x, test_x, train_y, test_y, listAtributeNames, MItrain = np.asarray(temp[0]), \
#                                                               np.asarray(temp[1]), \
#                                                               np.asarray(temp[2]), \
#                                                               np.asarray(temp[3]), \
#                                                               np.asarray(temp[4]), \
#                                                               np.asarray(temp[5])
