# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 23:46:34 2017

@author: BintangBuntoro
"""
import numpy as np
from scipy.special import digamma
from sklearn import neighbors


def _compute_mi_cd(c, d, n_neighbors):
    """Compute mutual information between continuous and discrete variables.

    Parameters
    ----------
    c : ndarray, shape (n_samples,)
        Samples of a continuous random variable.

    d : ndarray, shape (n_samples,)
        Samples of a discrete random variable.

    n_neighbors : int
        Number of nearest neighbors to search for each point, see [1]_.

    Returns
    -------
    mi : float
        Estimated mutual information. If it turned out to be negative it is
        replace by 0.

    Notes
    -----
    True mutual information can't be negative. If its estimate by a numerical
    method is negative, it means (providing the method is adequate) that the
    mutual information is close to 0 and replacing it by 0 is a reasonable
    strategy.

    References
    ----------
    .. [1] B. C. Ross "Mutual Information between Discrete and Continuous
       Data Sets". PLoS ONE 9(2), 2014.
    """
    n_samples = c.shape[0]
    c = c.reshape((-1, 1))

    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples)
    k_all = np.empty(n_samples)
    nn = neighbors.NearestNeighbors()
    for label in np.unique(d):
        mask = d == label
        count = np.sum(mask)
        if count > 1:
            k = min(n_neighbors, count - 1)
            nn.set_params(n_neighbors=k)
            nn.fit(c[mask])
            r = nn.kneighbors()[0]
            # Memasukan nilai terbesar dari hasil KNN pada var radius
            radius[mask] = np.nextafter(r[:, -1], 0)
            k_all[mask] = k
        label_counts[mask] = count

    # Ignore points with unique labels.
    mask = label_counts > 1
    n_samples = np.sum(mask)
    label_counts = label_counts[mask]
    k_all = k_all[mask]
    c = c[mask]
    radius = radius[mask]

    nn.set_params(algorithm='kd_tree')
    nn.fit(c)
    # Mencari tetangga terdekat berdasarkan radius hasil KNN
    ind = nn.radius_neighbors(radius=radius, return_distance=False)
    # Memberikan nilai balik sejumlah berapa tetangga yg dimiliki pada tiap data
    m_all = np.array([i.size for i in ind])

    mi = (digamma(n_samples) + np.mean(digamma(k_all)) -
          np.mean(digamma(label_counts)) -
          np.mean(digamma(m_all + 1)))

    result = max(0, mi)
    return result


def computeMIall(data_x, data_y):
    mi_all = []
    for i in range(0, data_x[0, :].shape[0]):
        result = _compute_mi_cd(data_x[:, i], data_y, 3)
        mi_all.append(result)

    return mi_all


#print('Load Pure Data ...')
#temp = np.load('Preprocessed Dataset/Data Pure/Data_Colon.npy')
#train_x, test_x, train_y, test_y, listAtributeNames = np.asarray(temp[0]), \
#                                                      np.asarray(temp[1]), \
#                                                      np.asarray(temp[2]), \
#                                                      np.asarray(temp[3]), \
#                                                      np.asarray(temp[4])
#print('Mutual Information Process ...')
#mi_array = computeMIall(train_x, train_y)
#mi_sorted = np.sort(mi_array)
#listMIselected = []
#for i in range(0,len(mi_array)):
#    if mi_array[i] >= 0.1:
#        listMIselected.append(i)
        
#mi_pertama = _compute_mi_cd(train_x[:,0], train_y, 3)
#print(mi_pertama)

#c = train_x[:,0]
#n_neighbors = 3
#d = train_y
#
#n_samples = c.shape[0]
#c = c.reshape((-1, 1))
#
#radius = np.empty(n_samples)
#label_counts = np.empty(n_samples)
#k_all = np.empty(n_samples)
#nn = neighbors.NearestNeighbors()
#for label in np.unique(d):
#    mask = d == label
#    count = np.sum(mask)
#    print(count)
#    if count > 1:
#        k = min(n_neighbors, count - 1)
#        print(k)
#        k=3
#        nn.set_params(n_neighbors=k)
#        nn.fit(c[mask])
#        r = nn.kneighbors()[0]
#        radius[mask] = np.nextafter(r[:, -1], 0)
#        k_all[mask] = k
#    label_counts[mask] = count
#
## Ignore points with unique labels.
#mask = label_counts > 1
#n_samples = np.sum(mask)
#label_counts = label_counts[mask]
#k_all = k_all[mask]
#c = c[mask]
#radius = radius[mask]
#
#nn.set_params(algorithm='kd_tree')
#nn.fit(c)
#ind = nn.radius_neighbors(radius=radius, return_distance=False)
#m_all = np.array([i.size for i in ind])
#
#mi = (digamma(n_samples) + np.mean(digamma(k_all)) -
#      np.mean(digamma(label_counts)) -
#      np.mean(digamma(m_all + 1)))

#result = max(0, mi)
#print(result)
