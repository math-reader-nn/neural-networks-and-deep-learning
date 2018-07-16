### Demo of different methods of combining networks.

import numpy as np
import scipy.signal as sg

import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10
net = Network([FullyConnectedLayer(n_in=784, n_out=100),SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)

    # inmat = matrix[i].reshape(28,28,order='F')
    # con = [[0,1,0],[1,-4,1],[0,1,0]]
    # outmat = sg.convolve2d(inmat,con,mode='same')
    # regularize relative to np.linalg.norm(outmat)

