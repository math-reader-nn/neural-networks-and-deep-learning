import mnist_loader as mnist
training_data, validation_data, test_data = mnist.load_data_wrapper()

import pyximport; pyximport.install()

import network2

import numpy as np

"""A zero denominator means a test point was far enough from all the data that the spreading value was not large enough to push the weight associated with any training point above machine precision."""

#network2.GRNN(test_data,training_data,1.2) # accuracy 9630/10000 
#network2.GRNN(test_data,training_data,1.0) # accuracy 9662/10000 
results_list2 = network2.GRNN(test_data,training_data,0.6) # accuracy 9673/10000 best so far
#network2.GRNN(test_data,training_data,0.5) # accuracy 9670/10000 
#network2.GRNN(test_data,training_data,0.1) # accuracy 4003/10000 one zero denom was encountered

#network2.GRNN(test_data,training_data[:1000],0.7) # accuracy /10000 
#network2.GRNNfast(test_data, test_mat_in, test_mat_out, tr_mat_in[:,:1000], tr_mat_out[:,:1000], 1.0)

# This sets up the mesh needed, etc.

mat_in = np.asarray([point[0].reshape((len(point[0]),)) for point in training_data])
mat_out = np.asarray([point[1].reshape((len(point[1]),)) for point in training_data])

mesh_sizes = (150,80,40)

mesh = [network2.meshify(mat_in,mat_out,size) for size in mesh_sizes]

gamma = network2.connect(mat_in,mesh,mesh_sizes)

results_list = network2.GRNNfast_lim(test_data,mat_in,mat_out,.85,mesh,gamma)