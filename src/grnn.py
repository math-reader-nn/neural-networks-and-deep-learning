# Standard Libraries

import datetime

# Third-party libraries
import numpy as np

def GRNN(test_data, training_data, sigma, classifier = True):
    """ Computes outputs for a set of training data using general regression. """
    correct = 0
    error = 0
    results_list = []
    print "Start date and time: " , datetime.datetime.now()

    spreading = 1 / (2*sigma*sigma)
    
    output_example = training_data[0][1]
    output_shape = output_example.shape
    
    for test_point in test_data:
        
        num = np.zeros_like(output_example)
        den = 0
        for stored_point in training_data:
            distance_squared = np.dot((test_point[0]-stored_point[0]).T,test_point[0]-stored_point[0])[0][0]
            t = distance_squared * spreading
            weight = np.exp(-t)
            den += weight
            num += weight * stored_point[1]
        
        """Since we're just comparing values, we don't need to normalize by the
        sum of the weights."""
        if classifier:
            correct += int(np.argmax(num/den)==test_point[1])
        else:
            error += np.sqrt(np.dot((test_point[1]-num/den).T,test_point[1]-num/den))
        results_list.append((num/den).reshape(output_shape))
    if classifier:
        print "Accuracy of GRNN: {0} / {1}".format(correct,len(test_data))
        print "Logarithmic accuracy of combined network: {0:.4f}\n".format(-np.log((len(test_data)-correct)/(len(test_data)+0.0)))
    else:
        print ""
    print "End date and time: " , datetime.datetime.now()
    return results_list


def GRNNfast(test_data, test_mat_in, test_mat_out, tr_mat_in, tr_mat_out, sigma):
    """" Implements GRNN using unified matrix operations, in the interest of increased performance (but in practice I haven't found the performance to be better)."""
    print "Start date and time: " , datetime.datetime.now()
    correct = 0
    spreading = 1 / (2*sigma*sigma)
    ones = np.ones((1,len(tr_mat_in[0])),type(0.0))
    data_size = len(tr_mat_in[0])
    for test_point in test_data:
#        print "Date and time 1: " , datetime.datetime.now()
#        diff_mat = tr_mat_in - test_point[0].dot(ones)   # .1974 per 10000
#        diff_mat = tr_mat_in - np.tile(test_point[0],(1,data_size)) # .1364 per 10000
        diff_mat = tr_mat_in - np.repeat(test_point[0],data_size,1) # .1362 per 10000
#        print "Date and time 2: " , datetime.datetime.now()
        dist_v = np.asarray((diff_mat*diff_mat).sum(0))     # .081 per 10000
#        print "Date and time 3: " , datetime.datetime.now()
        t = dist_v * spreading                              # .0002 per 10000
#        print "Date and time 4: " , datetime.datetime.now()
        weights = np.exp(-t)                                # .0062 per 10000
#        print "Date and time 5: " , datetime.datetime.now()
        den = weights.sum(0)                                # .0002 per 10000
#        print "Date and time 6: " , datetime.datetime.now()
        num = tr_mat_out.dot(weights)                       # .0008 per 10000
#        print "Date and time 7: " , datetime.datetime.now()
        correct += int(np.argmax(num/den)==test_point[1])   # .0002 per 10000
#        print "Date and time 8: " , datetime.datetime.now()
    print "Accuracy of GRNN: {0} / {1}".format(correct,len(test_data))
    print "Logarithmic accuracy of network: {0:.4f}\n".format(-np.log((len(test_data)-correct)/(len(test_data)+0.0)))
    print "End date and time: " , datetime.datetime.now()

def GRNNfast_lim(test_data, tr_mat_in, tr_mat_out, sigma, mesh, gamma):
    """ Implements GRNN on a subset of the data using a mesh. """
    print "Start date and time: " , datetime.datetime.now()
    correct = 0
    spreading = 1 / (2*sigma*sigma)
    results_list = []
    for test_point in test_data:
        num = 0
        den = 0
        traversal_set = mesh[0]
        for i in range(len(mesh)):
            minpoint = None
            mindist = float('inf')
            for j in traversal_set:
                diff = test_point[0] - tr_mat_in[j].reshape(test_point[0].shape)
                d = diff.T.dot(diff)
                if d < mindist:
                    mindist = d
                    minpoint = mesh[i].index(j)
                t = d * spreading
                den += np.exp(-t)
                num += tr_mat_out[j] * np.exp(-t)
#            if i+1<len(mesh): 
                traversal_set = gamma[i][minpoint]
        for j in traversal_set:
            diff = test_point[0] - tr_mat_in[j].reshape(test_point[0].shape)
            d = diff.T.dot(diff)
            t = d * spreading
            den += np.exp(-t)
            num += tr_mat_out[j] * np.exp(-t)
        correct += int(np.argmax(num/den)==test_point[1])
        results_list.append((num/den).reshape((10,1)))
    print "Accuracy of GRNN: {0} / {1}".format(correct,len(test_data))
    print "Logarithmic accuracy of network: {0:.4f}\n".format(-np.log((len(test_data)-correct)/(len(test_data)+0.0)))
    print "End date and time: " , datetime.datetime.now()
    return results_list

def data_diameter_mesh(data):
    """ Find maximum (square of the) distance between points in the data set, as well as the average (square of the) distance from any point to its nearest neighbor.  This data is useful for computing mesh sizes."""
    curr_max = 0
    data = data.T
    min_v = np.repeat(float('inf'),len(data))
    for i in range(len(data)):
        for j in range(i+1,len(data)):# comparing each training point pair
            diff_v = data[i]-data[j]
            d = diff_v.T.dot(diff_v)# square of the distance between points
            curr_max = max(curr_max,d)
            if d>0:
                min_v[i] = min(min_v[i],d)
                min_v[j] = min(min_v[j],d)
    curr_min = np.mean(min_v)
    return curr_max,curr_min

def convert_data_to_matrix(data):
    """ The standard format  of mnist is not in matrix form.  For GRNNfast we need matrices, so this function converts them. """
    mat_in = np.zeros((len(data[0][0]),len(data)))
    mat_out = np.zeros((len(data[0][1]),len(data)))
    for i in range(len(data)):
        mat_in[:,i] = np.reshape(data[i][0],(len(data[0][0]),))
        mat_out[:,i] = np.reshape(data[i][1],(len(data[0][1]),))
    return mat_in,mat_out
    
def meshify(mat_in,mat_out,mesh_size):
    """ Finds a maximal subset of the untagged training set no two of which are closer (squarewise) than mesh_size and stores their indices """
    print "Start date and time: " , datetime.datetime.now()
    mesh = []
    indexer = range(len(mat_in))
    random.shuffle(indexer)
    #for datum in mat_in:
    for i in indexer:
        too_close = False
        for j in mesh:
            if (mat_in[j]-mat_in[i]).T.dot(mat_in[j]-mat_in[i]) < mesh_size:
                too_close = True
                break
        if not too_close:
            mesh.append(i) # For now we don't store the data, just indices
    print "End date and time: " , datetime.datetime.now()
    return mesh

def remesh(mat_in, oldmesh):
    """ Converts a mesh that was stored as vectors to a mesh stored as indices. """
    newmesh = []
    for meshlayer in oldmesh:
        newmeshlayer = []
        for point in meshlayer:
            for i in range(len(mat_in)):
                if np.array_equal(mat_in[i],point):
                    newmeshlayer.append(i)
                    break
        newmesh.append(newmeshlayer)
    return newmesh

def connect(mat_in,mesh,mesh_sizes):
    """ Connects adjacent layers of a tiered mesh. """
    gamma = []
    for m, subm, size in zip(mesh[:-1],mesh[1:],mesh_sizes[:-1]):
        gammalayer = [[] for j in m]
        for i in subm:
#            minpoint = None
#            mindist = float('inf')
#            for j in m:
#                diff = mat_in[i]-mat_in[j]
#                d = diff.T.dot(diff)
#                if d < mindist:
#                    minindex = j
#                    mindist = d
            for j in m:
                diff = mat_in[i]-mat_in[j]
                d = diff.T.dot(diff)
                if d < size:
                    gammalayer[m.index(j)].append(subm[subm.index(i)])
#            gammalayer.append(minindex)
        gamma.append(gammalayer)
    gammalayer = [[] for j in mesh[-1]]
    for i in range(len(mat_in)):
        for j in mesh[-1]:
                diff = mat_in[i]-mat_in[j]
                d = diff.T.dot(diff)
                if d < mesh_sizes[-1]:
                    gammalayer[mesh[-1].index(j)].append(i)
    gamma.append(gammalayer)        
    return gamma