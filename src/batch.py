### Demo of different methods of combining networks.

import mnist_loader as mnist
training_data, validation_data, test_data = mnist.load_data_wrapper()

import network2

def reportcombine(networks, data):
    combo = network2.combine(networks)
    for net,i in zip(networks,range(len(networks))):
        print "Accuracy of network {0}: {1} / {2}".format(i,net.accuracy(data),len(data))
        print "Logarithmic accuracy of network {0}: {1:.4f}\n".format(i,net.score(data))
    print "Accuracy of combined network: {0} / {1}".format(combo.accuracy(data),len(data))
    print "Logarithmic accuracy of combined network: {0:.4f}\n".format(combo.score(data))
    
def reportcombine2(networks, data):
    combo = network2.combine2(networks)
    print "Accuracy of combination of single digit networks: {0} / {1}".format(combo.accuracy(data),len(data))
    print "Logarithmic accuracy of combination of single digit networks: {0:.4f}\n".format(combo.score(data))    

### Demo of combining networks.  3 networks of 30 neuron hidden layers.
networks = [network2.Network([784, 30, 10]) for i in range(3)]
for net,i in zip(networks,range(len(networks))): # Not much training done here, only 3 epochs, no regularization.
    print "Training network {0}:".format(i)
    net.SGD(training_data, 3, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)

reportcombine(networks, validation_data)

### High powered demo.  3 highly tuned networks of 100 neuron hidden layers.  Takes a long time, uncomment to run.
# networks2 = [network2.Network([784, 100, 10]) for i in range(3)]
# for net,i in zip(networks2,range(len(networks2))): # Serious training: this will take a while.
#     print "Training network {0}:".format(i)
#     net.SGD(training_data, 30, 10, 0.5, lmbda=5.0, evaluation_data=test_data, monitor_evaluation_accuracy=True)
#     net.SGD(training_data, 60, 10, 0.1, lmbda=5.0, evaluation_data=test_data, monitor_evaluation_accuracy=True)
# 
# reportcombine(networks2, validation_data)

### Demo of training digits individually then combining into network.

# First create specialized training data sets for individual digits...
straining_data = [[[training_datum[0],[training_datum[1][i]]] for training_datum in training_data] for i in range(10)]

# ...then create networks with single neuron output layers...
snet = [network2.Network([784,30,1]) for i in range(10) ]

# ...now train them.  Light training, 3 epochs each, no regularization.  
# Accuracy measurements aren't working yet for these specialized networks, so it's
# hard to tell how good they are on their own (probably pretty good).
for net,data,i in zip(snet,straining_data,range(len(snet))):
    print "Training network for digit {0}:".format(i)
    net.SGD(data, 3, 10, 0.5)

reportcombine2(snet,validation_data)