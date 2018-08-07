### Demo of most successful networks for coffee data
import network2
import coffee_loader

# Load data, compute some useful metrics
training_data,validation_data,test_data = coffee_loader.load_data()

tr_av = sum([datum[1] for datum in training_data])/float(len(training_data))
tr_er = sum([abs(datum[1]-tr_av) for datum in training_data])/float(len(training_data))
ts_av = sum([datum[1] for datum in test_data])/float(len(test_data))
ts_er = sum([abs(datum[1]-tr_av) for datum in test_data])/float(len(test_data))

# Create network and set regularization parameter

Sigmoid = network2.Sigmoid
Probabilistic = network2.Probabilistic
Linear = network2.Linear

coffee = network2.Network([21,15,15,1],[Probabilistic,Probabilistic,Linear])
lmbda = 0.01

# Alias some useful functions from network2

tr_avg_error = lambda: network2.avg_error(training_data,coffee)
tr_cost = lambda: coffee.total_cost(training_data,lmbda=lmbda)
ts_avg_error = lambda: network2.avg_error(test_data,coffee)
ts_cost = lambda: coffee.total_cost(test_data,lmbda=lmbda)

# Make lists to track development of network
tr_errors = []
tr_costs = []
ts_errors = []
ts_costs = []

def report_error(string,list):
    print list[-4:]

def report_cost(string,list):
    print list[-4:]

def train(training_data=training_data, epochs=1000, mini_batch_size=60, eta=0.5, lmbda = lmbda, max_momentum = 10, tr_errors = [], tr_costs = [], ts_errors = [], ts_costs = []):
    coffee.SGD(training_data, epochs, mini_batch_size, eta, lmbda, max_momentum = max_momentum, silent=True)
    tr_errors.append(tr_avg_error())
    tr_costs.append(tr_cost())
    ts_errors.append(ts_avg_error())
    ts_costs.append(ts_cost())
    report_error("",tr_errors[-4:])
    report_cost("",tr_costs[-4:])
    report_error("",ts_errors[-4:])
    report_cost("",ts_costs[-4:])

# Train 1500 sets of 1000 epochs, lowering eta when the cost isn't trending downward.
eta = 1.0
min_cost = float('inf')
delay = 0

for i in range(1500):
    print i
    train(training_data=training_data, epochs=1000, mini_batch_size=60, eta=eta, lmbda = lmbda, tr_errors = tr_errors, tr_costs = tr_costs, ts_errors = ts_errors, ts_costs = ts_costs)
    if min(tr_costs[-5:]) > min_cost and delay > 5:
        eta = eta*0.5
        delay = 0
        print "Reducing eta to "+str(eta)
    if tr_costs[-1] < min_cost:
        min_cost = tr_costs[-1]
        delay = 0
    delay = delay + 1