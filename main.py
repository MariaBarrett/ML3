from __future__ import division
import numpy as np
import pylab as plt
from OurSVM import *
from nn import *


"""
SVM runs by itself
"""

"""
Running our NN.
"""

train, test = loadFiles()

inpt, hidden, output = 2, 20, 1 #Change here to see the output with 2 hidden neurons
neurons, weights = initNeuronsAndWeights(inpt, hidden, output)

origNeurons = np.copy(neurons)
origWeights = np.copy(weights)

errors = []
learn = [0.01,0.001,0.0001]
for lrn in learn:
	temperrors, temptesterrors = [], []
	neurons = np.copy(origNeurons)
	weights = np.copy(origWeights)

	for i in np.arange(1000): #Change this for fewer epsilons
		totalError = 0
		testError = 0
		deltaWeights = np.zeros((len(weights),1))
	
		for t in train:
			forwardPropogation(t[0], neurons,weights)
			deltaWeights, error = backwardsPropogation(t[1], neurons, weights, deltaWeights)
			totalError += error**2

		#do the same for the test
		for te in test:
			forwardPropogation(te[0], neurons, weights)
			testError += (neurons[2][0][-1] - te[1])**2
	
		totalError /= len(train)
		testError /= len(test)
		temperrors.append([i, totalError])
		temptesterrors.append([i, testError])

		#update weights
		weights = updateWeights(weights, deltaWeights,learningRate=lrn)

	if lrn == 0.01:
		graph = []
		for i in np.arange(-10,10,0.05):
			forwardPropogation(i, neurons, weights)
			graph.append([i, neurons[2][0][-1]])

		graph = np.array(graph)
		plt.plot(graph[:,0], graph[:,1], "r-")

		plt.show()

		graph =[]
		for i in np.arange(-10,10,0.05):
			graph.append([i, np.sin(i)/i])


	errors.append(temperrors)
	errors.append(temptesterrors)

errors=np.array(errors)

plt.plot(errors[0][:,0], errors[0][:,1], "-",label="$\eta = 0.01 $")
plt.plot(errors[1][:,0], errors[1][:,1], "-", label="$\eta = 0.01 (test)$")
plt.plot(errors[2][:,0], errors[2][:,1], "-",label="$\eta = 0.001 $")
plt.plot(errors[3][:,0], errors[3][:,1], "-",label="$\eta = 0.001 (test)$")
plt.plot(errors[4][:,0], errors[4][:,1], "-", label="$\eta = 0.0001 $")
plt.plot(errors[5][:,0], errors[5][:,1], "-", label="$\eta = 0.0001 (test) $")

forwardPropogation(train[0][0], neurons, weights)
plt.rc('text', usetex=True)
plt.rc('font', family='Computer Modern',size=16)
plt.xlabel(r'\textit{Iterations} ($\epsilon$)')
plt.ylabel(r'\textit{Mean-squared error')
plt.legend()
plt.yscale('log')
plt.show()

### plot range -10..10 after nn is trained


print "Done."