from __future__ import division
import numpy as np
import pylab as plt

def loadFiles():
	train = np.loadtxt("sincTrain25.dt")
	test = np.loadtxt("sincValidate10.dt")
	return train, test

def initNeuronsAndWeights(inpt, hidden, output):
	neurons = np.array([np.zeros((inpt,3)), np.zeros((hidden, 3)), np.zeros((output,3))])
	neurons[0][:,0] = np.arange(inpt,dtype='float64') #add bias neuron
	neurons[1][:,0] = np.arange(hidden,dtype='float64')+inpt
	neurons[2][:,0] = np.arange(output,dtype='float64')+inpt+hidden

	noOfWeigths = (inpt*hidden)+(hidden*output)
	weights = np.zeros((noOfWeigths, 4),dtype='float64')

	np.random.seed(1337)
	counter = 0
	for i in range(inpt):
		for j in range(hidden):
			weights[counter+j][0] = i #from neuron i
			weights[counter+j][1] = inpt+j #to neuron j
			weights[counter+j][2] = np.random.uniform(0,0.2) #random weight
			weights[counter+j][3] = np.random.uniform(0,0.2) #random bias
		counter += hidden

	for i in range(hidden):
		for j in range(output):
			weights[counter+j][0] = i+inpt
			weights[counter+j][1] = inpt+hidden+j
			weights[counter+j][2] = np.random.uniform(0,0.2)
			weights[counter+j][3] = np.random.uniform(0,0.2)
		counter += output

	#weights[0][2] += 0.000001

	return neurons, weights

"""
Activation function
"""
def activationFnc(a):
	return a / (1 + np.abs(a))

"""
Derivative of the Activation function
"""
def derivativeActivationFunction(a):
	return 1 / (1 + np.abs(a))**2

def forwardPropogation(x, neurons, weights):
	inpt = neurons[0]
	hidden = neurons[1]
	output = neurons[2]

	newWeights = np.copy(weights)

	inpt[0][1] = x
	inpt[0][2] = x #both inpt and ouput are the same

	inpt[1][1] = 1
	inpt[1][2] = 1 #bias

	for h in hidden:
		a = 0
		#for each inpt neuron, get its weight
		count = 0
		for i in inpt:
			weight = newWeights[newWeights[:,1] == h[0]] #get links going into hidden layer
			a += (weight[count][2] *  i[2])
			count+=1
		h[1] = a

		#assign new neuron values
		h[2] = activationFnc(a)

	#get value for each output neuron using summation of hidden values * their weight
	for o in output:
		ow = newWeights[newWeights[:,1] == o[0]] #get links going into the output layer
		summation = 0
		for w in ow:
			summation += w[2] * hidden[hidden[:,0]==w[0]][0][2]
		o[1] = summation
		o[2] = summation

def backwardsPropogation(t, neurons, weights, deltaWeights):
	output = neurons[2]
	hidden = neurons[1]
	inpt = neurons[0]

	y = output[0][2]
	error = t - y
	newWeights = np.copy(weights)

	# modify each weight going into the output layer
	ow = weights[weights[:,1] == output[0][0]]
	pos = 0
	for w in newWeights:
		if len(ow[ow[:,0] == w[0]]) > 0 and len(ow[ow[:,1] == w[1]]):
			deltaWeights[pos] += (error * hidden[hidden[:,0]==w[0]][0][2]) #add weight to the derivation
		pos += 1

	#get weights going into the hidden layer
	hw = []
	for h in hidden[:,0]:
		hw.append(weights[weights[:,1] == h][0]) 
		hw.append(weights[weights[:,1] == h][1])
	hw = np.array(hw)

	#modify weights going into hidden neurons
	pos = 0
	for w in newWeights:
		if len(hw[hw[:,0] == w[0]]) > 0 and len(hw[hw[:,1] == w[1]]):
			#get output weight of hidden neuron
			v = newWeights[newWeights[:,0] == w[1]]
			if len(v) > 0:
				prevLayerWeights = np.sum(v[0][2])
				# get value of hidden neuron
				prevLayerValue = hidden[hidden[:,0]==w[1]][0][1]

				if pos < len(inpt):
					deltaWeights[pos] += (error * prevLayerWeights * derivativeActivationFunction(prevLayerValue) * inpt[0][1])  #add weight to the derivation
				else:
					deltaWeights[pos] += (error * prevLayerWeights * derivativeActivationFunction(prevLayerValue) * inpt[1][1])
		pos += 1
	return deltaWeights, error

def updateWeights(weights, deltaWeights, learningRate=0.0015):
	for i in range(len(weights)):
		weights[i][2] += learningRate * deltaWeights[i]

	return weights