from __future__ import division
import numpy as np

def loadFiles():
	train = np.loadtxt("sincTrain25.dt")
	test = np.loadtxt("sincValidate10.dt")
	return train, test

def initNeuronsAndWeights(input, hidden, output):
	neurons = np.array([np.zeros((input,3)), np.zeros((hidden, 3)), np.zeros((output,3))])
	neurons[0][:,0] = np.arange(input) #add bias neuron
	neurons[1][:,0] = np.arange(hidden)+input
	neurons[2][:,0] = np.arange(output)+input+hidden

	noOfWeigths = (input*hidden)+(hidden*output)
	weights = np.zeros((noOfWeigths, 4))

	np.random.seed(1337)
	counter = 0
	for i in range(input):
		for j in range(hidden):
			weights[counter+j][0] = i #from neuron i
			weights[counter+j][1] = input+j #to neuron j
			weights[counter+j][2] = np.random.uniform(0,1) #random weight
			weights[counter+j][3] = np.random.uniform(0,1) #random bias
		counter += hidden

	for i in range(hidden):
		for j in range(output):
			weights[counter+j][0] = i+input
			weights[counter+j][1] = input+hidden+j
			weights[counter+j][2] = np.random.uniform(0,1)
			weights[counter+j][3] = np.random.uniform(0,1)
		counter += output

	weights[0][2] += 0.000001

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
	input = neurons[0]
	hidden = neurons[1]
	output = neurons[2]

	input[0][1] = x
	input[0][2] = x #both input and ouput are the same

	input[1][1] = 1
	input[1][2] = 1 #bias

	for h in hidden:
		a = 0
		#for each input neuron, get its weight
		for i in input:
			weight = weights[weights[:,1] == h[0]] #get links going into hidden layer
			a += (weight[0][2] *  i[2])
			h[1] = a

		#assign new neuron values
		h[2] = activationFnc(a)

	#get value for each output neuron using summation of hidden values * their weight
	for o in output:
		ow = weights[weights[:,1] == o[0]] #get links going into the output layer
		summation = 0
		for w in ow:
			summation += w[2] * hidden[hidden[:,0]==w[0]][0][2]
		o[1] = summation
		o[2] = summation

def backwardsPropogation(t, neurons, weights):
	output = neurons[2]
	hidden = neurons[1]
	input = neurons[0]

	y = output[0][2]
	error = t - y

	newWeights = np.copy(weights)

	# modify each weight going into the output layer
	ow = weights[weights[:,1] == output[0][0]]
	for w in newWeights:
		if len(ow[ow[:,0] == w[0]]) > 0 and len(ow[ow[:,1] == w[1]]):
			w[2] += error * hidden[hidden[:,0]==w[0]][0][2] #add weight to the derivation

	#get hidden layer neurons
	hw = []
	for h in hidden[:,0]:
		hw.append(weights[weights[:,1] == h][0]) 
	hw = np.array(hw)

	#modify weights going into hidden neurons
	for w in newWeights:
		if len(hw[hw[:,0] == w[0]]) > 0 and len(hw[hw[:,1] == w[1]]):
			#get output weight of hidden neuron
			v = weights[weights[:,0] == w[1]]
			if len(v) > 0:
				prevLayerWeights = np.sum(v[0][2])
				# get value of hidden neuron
				prevLayerValue = hidden[hidden[:,0]==w[1]][0][1]

				w[2] += error * prevLayerWeights * derivativeActivationFunction(prevLayerValue) * input[0][1] #add weight to the derivation

				print error * prevLayerWeights * derivativeActivationFunction(prevLayerValue) * input[0][1] #add weight to the derivation

	return newWeights


"""
MAIN
"""
train, test = loadFiles()

input, hidden, output = 2, 5, 1
neurons, weights = initNeuronsAndWeights(input, hidden, output)

#print neurons
#print weights

for x in np.arange(1):
	forwardPropogation(train[0][0], neurons,weights)
	weights = backwardsPropogation(train[0][1], neurons, weights)

print neurons
print  weights