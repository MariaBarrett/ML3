from __future__ import division
from sklearn.svm import SVC, libsvm
import numpy as np
import random
from operator import itemgetter
from collections import Counter


trainfile = open("parkinsonsTrainStatML.dt", "r")
testfile = open("parkinsonsTrainStatML.dt", "r")

"""
This function reads in the files, strips by newline and splits by space char. 
It returns the labels as a 1D list and the features as one numpy array per row.
"""
def read_data(filename):
	features = []
	labels = []
	for l in filename.readlines():
		l = np.array(l.rstrip('\n').split(),dtype='float')
		features.append(l[:-1])
		labels.append(l[-1])
	feat = np.array(features)
	lab = np.array(labels)
	return lab, feat



##############################################################################
#
#                      Normalizing and transforming
#
##############################################################################
"""
This function takes a dataset with class labels and computes the mean and the variance of each input feature (leaving the class column out)
It returns two lists: [mean of first feature, mean of second feature] [variance of first feature, variance of second feature]
"""
def mean_variance(data):
	Variance = []

	#mean
	Mean = sum(data) / len(data)

	#variance:
	for i in xrange(len(data[0])): #for every feature
		su = 0
		for elem in data: 
			su += (elem[i] - Mean[i])**2
		variance = su / len(data)
		Variance.append(variance)
	return Mean, Variance
	

"""
This function expects a dataset without class labels.
It calls mean_variance to get the mean and the variance for each feature
Then these values are used to normalize every datapoint to zero mean and unit variance.
A copy of the data is created. 
The normalized values are inserted at the old index in the copy thus preserving class label 
The new, standardized data set with untouched class labels is returned
"""
def meanfree(data):
	mean, variance = mean_variance(data)

	new = np.copy(data)
	for i in xrange(len(new)):
		for num in xrange(number_of_features):
			#replacing at correct index in the copy
			new[i][num] = (new[i][num] - mean[num]) / np.sqrt(variance[num])
	return new 

"""
This function transforms the test set using the mean and variance from the train set.
It expects the train and test set without class labels. 
It makes a copy of the test sets and inserts the transformed feature value at the correct index.
It returns the transformed test set woth untouched labels. 

"""
def transformtest(trainset, testset):
	#getting the mean and variance from train:
	meantrain, variancetrain = mean_variance(trainset)
	newtest = np.copy(testset)

	for num in xrange(number_of_features):
		for i in xrange(len(testset)):
			#replacing at correct index in the copy
			newtest[i][num] = (testset[i][num] - meantrain[num]) / np.sqrt(variancetrain[num])
	return newtest

"""
This function splits the shuffled train set in s equal sized splits. The lambda constant makes sure that it's always shuffled the same way 
It returns a list of s slices containg lists of datapoints.
"""
def sfold(labels, features,s):
	random.shuffle(features, lambda: 0.5) 
	random.shuffle(labels, lambda: 0.5) #using the same shuffle 
	feature_slices = [features[i::s] for i in xrange(s)]
	label_slices = [labels[i::s] for i in xrange(s)]
	return label_slices, feature_slices

##############################################################################
#
#                      Cross validation
#
##############################################################################

"""
After having decorated, this function gets a slice for testing and uses the rest for training.
First we choose test-set - that's easy.
Then for every test-set for as many folds as there are: use the remaining as train sets exept if it's the test set. 
Then we sum up the result for every run and average over them and print the result.  
"""
def crossval(X_train, y_train, folds):
	# Set the parameters by cross-validation
	tuned_parameters = [{'gamma': [0.00001,0.0001,0.001,0.01,0.1,1],
                     'C': [0.001,0.01,0.1,1,10,100]}]

	print ('-'*45)
	print ('%d-fold cross validation' %folds)
	print ('-'*45)
	
	labels_slices, features_slices = sfold(y_train, X_train,folds)
	Accuracy = []

	for f in xrange(folds):
		crossvaltrain = []
		crossvaltrain_labels = []
		

		#define test-set for this run
		crossvaltest = features_slices[f]
		crossvaltest =  np.array(crossvaltest)
		crossvaltest_labels = labels_slices[f]
		crossvaltest_labels = np.array(crossvaltest_labels)

		for i in xrange(folds): #putting content of remaining slices in the train set 
			if i != f: 
				for elem in features_slices[i]:
					crossvaltrain.append(elem) #making a list of trainset for this run
					
				for lab in labels_slices[i]:
					crossvaltrain_labels.append(lab) #with adjacent labels
		crossvaltrain = np.array(crossvaltrain)
		crossvaltrain_labels = np.array(crossvaltrain_labels)

		#gridsearch
		for g in tuned_parameters[0]['gamma']:
			accuracy = []
			for c in tuned_parameters[0]['C']:
				clf = SVC(C=c, gamma=g)
				clf.fit(crossvaltrain, crossvaltrain_labels)
				y_pred = clf.predict(crossvaltest)
				counter = 0
				temp = []
				for y in xrange(len(y_pred)):
					if y_pred[y] != crossvaltest_labels[y]:
						counter +=1
				temp.append(g)
				temp.append(c)
				temp.append(counter / len(crossvaltrain))
				accuracy.append(temp)
			
		#For every fold get the best hyperparam:
		accuracy.sort(key=itemgetter(2))
		best_of_the_fold = tuple(accuracy[0][:2]) #not taking the error value - only the hyperparams
		print "Best hyperparameters of fold %d (gamma, C) %s" %(f+1, best_of_the_fold)
		Accuracy.append(best_of_the_fold) 

	# finding the most frequent pair
	counted = Counter(Accuracy).most_common()
	most_frequent_pair = counted[0][0]
	print "Best hyperparameter", most_frequent_pair
	return most_frequent_pair

def error_svc(X_train, y_train, X_test, y_test):
	clf = SVC(C=1, gamma=0.001)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	counter = 0
	for y in xrange(len(y_pred)):
		if y_pred[y] != y_test[y]:
			counter +=1
	error = counter / len(X_test)
	return error

def freeandbounded_libsvm(X_train, y_train, X_test, y_test, c):
	out = libsvm.fit(X_train, y_train, C=c, gamma=0.0000000001)
	return len(out[1])

def differentC(X_train, y_train, X_test, y_test):
	C = [1,10000,100000,1000000,10000000, 100000000, 1000000000, 10000000000 ]
	for c in C:
		sup_vectors = freeandbounded_libsvm(X_train, y_train, X_test, y_test, c)
		print "Number of free and bounded support vectors in trainset with C = %d: %d" %(c,sup_vectors)
	


 ##############################################################################
#
#                   	  Calling
#
##############################################################################

y_train, X_train = read_data(trainfile)
y_test, X_test = read_data(testfile)

number_of_features = len(X_train[0])
train_mean, train_variance = mean_variance(X_train)

print "Mean of train set before normalization: \n", train_mean
print "Variance of train set before normalization: \n", train_variance

X_train_norm = meanfree(X_train)
X_test_trans = transformtest(X_train, X_test) 
transformtest_mean, transformtest_var = mean_variance(X_test_trans)
print "Mean of transformed test set: \n", transformtest_mean
print "Variance of transformed test set: \n", transformtest_var

print '*'*45
print "Not normalized"
print '*'*45
best_hyperparam_pair = crossval(X_train, y_train, 5)

print '*'*45
print "Normalized"
print '*'*45
best_hyperparam_pair_norm = crossval(X_train_norm, y_train, 5)

print '*'*45
print "Error when trained on train set tested on test set with C = 1, gamma = 0.001"
print '*'*45
err = error_svc(X_train, y_train, X_test, y_test)
print "Not normalized: ", err
err_norm = error_svc(X_train_norm, y_train, X_test_trans, y_test)
print "Normalized: ", err_norm
		
print '*'*45
print "Number of free and bounded support vectors with different C, gamma = 0.0000000001"
print '*'*45		
differentC(X_train, y_train, X_test, y_test)

#when doing crossval on the normalized set should we normalize the trainset and transform the test set?
