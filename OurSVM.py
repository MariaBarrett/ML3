from __future__ import division
from sklearn.svm import libsvm
import numpy as np
import random
from operator import itemgetter
from collections import Counter


trainfile = open("parkinsonsTrainStatML.dt", "r")
testfile = open("parkinsonsTestStatML.dt", "r")


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
This function takes a dataset with class labels and computes the mean and the variance of each input feature 
It returns two lists: [mean of first feature, mean of second feature] [variance of first feature, variance of second feature]
"""
def mean_variance(data):
	mean = sum(data) / len(data)
	variance = sum((data - mean)**2) / len(data)
	return mean, variance
	

"""
This function expects a dataset.
It calls mean_variance to get the mean and the variance for each feature.
Then each datapoint is normalized using means and variance. 
"""
def meanfree(data):
	mean, variance = mean_variance(data)
	meanfree = (data - mean) / np.sqrt(variance)
	return meanfree

"""
This function transforms the test set using the mean and variance from the train set.
It expects the train and test set and call a function to get mean and variance of the train set.
It uses these to transform all datapoints of the test set. 
It returns the transformed test set. 

"""
def transformtest(trainset, testset):
	#getting the mean and variance from train:
	meantrain, variancetrain = mean_variance(trainset)
	transformed = (testset - meantrain) / np.sqrt(variancetrain)
	return transformed

"""
This function splits the shuffled train set in s equal sized splits. 
It starts by making a copy of the labels and features and shuffles them. The lambda constant makes sure that it's always shuffled the same way 
It returns a list of s slices containg lists of datapoints belonging to s.
"""
def sfold(labels, features,s):
	featurefold = np.copy(features)
	labelfold = np.copy(labels)

	random.shuffle(featurefold, lambda: 0.5) 
	random.shuffle(labelfold, lambda: 0.5) #using the same shuffle 
	feature_slices = [featurefold[i::s] for i in xrange(s)]
	label_slices = [labelfold[i::s] for i in xrange(s)]
	return label_slices, feature_slices

##############################################################################
#
#                      Cross validation
#
##############################################################################

"""
After having decorated, the function gets all combinations of C's and gammas. For each combination it
runs 5 fold crossvalidation: 
For every test-set for as many folds as there are: use the remaining as train sets exept if it's the test set. 
Then we sum up the result for every run and average it. The average performance per combination is stored.
The best /lowest average and the combination that produced it is returned.   
"""
def crossval(X, y, folds):
	# Set the parameters by cross-validation
	tuned_parameters = [{'gamma': [0.00001,0.0001,0.001,0.01,0.1,1],
                     'C': [0.001,0.01,0.1,1,10,100]}]

	print ('-'*45)
	print ('%d-fold cross validation' %folds)
	print ('-'*45)
	
	labels_slices, features_slices = sfold(y, X,folds)
	Accuracy = []

	#gridsearch
	for g in tuned_parameters[0]['gamma']:
		accuracy = []
		for c in tuned_parameters[0]['C']:
			temp = []
			#crossvalidation
			for f in xrange(folds):
				crossvaltrain = []
				crossvaltrain_labels = []
				
				#define test-set for this run
				crossvaltest = features_slices[f]
				crossvaltest =  np.array(crossvaltest)
				crossvaltest_labels = labels_slices[f]
				crossvaltest_labels = np.array(crossvaltest_labels)

				for i in xrange(folds): #putting content of remaining slices in the train set 
					if i != f: # - if it is not the test slice: 
						for elem in features_slices[i]:
							crossvaltrain.append(elem) #making a list of trainset for this run
							
						for lab in labels_slices[i]:
							crossvaltrain_labels.append(lab) #...and a list of adjacent labels
				
				crossvaltrain = np.array(crossvaltrain)
				crossvaltrain_labels = np.array(crossvaltrain_labels)

				#Classifying using libsvm
				out = libsvm.fit(crossvaltrain, crossvaltrain_labels, svm_type=0, C=g, gamma=g)
				y_pred = libsvm.predict(crossvaltest, *out)
				#getting the error count
				counter = 0
				
				for y in xrange(len(y_pred)):
					if y_pred[y] != crossvaltest_labels[y]:
						counter +=1
				#and storing the error result. 
				temp.append(counter / len(crossvaltrain))

			#for every setting, get the average performance of the 5 runs:
			temp = np.array(temp)
			mean = np.mean(temp)
			print "Average error of %s: %.6f" %((g,c), mean)
			accuracy.append([c,g,mean])

	#After all C's and gammas have been tried: get the best performance and the hyperparam pairs for that:
	accuracy.sort(key=itemgetter(2)) #sort by error - lowest first
	bestperf = accuracy[0][-1]
	bestpair = tuple(accuracy[0][:2])
	print "\nAverage error of best hyperparameter, %s: %.4f" %(bestpair, bestperf)


def error_svc(X_train, y_train, X_test, y_test):
	out = libsvm.fit(X_train, y_train, svm_type=0, C=1, gamma=0.001)
	y_pred = libsvm.predict(X_test, *out)

	counter = 0
	for y in xrange(len(y_pred)):
		if y_pred[y] != y_test[y]:
			counter +=1
	error = counter / len(X_test)
	return error
	
"""
This function tries fitting using different C's and looks at the output. 
If the coefficient = c, then the support vector is bounded.
The rest of the support vectors are free.
It prints the number of free and bound support vectors. 
"""
def differentC(X_train, y_train, X_test, y_test):
	C = [1,10,100,1000,10000]
	
	for c in C:
		bounded = 0
		out = libsvm.fit(X_train, y_train, C=c, gamma=0.001)	

		supportvectors = len(out[0])
		coef = out[3]
		for co in coef[0]:
			if co == c:
				bounded += 1
		free = supportvectors - bounded
		print "C = %d: free: %d, bounded: %d " %(c, free, bounded)
	

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
best_hyperparam = crossval(X_train, y_train, 5)

print '*'*45
print "Normalized"
print '*'*45
best_hyperparam_norm = crossval(X_train_norm, y_train, 5)

print '*'*45
print "Error when trained on train set tested on test using best hyperparameter"
print '*'*45

err = error_svc(X_train, y_train, X_test, y_test)
print "Not normalized: ", err

err_norm = error_svc(X_train_norm, y_train, X_test_trans, y_test)
print "Normalized: ", err_norm
		
print '*'*45
print "Number of free and bounded support vectors with different C, gamma = 0.001"
print '*'*45	

differentC(X_train, y_train, X_test, y_test)
