from __future__ import division
from sklearn import svm
import numpy as np

trainfile = open("parkinsonsTrainStatML.dt", "r")
testfile = open("parkinsonsTrainStatML.dt", "r")

"""
This function reads in the files, strips by newline and splits by space char. 
It returns the dataset as numpy arrays.
"""
def read_data(filenamezaqaa):
	features = ([])
	labels = []
	for l in filename.readlines():
		l = np.array(l.rstrip('\n').split(),dtype='float')
		features.append(l[:-1])
		labels.append(l[-1])
	return labels, features



##############################################################################
#
#                             Normalizing 
#
##############################################################################
"""
This function takes a dataset with class labels and computes the mean and the variance of each input feature (leaving the class column out)
It returns two lists: [mean of first feature, mean of second feature] [variance of first feature, variance of second feature]
"""
def mean_variance(data):
	Mean = []
	Variance = []
	for i in xrange(number_of_features): 
		s = 0
		su = 0

		#mean
		for elem in data:
			s += elem[i]
		mean = s / len(data)
		Mean.append(mean)

		#variance:
		for elem in data:
			su += (elem[i] - Mean[i])**2
			variance = su/len(data)	
		Variance.append(np.sqrt(variance))
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


Y_train, X_train = read_data(trainfile)
Y_test, X_test = read_data(testfile)

number_of_features = len(X_train[0])
X_train_norm = meanfree(X_train)
X_test_trans = transformtest(X_train, X_test) 
transformtest_mean, transformtest_var = mean_variance(X_test_trans)
print transformtest_mean, transformtest_var

