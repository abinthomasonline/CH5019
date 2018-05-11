import numpy as np
import pandas as pd

features = pd.read_csv('../data/q1_data_matrix.csv', header=None, names=['temp', 'press', 'ffr', 'cfr', 'irc'])
labels = pd.read_csv('../data/q1_labels.csv', header=None, names=['oper'])

features = (features-features.mean())/features.std()
features.head()
features.insert(0, 'bias', 1)

[train_X, test_X] = np.split(features, [int(0.7*features.shape[0])], axis=0)
[train_Y, test_Y] = np.split(labels, [int(0.7*labels.shape[0])], axis=0)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost(probability, label):
    return (-label.iloc[:,0]*np.log(probability)-(1-label.iloc[:,0])*np.log(1-probability)).mean()

def gradient(features, parameters, labels):
    h = sigmoid(np.dot(features, parameters))
    return np.dot(features.transpose(), h-labels.iloc[:,0])/labels.shape[0]

def predict(features, parameters, threshold):
    return sigmoid(np.dot(features, parameters)) >= threshold

def missclassification_error(features, labels, parameters, threshold):
	return (predict(features, parameters, threshold)!=labels.iloc[:,0]).mean()

for k in range(20):
	alpha = 0.01
	theta = np.random.normal(0, 1, 6)
	i=1
	test_cost=1000
	while 1:
		theta -= alpha*gradient(train_X, theta, train_Y)
		train_cost = cost(sigmoid(np.dot(train_X, theta)), train_Y)
		temp = cost(sigmoid(np.dot(test_X, theta)), test_Y)
		if temp>test_cost:
			break;
		test_cost = temp
		print(i, train_cost, test_cost)
		i+=1

	print(missclassification_error(train_X, train_Y, theta, 0.5), missclassification_error(test_X, test_Y, theta, 0.5))

