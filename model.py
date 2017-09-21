# coding: utf-8

"""

Credit card fraud detection using tensor flow. 

The sample data is taken from kaggle - https://www.kaggle.com/dalpozz/creditcardfraud
"""

from sklearn.utils import shuffle
from model_config import *
import tensorflow as tf 
import pandas as pd 
import numpy as np 

# utility function to slice the input data frame and generate train and test examples
def get_test_train(inputDF):
	# create extra class - non fraud
	inputDF.loc[inputDF['Class'] == 0, 'NotFraud'] = 1
	inputDF.loc[inputDF['Class'] == 1, 'NotFraud'] = 0
	inputDF = inputDF.rename(columns={'Class': 'Fraud'})

	# separate fraud and non fraud rows
	FraudDF = inputDF[inputDF['Fraud'] == 1]
	notFraudDF = inputDF[inputDF['NotFraud'] == 1]

	# out of the complete data - generate training examples (70 percent of the entire data frame)
	train_1 = FraudDF.sample(frac = 0.7)
	train_2 = notFraudDF.sample(frac = 0.7)

	trainX = pd.concat([train_1, train_2], axis = 0)    

	# rest of the rows - test data
	testX = inputDF.loc[~inputDF.index.isin(trainX.index)]

	trainX = shuffle(trainX)
	testX = shuffle(testX)

	# slice the target variable
	trainY_F = trainX.Fraud
	trainY_NF = trainX.NotFraud
	trainY = pd.concat([trainY_F, trainY_NF], axis = 1)

	testY_F = testX.Fraud
	testY_NF = testX.NotFraud
	testY = pd.concat([testY_F, testY_NF], axis = 1)

	trainX = trainX.drop(['Fraud','NotFraud'], axis = 1)
	testX = testX.drop(['Fraud','NotFraud'], axis = 1)

	return trainX, trainY, testX, testY

# Get train and test data
file_path = 'data/creditcard.csv'
inputDF = pd.read_csv(file_path)
trainX, trainY, testX, testY = get_test_train(inputDF)

# Feature Normalization
feature_list = trainX.columns.values
for feature in feature_list:
	mean = inputDF[feature].mean()
	stdv = inputDF[feature].std()
	
	trainX.loc[:, feature] = (trainX[feature] - mean) / stdv
	testX.loc[:, feature] = (testX[feature] - mean) / stdv

# Create TensorFlow Neural Net Architecture
# three hidden layers with 16,8,4 nodes
# one input layer : 30 nodes
# one output layer : 2 nodes

weights = {
'h1' : tf.Variable(tf.random_normal([input_nodes, hidden_nodes1])),
'h2' : tf.Variable(tf.random_normal([hidden_nodes1, hidden_nodes2])),
'h3' : tf.Variable(tf.random_normal([hidden_nodes2, hidden_nodes3])),
'op' : tf.Variable(tf.random_normal([hidden_nodes3, output_nodes])),
}

biases = {
	'h1': tf.Variable(tf.random_normal([hidden_nodes1])),
	'h2': tf.Variable(tf.random_normal([hidden_nodes2])),
	'h3': tf.Variable(tf.random_normal([hidden_nodes3])),
	'op': tf.Variable(tf.random_normal([output_nodes]))
}

def model(input_row):
	# currently using RelU, other choices - sigmoid, tanH
	activation1 = tf.add(tf.matmul(input_row, weights['h1']), biases['h1'])
	layer1 = tf.nn.relu(activation1)
	
	activation2 = tf.add(tf.matmul(layer1, weights['h2']), biases['h2'])
	layer2 = tf.nn.relu(activation2)
	
	activation3 = tf.add(tf.matmul(layer2, weights['h3']), biases['h3'])
	layer3 = tf.nn.relu(activation3)
	
	activation4 = tf.add(tf.matmul(layer3, weights['op']), biases['op'])
	output = tf.nn.relu(activation4)
	
	return output

# tensor flow placeholders for training and test examples
X = tf.placeholder(tf.float32, [None, input_nodes])
Y = tf.placeholder(tf.float32, [None, output_nodes])
Y_pred = model(X)

# compute the prediction, cost and optimize it
softmax = tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=Y)
cost = tf.reduce_mean(softmax)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# architecture for accuracy and validation
true_cases = tf.equal(tf.argmax(Y_pred,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(true_cases, tf.float32))

init = tf.global_variables_initializer()

# need to convert df to matrix for tensor flow feeding
inputX = trainX.as_matrix()
inputY = trainY.as_matrix()
inputX_test = testX.as_matrix()
inputY_test = testY.as_matrix()

dropout_p = tf.placeholder(tf.float32)

train_accuracies = []
train_costs = []

with tf.Session() as sess:
	sess.run(init)

	# run iteratice stochastic training
	for epoch in range(training_epochs):
		avg_cost = 0.0
		total_batch = int(trainY.shape[0]/batch_size)
 
		for i in range(total_batch):
			batch_x = inputX[i*batch_size : (1+i)*batch_size]
			batch_y = inputY[i*batch_size : (1+i)*batch_size]
			
			# compute cost for every training example batch             
			_, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})
		
		if epoch % display_step == 0:
			print epoch + 1, avg_cost
			
			train_accuracy, updated_cost = sess.run([accuracy, cost], feed_dict={X: inputX, 
																				 Y: inputY,
																				 dropout_p: dropout})

			
			train_accuracies.append(train_accuracy)
			train_costs.append(updated_cost)


# get predictions on test data
init = tf.global_variables_initializer()

predicted = tf.argmax(Y, 1)
with tf.Session() as sess:  
	sess.run(init)
   	predictions, test_accuracy = sess.run([predicted, accuracy], feed_dict={ X: inputX_test, 
															    Y:inputY_test,
															    dropout_p: dropout
															    })
   
   	print predictions
   	print test_accuracy
   