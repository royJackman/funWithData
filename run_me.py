import os
import sys
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

print("Please choose a classifier: ")
print("1: K Nearest Neighbors   4: + verbose")	
print("2: Neural Network        5: + verbose")
print("3: Random Forest         6: + verbose")

# Collect input
c = int(input())

# Start timing
start_time = time.time()

# Load data
data = np.loadtxt(open("creditcard.csv", "rb"), delimiter=",", skiprows=1, converters={30: lambda s: float(s.replace('"',''))})

# Massage the data into shape
train = data[:-90000, :]
test = data[-90000:, :]
train_x = train[:, 0:train.shape[1]-1]
train_y = train[:, -1]
test_x = test[:, 0:test.shape[1]-1]
test_y = test[:, -1]

# For each option...
if c == 1:
	# Define the classifier
	clf = KNeighborsClassifier(n_jobs=-1)

	# Fit the data, predict new values on test set
	clf.fit(train_x, train_y)
	prediction = clf.predict(test_x)

	# Print the accuracy and the time taken
	print("Nearest Neighbors Accuracy: " + str(100.0 * accuracy_score(test_y, prediction)) + "%")
	print("Time taken: " + str(time.time() - start_time))

elif c == 2:
	clf = MLPClassifier(activation='tanh', learning_rate='adaptive')

	clf.fit(train_x, train_y)
	prediction = clf.predict(test_x)

	print("Neural Net Accuracy: " + str(100.0 * accuracy_score(test_y, prediction)) + "%")
	print("Time taken: " + str(time.time() - start_time))

elif c == 3:
	clf = RandomForestClassifier()

	clf.fit(train_x, train_y)
	prediction = clf.predict(test_x)

	print("Random Forest Accuracy: " + str(100.0 * accuracy_score(test_y, prediction)) + "%")
	print("Time taken: " + str(time.time() - start_time))

elif c == 4:
	print("There is no verbose for KNN...sorry")
	clf = KNeighborsClassifier(n_jobs=-1)

	clf.fit(train_x, train_y)
	prediction = clf.predict(test_x)

	print("Nearest Neighbors Accuracy: " + str(100.0 * accuracy_score(test_y, prediction)) + "%")
	print("Time taken: " + str(time.time() - start_time))

elif c == 5:
	clf = MLPClassifier(activation='tanh', verbose=True, learning_rate='adaptive')

	clf.fit(train_x, train_y)
	prediction = clf.predict(test_x)

	print("Neural Net Accuracy: " + str(100.0 * accuracy_score(test_y, prediction)) + "%")
	print("Time taken: " + str(time.time() - start_time))

elif c == 6:
	clf = RandomForestClassifier(verbose=1)

	clf.fit(train_x, train_y)
	prediction = clf.predict(test_x)

	print("Random Forest Accuracy: " + str(100.0 * accuracy_score(test_y, prediction)) + "%")
	print("Time taken: " + str(time.time() - start_time))
elif c == -1:
	print("You fell for that?")
else:
	print("Nice try, but there totally isn't a hidden classifier")
	print("It's totally not at the location e^(pi*i)")
	print("Now try one of the numbers I showed you earlier")