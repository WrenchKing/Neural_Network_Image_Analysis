# Import modules
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from numpy.random import normal
from numpy.random import multivariate_normal as mnormal
import matplotlib.lines as mlines
from sklearn.metrics import confusion_matrix
from simple_network_functions import Activate, dActivate, xentropy, Layer, NeuralNetwork

# Test the classes: for our simple Neural Network, we had 2 input nodes, 4 hidden nodes and 2 output nodes.  
ni = 2
nh = [4]
no = 2
nn = NeuralNetwork(ni, nh, no)
inputdata = np.array([5,6])
guess = nn.Train(inputdata, np.array([[1],[0]]))

# Look at the input layer: 
print("The", nn.layers[0].name, "  has",  nn.layers[0].nodes, "nodes and weights W_i with shape", nn.layers[0].weights.shape)

# Look at the hidden layer: 
print("The", nn.layers[1].name, "has",  nn.layers[1].nodes, "nodes and weights W_o with shape", nn.layers[1].weights.shape)

# Look at the hidden layer: 
print("The", nn.layers[2].name, " has",  nn.layers[2].nodes, "nodes and weights W_h with shape", nn.layers[2].weights)

# This corresponds to the architecture in the course note

# Make some training data with labels 0 and 1:
classSamples = 50
sigma = np.eye(2)  

c1 = np.hstack((mnormal([1,1], sigma, classSamples), np.zeros((classSamples,1), dtype=np.int))) # label 0
c2 = np.hstack((mnormal([5,1], sigma, classSamples), np.ones((classSamples,1), dtype=np.int))) # label 1
c3 = np.hstack((mnormal([-4,2], sigma, classSamples), np.ones((classSamples,1), dtype=np.int))) # label 1 - extra
traindata = np.random.permutation(np.vstack((c1, c2, c3)))
trainSamples = traindata.shape[0]

# Train the network:
ni = 2
nh = [10]
no = 2
nn = NeuralNetwork(ni, nh, no)

epochs = 300
guesses = np.zeros((trainSamples,1))
xerror = np.zeros((epochs,1))

for i in range(epochs):
    for j in range(trainSamples):
        inputdata = traindata[j,0:-1].reshape((ni,1))
        # Make one hot encoded target vector:
        target = np.zeros((no,1))
        target[int(traindata[j, -1])] = 1
        
        guess = nn.Train(inputdata, target, lr = 0.001)
        if (j == 10):
            xerror[i] = xentropy(guess, target)
        if (i == epochs-1): 
            # Save the predictions of the last epoch
            guesses[j] = np.argmax(guess)
    
# Plot results:
tr = traindata[:,2].astype(np.int)
te = guesses.astype(np.int).reshape(trainSamples)-tr
classcolor = np.array(["black", "white"])
corrcolor = np.array(["green", "red"])

f,ax = plt.subplots(figsize=(15,5), ncols = 2)
ax[0].set_title('Test data')

ax[0].scatter(traindata[:,0], traindata[:,1], facecolors = classcolor[tr], edgecolors='k', s = 100);
ax[0].scatter(traindata[:,0], traindata[:,1], facecolors = corrcolor[te], edgecolors ='none', s = 15);

ax[1].plot(xerror)
ax[1].set_title('Cross-entropic error');

# Make a nice example as in an image of 300x300 pixels:
w = 300
h = 300
classSamples = 100
sigma = 350*np.eye(2)


# label 0:
c01 = np.hstack((mnormal([w/4,h/4], sigma, classSamples), np.zeros((classSamples,1), dtype=np.int))) 
c02 = np.hstack((mnormal([3*w/4,3*h/4], sigma, classSamples), np.zeros((classSamples,1), dtype=np.int))) 
# label 1:
c11 = np.hstack((mnormal([w/4,3*h/4], sigma, classSamples), np.ones((classSamples,1), dtype=np.int)))
c12 = np.hstack((mnormal([3*w/4,h/4], sigma, classSamples), np.ones((classSamples,1), dtype=np.int))) 

# All training data:
traindata = np.random.permutation(np.vstack((c01, c02, c11, c12)))
trainSamples = traindata.shape[0]


# Train the network:
ni = 2
nh = [6,6,6]
no = 2
nn = NeuralNetwork(ni, nh, no)

epochs = 500
guessTrain = np.zeros((trainSamples,1))
xerror = np.zeros((epochs,1))

for i in range(epochs):
    for j in range(trainSamples):
        
        inputdata = traindata[j,0:-1].reshape((ni,1))
        target = np.zeros((no,1))
        target[int(traindata[j, -1])] = 1
        
        guess = nn.Train(inputdata, target, lr = 0.0007)
        if (j == 10):
            xerror[i] = xentropy(guess, target)
        if (i == epochs-1):
            guessTrain[j] = np.argmax(guess)
    
# Plot results: 
# tr returns black for label 0, white for label 1
# te returns green for a correct prediction (difference between label and prediction = 0) and red for incorrect prediction
tr = traindata[:,2].astype(np.int)
te = guessTrain.astype(np.int).reshape(trainSamples)-tr
classcolor = np.array(["black", "white"])
corrcolor = np.array(["green", "red"])

f,ax = plt.subplots(figsize=(15,5), ncols = 2)
ax[0].set_title('Test data')

ax[0].scatter(traindata[:,0], traindata[:,1], facecolors = classcolor[tr], edgecolors='k', s = 100);
ax[0].scatter(traindata[:,0], traindata[:,1], facecolors = corrcolor[te], edgecolors ='none', s = 15);

ax[1].plot(xerror)
ax[1].set_title('Cross-entropic error');

# Test the network: feed the 'background' of (w,h) through the network
testdata = []
guessTest = []
interval = 2
for i in range(0,w,interval):
    for j in range(0,h,interval):
        inputdata = np.array([i,j])
        testdata.append(inputdata)
        guessTest.append(np.argmax(nn.Test(inputdata)))
        
testdata = np.array(testdata)
testSamples = testdata.shape[0]
guessTest = np.array(guessTest)


# Plot everything together:
f,ax = plt.subplots(figsize=(7,7))
ax.set_title('Training and testing of a neural network')

#Plot the testing data (the background):
te = guessTest.astype(np.int).reshape(testSamples)
classcolor = np.array(["white", "black"])
ax.scatter(testdata[:,0], testdata[:,1], facecolors = classcolor[te], edgecolors = 'none', s = 1);

# Plot the training data
tr = traindata[:,2].astype(np.int)
te = guessTrain.astype(np.int).reshape(trainSamples)-tr
classcolor = np.array(["black", "white"])
corrcolor = np.array(["green", "red"])

ax.scatter(traindata[:,0], traindata[:,1], facecolors = classcolor[tr], edgecolors='k', s = 100);
ax.scatter(traindata[:,0], traindata[:,1], facecolors = corrcolor[te], edgecolors ='none', s = 15);



