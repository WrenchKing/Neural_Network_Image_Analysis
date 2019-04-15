# Import modules
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from numpy.random import normal
from numpy.random import multivariate_normal as mnormal
import matplotlib.lines as mlines
from sklearn.metrics import confusion_matrix
from simple_network_functions import Activate, dActivate, xentropy, Layer, NeuralNetwork
#Load the image data (download .mat file from inside):
import scipy.io
mat_contents = scipy.io.loadmat('MNIST_train.mat')
mnist = mat_contents['train_data']
labels = mat_contents['train_label']

mnist.astype(np.float32)
def Im(l):
    return np.reshape(l,(28,28))

index = 0
testIm = Im(mnist[index])
plt.imshow(testIm, cmap = 'gray');
print(labels[index])

trainlabels = np.array([np.where(r == 1)[0][0] for r in labels])
zerosIndex = np.array(np.where(trainlabels == 0)).reshape((5923))
onesIndex = np.array(np.where(trainlabels == 1)).reshape((6742))

print(labels[zerosIndex][0][0:2])
zeros = np.hstack((mnist[zerosIndex,:], np.zeros((5923,1)) ))
ones = np.hstack((mnist[onesIndex,:], np.ones((6742,1)) ))
traindata = np.random.permutation(np.vstack((zeros,ones)))
trainSamples = traindata.shape[0]
print(type(traindata[0][1]))

ni = mnist.shape[1]
nh = [32]
no = 2
nn = NeuralNetwork(ni, nh, no)

epochs = 20
guesses = np.zeros((trainSamples,1))

for i in range(epochs):
    for j in range(trainSamples):
        inputdata = traindata[j,0:-1].reshape((ni,1))
        target = np.zeros((no,1))
        target[int(traindata[j, -1])] = 1
        guess = nn.Train(inputdata, target, lr = 0.0001)
        if (i == epochs-1):
            guesses[j] = np.argmax(guess)

confusion_matrix(traindata[:,-1].astype(np.int), guesses.astype(np.int), np.linspace(0,1,2).astype(np.int))



# WATCH OUT: CURRENT NETWORK TAKES 30 MINUTES TO RUN!!
ni = mnist.shape[1]
nh = [64]
no = 10
nn = NeuralNetwork(ni, nh, no)

traindata = mnist
trainlabelsMatrix = labels
trainSamples = traindata.shape[0]

# Construct labeled data:
#trainlabels = np.array([np.where(r == 1)[0][0] for r in trainlabelsMatrix]).reshape((trainSamples,1))
#traindata = np.hstack((traindata, trainlabels))

print(traindata.shape, trainlabelsMatrix.shape, trainSamples)


# Does the network train with a single sample?
guess = nn.Train(traindata[10], trainlabelsMatrix[10].T.reshape((10,1)), lr = 0.001)
print(np.argmax(guess))


epochs = 10
guesses = np.zeros((trainSamples,1))

for i in range(epochs):
    for j in range(trainSamples):
        inputdata = traindata[j,:].reshape((784,1))
        target = trainlabelsMatrix[j].reshape((10,1))
        guess = nn.Train(inputdata, target, lr = 0.0001)
        if (i == epochs-1):
            guesses[j] = np.argmax(guess)

# Make sound when ready
import os
os.system("printf '\7'") # or '\7'


confusion_matrix(trainlabels, guesses.astype(np.int), np.linspace(0,9,10).astype(np.int))