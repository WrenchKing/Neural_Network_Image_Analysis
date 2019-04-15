# Import modules
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
from numpy.random import multivariate_normal as mnormal
import matplotlib.lines as mlines
from sklearn.metrics import confusion_matrix


# Define some of the functions we will need in the Neural Network, like activations and Entropic Error measurement
def Activate(x, r):
    if r == "softmax":
        # Compute the softmax function in a numerically stable way: shift the inputs x to a range close to zero.
        # Take a look at https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        return exps/np.sum(exps)
        
    if r == "max":
        return np.maximum(0, x)
    
def dActivate(x, r):
    if r == "softmax":
        S = Activate(x, r)
        return S*(1-S)
    if r == "max":
        return (x>0)*1
    
def xentropy(guess, target):
    L = - np.sum(np.log(guess)*target)
    return L
class Layer:
    def __init__(self, name, nodes, weights):
        self.name = name         # Keep track of the layers
        self.nodes = nodes       # Keep track of the number of nodes in the layer
        self.weights = weights   # Associate a weight matrix
        self.input = 0.          # Input to the current layer (input to the Hidden Layer is l.c. of input layer inputs!)
        self.output = 0.         # Output of the current layer (output of Input Layer is also its input)
        self.deltas = 0.         # deltas of the current layer
        
# Define the NeuralNetwork-class: initialiaze the input, hidden and output layers as a list of layers.
# NeuralNetwork(..).layers[0] = Input Layer etcetera
# Add functions to develop functionality, like training of the network.

class NeuralNetwork:
    def __init__(self, ni, nh, no):
        # ni = number of inputs (without bias)
        # nh = number of hidden layers (each without bias), inserted as a list: the list length declares 
        #      the number of hidden layers, the list elements contain the number of nodes per hidden layer
        # no = number of outputs
        
        # Every layer except the Output Layer has weights
        inputweights = normal(scale=np.sqrt(2/ni), size=(nh[0], ni+1))
        self.layers = [Layer("Input Layer", ni + 1, inputweights)]
        
        # Construct nodelist to perform computations on the hidden layer in one loop
        nodelist = [ni] + nh + [no]
        
        for i in range(1, len(nodelist)-1):
            hiddenweights = normal(scale=np.sqrt(2/nodelist[i]), size=(nodelist[i+1], nodelist[i]+1))
            self.layers.append( Layer("Hidden Layer" + str(i-1), nodelist[i] + 1, hiddenweights))
            
        # See the string "NoWeights"? This will show an error about 'str' objects if we make a mistake
        self.layers.append(Layer("Output Layer", no, "NoWeights!"))
    
    
    def Train(self, inputdata, target, lr):
        # Train on one single item at a time! 
        # inputdata = np.array column vector of shape (ni,1)
        # target    = one hot encoded np.array column vector of shape (no,1) with a '1' indicating the correct class
        # All inputs, outputs and deltas will be taken as np.array column vectors.
        
        ## Perform feedforward: take inputdata, add bias, reshape, and feed into the Input Layer
        self.layers[0].input = inputdata
        self.layers[0].output = np.insert(inputdata,0,1).reshape((ni+1,1))    # Pro forma
        
        for i in range(1, len(nh)+1):
            self.layers[i].input = self.layers[i-1].weights@self.layers[i-1].output
            self.layers[i].output = Activate(self.layers[i].input, "max")
            self.layers[i].output = np.insert(self.layers[i].output,0,1).reshape((self.layers[i].nodes,1))
        
        self.layers[-1].input = self.layers[-2].weights@self.layers[-2].output
        self.layers[-1].output = Activate(self.layers[-1].input, "softmax")

        ## Perform backpropagation:
        self.layers[-1].deltas = self.layers[-1].output - target
        for i in range(len(nh), -1, -1):
            activ = dActivate(self.layers[i].input, "max")
            tmp = self.layers[i].weights.T@self.layers[i+1].deltas
            self.layers[i].deltas = np.diagflat(activ)@tmp[1:,:]
            self.layers[i].weights -= lr*(self.layers[i+1].deltas@self.layers[i].output.T)

        # Return the output of Output Layer    
        return self.layers[-1].output  
    
    
    def Test(self, inputdata): 
        # Predict given testdata after training
        inputdata = inputdata.reshape((ni,1))

        ## Perform feedforward: take inputdata, add bias, reshape, and feed into the Input Layer
        self.layers[0].input = inputdata
        self.layers[0].output = np.insert(inputdata,0,1).reshape((ni+1,1))    # Pro forma
        for i in range(1, len(nh)+1):
            self.layers[i].input = self.layers[i-1].weights@self.layers[i-1].output
            self.layers[i].output = Activate(self.layers[i].input, "max")
            self.layers[i].output = np.insert(self.layers[i].output,0,1).reshape((self.layers[i].nodes,1))
        
        self.layers[-1].input = self.layers[-2].weights@self.layers[-2].output
        self.layers[-1].output = Activate(self.layers[-1].input, "softmax")
        return self.layers[-1].output  