import numpy as np
import mnist_data as mnist

#define sigmoid function, for logistic regression
def sigmoid ( Z ):
    return ( 1 / ( 1 + numpy.exp(-Z) ) )

#get mnist data
X, y, num_classes = mnist.load() 
# initialize random weights
W = np.random.rand( num_classes, X.shape[0] )

#append bias
bias = np.ones ((1, X.shape[1]))
X = np.vstack ( (X, bias) ) 

bias = np.ones ((W.shape[0],1))
W = np.hstack ( (W, bias) ) 

print(W.shape)

#forward prop
newX = np.dot(W, X)

