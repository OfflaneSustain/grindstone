import numpy as np
import mnist_data as mnist

def sigmoid ( Z ):
    return ( 1 / ( 1 + numpy.exp(-Z) ) )

X, y = mnist.load()
W = np.random.rand ( y.shape[0], X.shape[1] )

print (W.shape)

