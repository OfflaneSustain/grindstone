import os
import io
import struct
import numpy as np

def load():

    #remove later
    dtype = np.int8
    
    header_bytes = 4
    label_bytes = 1
    picture_bytes = 1
    
    data_dir = "/data/"
    label_file = "train-labels-idx1-ubyte"
    image_file = "train-images-idx3-ubyte"
    
    dir_path = os.path.abspath('..')+data_dir
    label_filename = os.path.join(dir_path, label_file)
    image_filename = os.path.join(dir_path, image_file)
    
    print("Reading MNIST label data")
    
    with open(label_filename, "rb") as label_mnist_file:
    
        label_magic_num, label_num_items = struct.unpack (
                ">II", label_mnist_file.read(8))
    
        y = np.fromfile( label_mnist_file, dtype, -1, "") 
    
    print("Finished reading MNIST labels, label array size %s" 
               %str((y.shape)))
    
    print("Reading MNIST image data")
    
    with open(image_filename, "rb") as image_mnist_file:
        
        image_magic_num, image_num_items, \
        image_num_rows, image_num_columns = struct.unpack(
                ">IIII", image_mnist_file.read(16))
        X = np.fromfile ( image_mnist_file, dtype, -1, "")
    
    X = X.reshape(image_num_items, image_num_rows, image_num_columns)
    
    print("Finished reading MNIST images, image array size %s" %(str(X.shape)))

    return X, y
    
    
    
    
