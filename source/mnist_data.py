import os
import io
import struct
import numpy as np

def load():

    # loading into signed floats even though data in file is unisgned byte 
    # need negative values and floats for mean normalization
    dtype = np.float32
    
    # constants, defined as per the idx file format in use
    # labels file - 4 bytes magic no, 4 bytes number of rows, rest data
    # image file - 4 bytes magic no, 4 bytes number of items, 
    #              4 bytes number of rows, 4 bytes number of columns. Rest data
    header_bytes = 4
    label_bytes = 1
    picture_bytes = 1
    
    # data dir. Project directory must be arranged as <project dir>/source/<py file>, and <project dir/data/two files below
    data_dir = "/data/"
    label_file = "train-labels-idx1-ubyte"
    image_file = "train-images-idx3-ubyte"
    
    dir_path = os.path.abspath('..')+data_dir
    label_filename = os.path.join(dir_path, label_file)
    image_filename = os.path.join(dir_path, image_file)
    
    print("Reading MNIST label data")
    
    with open(label_filename, "rb") as label_mnist_file:
    #big endian file format
        label_magic_num, label_num_items = struct.unpack (
                ">II", label_mnist_file.read(8))
    
        y = np.fromfile( label_mnist_file, dtype, -1, "") 
    
    #sanity check on collected data
    print("Finished reading MNIST labels, label array size %s" 
               %str((y.shape)))
    
    print("Reading MNIST image data")
    
    with open(image_filename, "rb") as image_mnist_file:
        #big endian file format
        image_magic_num, image_num_items, \
        image_num_rows, image_num_columns = struct.unpack(
                ">IIII", image_mnist_file.read(16))
        X = np.fromfile ( image_mnist_file, dtype, -1, "")
    
    #reformat single vector into a matrix, 1 image per column, 60000 columns
    X = X.reshape(image_num_items, image_num_rows, image_num_columns)
    
    #sanity check
    print("Finished reading MNIST images, image array size %s" %(str(X.shape)))

    return X, y
    
    
    
    
