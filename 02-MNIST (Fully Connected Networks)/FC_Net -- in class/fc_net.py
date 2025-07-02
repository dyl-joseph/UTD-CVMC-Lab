import argparse, struct, os
from array import array
import numpy as np
from model import FCNet

def parser():

    args_reader = argparse.ArgumentParser(description="Script to train a linear regression model with two parameteres on house rent datasest.")

    args_reader.add_argument('--data_path', type=str, default='/Users/dylanjoseph/Downloads/archive', help = 'Path to the  file containing MNIST data')
    
    args = args_reader.parse_args()

    return args

def read_images_labels(images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        images = [np.array(elem).astype(np.float32)/255 for elem in images]
        return images, labels


if __name__ == "__main__":

    args = parser()

    train_images_path = os.path.join(args.data_path, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(args.data_path, 'train-labels.idx1-ubyte')
    train_images, train_labels = read_images_labels(train_images_path, train_labels_path)

    test_images_path = os.path.join(args.data_path, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(args.data_path, 't10k-labels.idx1-ubyte')
    test_images, test_labels = read_images_labels(test_images_path, test_labels_path)

    print('train_images: ', train_images[0])
    print('train_labels: ', train_labels[0])

    fc_net_obj = FCNet()
    fc_net_obj.train(train_images, train_labels)
    fc_net_obj.evaluate(test_images, test_labels)

