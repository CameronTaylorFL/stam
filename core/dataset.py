import numpy as np
import pandas as pd
import os
import random
import pickle
import scipy.io as sio
from keras.datasets import mnist
from scipy import linalg
from sklearn.utils import shuffle
from scipy import ndimage
import cv2
import imageio
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt

##############################################
########### PREPROCESSING STEPS ##############
##############################################

# given training data, return ZCA transformation properties
# as well as transformed training data        
def ZCA(data_flat, eps=1e-5):
    
    # flatten data array and convert to zero mean
    data_var = (np.var(data_flat, axis = 1) + 10)**(1/2)
    data_flat = (data_flat - np.mean(data_flat, axis = 1)[:,None]) / data_var[:, None]
       
    # calculate covariance matrix
    mean_zca = np.mean(data_flat, axis = 0)
    cov = np.dot(data_flat.T, data_flat) / data_flat.shape[0]
    U,S,V = np.linalg.svd(cov)
    W_zca = np.dot(np.dot(U,np.diag(np.sqrt(1.0/(S + eps)))),U.T)
    data_zca = np.dot(data_flat - mean_zca, W_zca)  
      
    return W_zca, mean_zca, data_zca
    
# transform data with pre-existing zca parameters
def white_data(data_flat, mean_zca, W_zca, norm_f):

    data_var = (np.var(data_flat, axis = 1) + 10)**(1/2)
    data_flat = (data_flat - np.mean(data_flat, axis = 1)[:,None]) / data_var[:, None]
    
    if norm_f:
        data_out = np.dot(data_flat - mean_zca, W_zca)
    else:
        data_out = data_flat
   
    return data_out

def normalization(configs, args, x_, x_eval):
    # no normalization
    if args.norm_flag == 0:
        configs['im_scale'] = [0, 255]

    # per image standardization
    elif args.norm_flag == 1:  
        configs['im_scale'] = [-3, 3]
        x_ = (x_ - np.mean(x_, axis=(1,2,3))[:,None,None,None]) / np.std(x_, axis=(1,2,3))[:,None,None,None]
        x_eval = (x_eval - np.mean(x_eval, axis=(1,2,3))[:,None,None,None]) / np.std(x_eval, axis=(1,2,3))[:,None,None,None]

    # per image zca whitening
    elif args.norm_flag == 2:  
        configs['im_scale'] = [-3, 3]   
        
        # zca
        eps = 1e-5

        # train
        shape = x_.shape
        data_flat = x_.reshape((shape[0], -1))
        
        # flatten data array and convert to zero mean
        data_var = (np.var(data_flat, axis = 1) + 10)**(1/2)
        data_flat = (data_flat - np.mean(data_flat, axis = 1)[:,None]) / data_var[:, None]
            
        # calculate covariance matrix
        mean_zca = np.mean(data_flat, axis = 0)
        cov = np.dot(data_flat.T, data_flat) / data_flat.shape[0]
        U,S,V = np.linalg.svd(cov)
        W_zca = np.dot(np.dot(U,np.diag(np.sqrt(1.0/(S + eps)))),U.T)
        data_zca = np.dot(data_flat - mean_zca, W_zca)  
        x_ = data_zca.reshape(shape)
        
        # val
        shape = x_eval.shape
        data_flat = x_eval.reshape((shape[0], -1))
        
        # flatten data array and convert to zero mean
        data_var = (np.var(data_flat, axis = 1) + 10)**(1/2)
        data_flat = (data_flat - np.mean(data_flat, axis = 1)[:,None]) / data_var[:, None]
            
        # calculate covariance matrix
        mean_zca = np.mean(data_flat, axis = 0)
        cov = np.dot(data_flat.T, data_flat) / data_flat.shape[0]
        U,S,V = np.linalg.svd(cov)
        W_zca = np.dot(np.dot(U,np.diag(np.sqrt(1.0/(S + eps)))),U.T)
        data_zca = np.dot(data_flat - mean_zca, W_zca)  
        x_eval = data_zca.reshape(shape)
        
    # per image sobel filtering
    elif args.norm_flag == 3:  
        configs['im_scale'] = [0, 1] 

        # normalization
        for i in range(len(x_)):
            dx = ndimage.sobel(x_[i], 0)  # horizontal derivative
            dy = ndimage.sobel(x_[i], 1)  # vertical derivative
            mag = np.hypot(dx, dy)  # magnitude
            x_[i] = mag / np.max(mag)
        for i in range(len(x_eval)):
            dx = ndimage.sobel(x_eval[i], 0)  # horizontal derivative
            dy = ndimage.sobel(x_eval[i], 1)  # vertical derivative
            mag = np.hypot(dx, dy)  # magnitude
            x_eval[i] = mag / np.max(mag)

    # put into -1,1 range
    elif args.norm_flag == 4:  
        configs['im_scale'] = [-1, 1]
        x_ = (x_ * 2.0 / 255.0 - 1.0)
        x_eval = (x_eval * 2.0 / 255.0 - 1.0)

    # put into 0,1 range
    elif args.norm_flag == 5:  
        configs['im_scale'] = [0, 1]
        x_ = (x_ - np.mean(x_, axis=(1,2,3))[:,None,None,None]) / (np.std(x_, axis=(1,2,3)))[:,None,None,None]
        x_eval = (x_eval - np.mean(x_eval, axis=(1,2,3))[:,None,None,None]) / (np.std(x_eval, axis=(1,2,3)))[:,None,None,None]

    # put into -1,1 range
    elif args.norm_flag == 6:
        configs['im_scale'] = [-1, 1]
        x_ = (x_ - np.mean(x_, axis=(1,2,3))[:,None,None,None]) / (3*np.std(x_, axis=(1,2,3)))[:,None,None,None]
        x_eval = (x_eval - np.mean(x_eval, axis=(1,2,3))[:,None,None,None]) / (3*np.std(x_eval, axis=(1,2,3)))[:,None,None,None]

    return configs, x_, x_eval

# function to transform RBG to grayscale
def rgb2grey(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

###################################
######### LOAD DATASETS ###########
###################################

# load cifar-10 dataset
def load_cifar10(color_format='rgb'):
    data_dir = os.path.join(os.getcwd(), 'datasets/cifar-10')

    # load train data
    for i in range(1, 6):
        train_batch = os.path.join(data_dir, 'data_batch_' + str(i))
        with open(train_batch, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
            x_batch = np.transpose(np.reshape(dict['data'], (10000, 3, 32, 32)), (0, 2, 3, 1))
            y_batch = np.array(dict['labels'])

            if i == 1:
                x_train = x_batch
                y_train = y_batch
            else:
                x_train = np.concatenate((x_train, x_batch))
                y_train = np.concatenate((y_train, y_batch))

    # load test data
    test_batch = os.path.join(data_dir, 'test_batch')
    with open(test_batch, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
        x_test = np.transpose(np.reshape(dict['data'], (10000, 3, 32, 32)), (0, 2, 3, 1))
        y_test = np.array(dict['labels'])

    # cast
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
      
    # color format
    if color_format == 'gray':
        x_train = rgb2grey(x_train)[:,:,:,None]
        x_test = rgb2grey(x_test)[:,:,:,None]

    elif color_format == 'hsv':
        x_train = rgb2hsv(x_train)
        x_test = rgb2hsv(x_test)
    
    elif color_format == 'hv':
        x_train = rgb2hsv(x_train)[:, :, :, [0,2]]
        x_test = rgb2hsv(x_test)[:, :, :, [0,2]]


    # labels 
    class_labels = ['airplane', 'auto.', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  

    return (x_train, y_train), (x_test, y_test), class_labels

# load cifar-100 dataset
def load_cifar100(color_format='rgb'):
    data_dir = os.path.join(os.getcwd(), 'datasets/cifar-100')

    # load train data
    train_batch = os.path.join(data_dir, 'train')
    with open(train_batch, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
        x_train = np.transpose(np.reshape(dict['data'], (-1, 3, 32, 32)), (0, 2, 3, 1))
        y_train = np.array(dict['coarse_labels'])

    # load test data
    test_batch = os.path.join(data_dir, 'test')
    with open(test_batch, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
        x_test = np.transpose(np.reshape(dict['data'], (-1, 3, 32, 32)), (0, 2, 3, 1))
        y_test = np.array(dict['coarse_labels'])

    # cast
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
      
    # color format
    if color_format == 'gray':
        x_train = rgb2grey(x_train)[:,:,:,None]
        x_test = rgb2grey(x_test)[:,:,:,None]

    elif color_format == 'hsv':
        x_train = rgb2hsv(x_train)
        x_test = rgb2hsv(x_test)
    
    elif color_format == 'hv':
        x_train = rgb2hsv(x_train)[:, :, :, [0,2]]
        x_test = rgb2hsv(x_test)[:, :, :, [0,2]]

    
    # labels 
    class_labels = ['aquatic m', 'fish', 'flowers', 'food containers', 'fruit/veggies', 'electric', 'furniture', 'insects', 'carnivores', 'man made', 'omnivores', 'mammals', 'invertebrates', 'people', 'reptiles', 'sm mammals', 'trees', 'vehicles 1', 'vehicles 2']  

    return (x_train, y_train), (x_test, y_test), class_labels  

# load emnist dataset
def load_emnist():

    # load training data
    train = pd.read_csv("datasets/emnist-balanced-train.csv")
    x_train = train.values[:,1:].reshape(-1, 28, 28)
    x_train = np.transpose(x_train, (0,2,1))
    y_train = train.values[:,0]
    
    # load testing data
    test = pd.read_csv("datasets/emnist-balanced-test.csv")
    x_test = test.values[:,1:].reshape(-1, 28, 28)
    x_test = np.transpose(x_test, (0,2,1))
    y_test = test.values[:,0]
    
    # cast
    x_train = x_train.astype('float32')[:,:,:,None]
    x_test = x_test.astype('float32')[:,:,:,None]

    # labels
    class_labels = ['0','1','2','3','4','5','6','7','8','9', 
                    'a','b','c','d','e','f','g','h','i','j',
                    'k','l','m','n','o','p','q','r','s','t',
                    'u','v','w','x','y','z','A','B','D','E',
                    'F','G','H','N','Q','R','T']  
    
    return (x_train, y_train), (x_test, y_test), class_labels
    
# load mnist dataset
def load_mnist():

    # load from already required keras
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # cast
    x_train = x_train.astype('float32')[:,:,:,None]
    x_test = x_test.astype('float32')[:,:,:,None]
    
    # labels
    class_labels = ['0','1','2','3','4','5','6','7','8','9']  

    return (x_train, y_train), (x_test, y_test), class_labels


# load svhn dataset
def load_svhn(color_format='rgb'):

    # load
    train_data = sio.loadmat('datasets/svhn/train_32x32.mat')
    x_train = np.transpose(train_data['X'], (3,0,1,2))
    y_train = np.squeeze(train_data['y']) - 1
    test_data = sio.loadmat('datasets/svhn/test_32x32.mat')
    x_test = np.transpose(test_data['X'], (3,0,1,2))
    y_test = np.squeeze(test_data['y']) - 1

    extra_data = sio.loadmat('datasets/svhn/extra_32x32.mat')
    x_extra = np.transpose(extra_data['X'], (3,0,1,2))
    y_extra = np.squeeze(extra_data['y']) - 1

    x_train = np.concatenate((x_train, x_extra[:10000]), axis=0)
    y_train = np.concatenate((y_train, y_extra[:10000]), axis=0)
    

    # cast
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    # color format
    if color_format == 'gray':
        x_train = rgb2grey(x_train)[:,:,:,None]
        x_test = rgb2grey(x_test)[:,:,:,None]

    elif color_format == 'hsv':
        x_train = rgb2hsv(x_train)
        x_test = rgb2hsv(x_test)
    
    elif color_format == 'hv':
        x_train = rgb2hsv(x_train)[:, :, :, [0,2]]
        x_test = rgb2hsv(x_test)[:, :, :, [0,2]]

    # want class to match digit
    y_train += 1
    y_test += 1
    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0
    
    #label
    #class_labels = ['one','two','three','four','five','six','seven','eight','nine','zero']  
    class_labels = ['0','1','2','3','4','5','6','7','8','9']  
    #class_labels = ['\'1\'','\'2\'','\'3\'','\'4\'','\'5\'','\'6\'','\'7\'','\'8\'','\'9\'','\'0\''] 

    return (x_train, y_train), (x_test, y_test), class_labels

def load_core50(schedule_flag, configs, args, color_format='rgb'):

    pkl_file = open('datasets/core50.p', 'rb')
    data = pickle.load(pkl_file)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    class_labels = ['plug', 'phone', 'sciccors', 'light_bulb', 'can', 'sun_glasses', 'ball', 'marker', 'cup', 'remote']


    if schedule_flag == 3:
        train_sessions = [1,2,4,5,6,8,9,11]
        test_sessions = [3,7,10]

        #object_order = [1,  2,  4,  10, 8,  3,  5,  6,  7,  9,
        #                11, 12, 14, 20, 18, 13, 15, 16, 17, 19,
        #                21, 22, 24, 30, 28, 23, 25, 26, 27, 29,
        #                31, 32, 34, 40, 38, 33, 35, 36, 37, 39,
        #                41, 42, 44, 50, 48, 43, 45, 46, 47, 49]
        
        object_order = [1,  6, 16, 46, 36, 11, 21, 26, 31, 40,
                        2,  7, 17, 47, 37, 12, 22, 27, 32, 41,
                        3,  8, 18, 48, 38, 13, 23, 28, 33, 42,
                        4,  9, 19, 49, 39, 14, 24, 29, 34, 43,
                        5, 10, 20, 50, 40, 15, 25, 30, 35, 44]
              
        x_train = {}
        x_test = {}

        y_train = {}
        y_test = {}

        for s, session in enumerate(train_sessions):
            x_train[s] = {}
            y_train[s] = {}

            for i, obj in enumerate(object_order):
                temp = []
                for j in range(len(data[session][obj])):
                    temp.append(cv2.resize(data[session][obj][j], (64, 64)))
                x_train[s][i] = np.array(temp)
                y_train[s][i] = np.array([i for x in range(len(data[session][obj]))])

        for s, session in enumerate(test_sessions):
            x_test[s] = {}
            y_test[s] = {}

            for i, obj in enumerate(object_order):
                temp = []
                for j in range(len(data[session][obj])):
                    temp.append(cv2.resize(data[session][obj][j], (64, 64)))

                x_test[s][i] = np.array(temp)
                y_test[s][i] = np.array([i for x in range(len(data[session][obj]))])


    if color_format == 'gray':
        for session in range(len(train_sessions)):
            for i in range(50):
                x_train[session][i] = rgb2grey(x_train[session][i])[:,:,:,None]
        
        for session in range(len(test_sessions)):
            for i in range(50):
                x_test[session][i] = rgb2grey(x_test[session][i])[:,:,:,None]

    elif color_format == 'hsv':
        for session in range(len(train_sessions)):
            for i in range(50):
                x_train[session][i] = rgb2hsv(x_train[session][i])
        
        for session in range(len(test_sessions)):
            for i in range(50):
                x_test[session][i] = rgb2hsv(x_test[session][i])

    elif color_format == 'hv':
        for session in range(len(train_sessions)):
            for i in range(50):
                x_train[session][i] = rgb2hsv(x_train[session][i])[:, :, :, [0,2]]
        
        for session in range(len(test_sessions)):
            for i in range(50):
                x_test[session][i] = rgb2hsv(x_test[session][i])[:, :, :, [0,2]]

    for session in range(len(train_sessions)):
        for i in range(50):
            configs, x_train[session][i], _ = normalization(configs, args, x_train[session][i], x_train[session][i])
    
    for session in range(len(test_sessions)):
        for i in range(50):
            configs, _, x_test[session][i] = normalization(configs, args, x_test[session][i], x_test[session][i])
    
    return (x_train, y_train), (x_test, y_test), class_labels

def load_tiny_imagenet(gray_flag=True, hsv_flag = False):
    x_train = np.zeros((100000, 64, 64, 3))
    x_test = np.zeros((10000, 64, 64, 3))
    y_train = []
    y_test = []

    label = 0
    i = 0
    # Load Training Data
    for d in os.listdir(os.getcwd() + '/datasets/tiny-imagenet-200/train'):
        for im in os.listdir(os.getcwd() + '/datasets/tiny-imagenet-200/train/' + d + '/images'):
            image = imageio.imread(os.getcwd() + '/datasets/tiny-imagenet-200/train/' + d + '/images/' + im)
            if image.shape != (64, 64, 3):
                image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            x_train[i] = image
            y_train.append(label)
            i += 1
        label += 1
    
    # Load Validation Data
    match_dict = {}
    label_dict = {}
    counter = 0

    f = open(os.getcwd() + '/datasets/tiny-imagenet-200/val/val_annotations.txt', 'r')
    line = f.readline()

    while line:
        im_file = line.split('\t')[0]
        code = line.split('\t')[1]

        if code in match_dict:
            label_dict[im_file] = match_dict[code]
        else:
            match_dict[code] = counter
            label_dict[im_file] = match_dict[code]
            counter += 1
        
        line = f.readline()

    for i, im in enumerate(os.listdir(os.getcwd() + '/datasets/tiny-imagenet-200/val/images')):
        image = imageio.imread(os.getcwd() + '/datasets/tiny-imagenet-200/val/images/' + im)
        if image.shape != (64, 64, 3):
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        label = label_dict[im]
        x_test[i] = image
        y_test.append(label)

    
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # color format
    if color_format == 'gray':
        x_train = rgb2grey(x_train)[:,:,:,None]
        x_test = rgb2grey(x_test)[:,:,:,None]

    elif color_format == 'hsv':
        x_train = rgb2hsv(x_train)
        x_test = rgb2hsv(x_test)
    
    elif color_format == 'hv':
        x_train = rgb2hsv(x_train)[:, :, :, [0,2]]
        x_test = rgb2hsv(x_test)[:, :, :, [0,2]]

    return (x_train, y_train), (x_test, y_test), np.arange(200)


# loads the dataset depending on experiment arguments
# includes dataset normalization
def load_dataset(configs, args):
    color_format = args.color_format
    # load dataset
    if args.dataset == 'mnist':
        (x_, y_), (x_2, y_2), class_labels = load_mnist()
        configs['im_size'] = 28 
        configs['channels'] = 1
        configs['num_phases'] = 5
        configs['class_labels'] = class_labels
        num_classes = 10
    if args.dataset == 'emnist':
        (x_, y_), (x_2, y_2), class_labels = load_emnist()
        configs['im_size'] = 28 
        configs['channels'] = 1
        configs['num_phases'] = 23
        configs['class_labels'] = class_labels
        num_classes = 47
    if args.dataset == 'cifar-10':
        (x_, y_), (x_2, y_2), class_labels = load_cifar10(color_format=args.color_format)
        
        configs['im_size'] = 32

        if color_format == 'gray':
            configs['channels'] = 1
        elif color_format  == 'hv':
            configs['channels'] = 2
        else:
            configs['channels'] = 3
        
        configs['num_phases'] = 5
        configs['class_labels'] = class_labels
        num_classes = 10
    
    if args.dataset == 'cifar-100':
        (x_, y_), (x_2, y_2), class_labels = load_cifar100(color_format=args.color_format)
        configs['im_size'] = 32 

        if color_format == 'gray':
            configs['channels'] = 1
        elif color_format  == 'hv':
            configs['channels'] = 2
        else:
            configs['channels'] = 3
        
        configs['num_phases'] = 10
        configs['class_labels'] = class_labels
        num_classes = 20
    if args.dataset == 'svhn':
        (x_, y_), (x_2, y_2), class_labels = load_svhn(color_format=args.color_format)
        configs['im_size'] = 32 

        if color_format == 'gray':
            configs['channels'] = 1
        elif color_format  == 'hv':
            configs['channels'] = 2
        else:
            configs['channels'] = 3
        
        configs['num_phases'] = 5
        configs['class_labels'] = class_labels
        num_classes = 10
    if args.dataset == 'core50':
        (x_, y_), (x_2, y_2), class_labels = load_core50(args.schedule_flag, configs, args, color_format=args.color_format)
        configs['im_size'] = 64

        if color_format == 'gray':
            configs['channels'] = 1
        elif color_format  == 'hv':
            configs['channels'] = 2
        else:
            configs['channels'] = 3

        if args.schedule_flag == 3:
            configs['num_phases'] = 10
            num_classes = 50
        
        configs['class_labels'] = class_labels

    if args.dataset == 'tinyimagenet':
        (x_, y_), (x_2, y_2), class_labels = load_tiny_imagenet(color_format=args.color_format)
        configs['im_size'] = 64

        if color_format == 'gray':
            configs['channels'] = 1
        elif color_format  == 'hv':
            configs['channels'] = 2
        else:
            configs['channels'] = 3
        
        
        configs['num_phases'] = 20
        configs['class_labels'] = class_labels        
        num_classes = 200

#############################

    # split dataset (testing vs validation)
    x_eval = x_2
    y_eval = y_2

    if args.dataset != 'core50':
        configs, x_, x_eval = normalization(configs, args, x_, x_eval)

    # add info to configs
    configs['num_classes'] = num_classes
    configs['class_labels'] = class_labels
    configs['scale_flag'] = args.scale_flag
    configs['transfer'] = args.transfer


    return (x_, y_), (x_eval, y_eval), configs


if __name__ == "__main__":
    x, xx, y, yy, l = load_tiny_imagenet(True)