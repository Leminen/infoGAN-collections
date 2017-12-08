# -*- coding: utf-8 -*-
import os
import urllib

from tensorflow.contrib.learn.python.learn.datasets import mnist

import src.utils as utils

def make_dataset(dataset):
    dir_rawData = 'data/raw/'+ dataset
    utils.checkfolder(dir_rawData)
    
    if dataset == 'mnist':
        # Download the MNIST dataset from source and save it in 'data/raw/mnist'
        data = mnist.read_data_sets('data/raw/mnist', one_hot=True)
    
    if dataset == 'PSD_Segmented':
        # Download the Plant Seedlings Dataset Segmented from source (http://vision.eng.au.dk/data/WeedData/Segmented.zip) 
        # and save it in data/raw/Plant_Seedlings_Dataset_Segmented
        
        urllib.request.urlretrieve('http://vision.eng.au.dk/data/WeedData/Segmented.zip',
                                   'data/raw/PSD_Segmented/data.zip')
    
    if dataset == 'PSD_NonSegmented':
        # Download the Plant Seedlings Dataset NonSegmented from source (http://vision.eng.au.dk/data/WeedData/Nonsegmented.zip) 
        # and save it in data/raw/Plant_Seedlings_Dataset_NonSegmented
        
        urllib.request.urlretrieve('http://vision.eng.au.dk/data/WeedData/Nonsegmented.zip',
                                   'data/raw/PSD_NonSegmented/data.zip')
