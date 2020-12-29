import os # provides functions for interacting with operating envrionment
from glob import glob # useful for retrive files/pathnames 
import random # implements pseudo-random number generators for various distributions.
import time # provide system time related functions
import tensorflow # tensorflow packagae for neural network modules
import datetime # The datetime module supplies classes for manipulating dates and times.


from tqdm import tqdm # gives progress bar 

import numpy as np # numpy module for scientific data calculations 
import pandas as pd # package for operting on dataframes 
from IPython.display import FileLink # Class for embedding a local file link in an IPython session, based on path
import matplotlib.pyplot as plt # datavisualization package
import warnings # to show warnings
warnings.filterwarnings('ignore') 
import seaborn as sns # datavisualization package
%matplotlib inline
from IPython.display import display, Image # display module presents Public API for display tools in IPython.
import matplotlib.image as mpimg 
import cv2 # OpenCV-Python is a library of Python bindings designed to solve computer vision problems.

from sklearn.model_selection import train_test_split # Split arrays or matrices into random train and test subsets
from sklearn.datasets import load_files   # load datasets  text files with categories as subfolder names.    
from keras.utils import np_utils # numpy utilities in keras
from sklearn.utils import shuffle # Shuffle arrays or sparse matrices in a consistent way
from sklearn.metrics import log_loss # calculates the negative log-likelihood of a logistic model that returns y_pred probabilities for its training data y_true

from keras.models import Sequential, Model # A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D # multiple hidden layers for building network graph
from keras.preprocessing.image import ImageDataGenerator # Generate batches of tensor image data with real-time data augmentation.
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping # ModelCheckpoint - to save a model or weights;  EarlyStopping- Stop training when a monitored metric has stopped improving.
from keras.applications.vgg16 import VGG16 # Keras Applications are deep learning models that  available alongside pre-trained weights which be used for prediction, feature extraction, and fine-tu



