import os
from keras import utils as np_utils
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pprint
import numpy as np
import models.models as bm
import cv2
import glob
import keras
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import utils.filereader as fr
import utils.utils as utils
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.models import Model

batch_size = 20
appearance_representation_size = 225 #size of activations of i3D network
number_of_classes = 20 #number of dataset classes
number_of_frames = 16
load_mode = 1 #set this value to 1 to load a pre-trained model
path_to_load_model = '' #set this path to find the model file

if __name__ == "__main__":
  
   #store your data on these arrays
   input_appearance_list_train = [] #(# of training samples, number_of_frames, appearance_representation_size)
   input_appearance_list_test = [] #(# of testing samples, number_of_frames, appearance_representation_size)
   input_flow_list_train = [] #(# of training samples, number_of_frames, width, height, channels)
   input_flow_list_test = [] #(# of testing samples, number_of_frames, width, height, channels)
   video_labels = [] #(# of samples, number_of_classes) #this array must be categorical 
  
   #you can create a model from scratch or load a trained model
   if load_mode == 0:
        bubble_model = bm.create_bubble_network(batch_size, number_of_frames, appearance_representation_size, new_classes) #creating a model
   else:
        bubble_model = utils.load_weights(path_to_load_model)
        #bubble_model.compile(optimizer=Adam(lr=0.0005, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy') #use this line if u need to compile the model
   
   #if you need to fit the model, use this. The dimensions shown below are related to our already trained model. You can train from scratch with different values
   bubble_model = bm.bubble_network_fit(bubble_model, input_flow_list_train, input_appearance_list_train, number_of_frames, output_label_list_train, number_of_epochs, batch_size, number_of_classes)
   
   #to predict samples -> parameters are (samples, 16, 225), (# of samples, 16, 112, 112, 3), batch_size
   predictions = bubble_model.predict([input_appearance_list_test, input_flow_list_test], batch_size)

