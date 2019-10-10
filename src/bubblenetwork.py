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

batch_size = 20
clip_size = 4
number_of_timesteps = 4
appearance_representation_size = 4096
number_of_classes = 51
base_path = "D:/HMDB51/videos/"
protocol_path = "D:/HMDB51/splits/"
number_of_epochs = 50
split = 1

if __name__ == "__mains__":
    #bubble_model = bm.create_bubble_model2(batch_size, number_of_timesteps, clip_size, appearance_representation_size, number_of_classes)
    bubble_model = bm.create_bubble_network(batch_size, 32, number_of_classes)
   
if __name__ == "__main__":
  
   #generating data list to train/validate/test the model
   input_appearance_list_train = []
   input_flow_list_train = []
   output_label_list_train = []
   
   input_appearance_list_test = []
   input_flow_list_test = []
   output_label_list_test = []
   
   #retrieving all directories of the dataset
   os.chdir(base_path)
   
   #retrieving the list of activities
   list_of_activities = glob.glob("*")
   
   new_classes = 6
   
   #for i in range(0, number_of_classes):
   for i in range(0, new_classes):
        #reading optical flow videos
        
        video_path = base_path + str(list_of_activities[i]) + "/"
        print(video_path)
        protocol_list = fr.read_protocol_path(protocol_path, split, str(list_of_activities[i]))
        
        flow_list_train, flow_list_test = fr.read_flow_videos(video_path, clip_size, number_of_timesteps, protocol_list)
        appearance_list_train, appearance_list_test = fr.read_appearance_files(video_path, clip_size, number_of_timesteps, protocol_list)
        
        #print(np.shape(appearance_list_train))
        
        input_appearance_list_train.extend(appearance_list_train)
        input_appearance_list_test.extend(appearance_list_test)
        input_flow_list_train.extend(flow_list_train)
        input_flow_list_test.extend(flow_list_test)
        
        for z in range(0, len(flow_list_train)):
            output_label_list_train.append(i)
            
        for z in range(0, len(flow_list_test)):
            output_label_list_test.append(i)
        
   #showing data dimensions
   print("Data dimensions ------------------------------------------")
   print(str(np.shape(input_appearance_list_train)) + " " + str(np.shape(input_appearance_list_test)))
   print(str(np.shape(input_flow_list_train)) + " " + str(np.shape(input_flow_list_test)))
   print(str(np.shape(output_label_list_train)) + " " + str(np.shape(output_label_list_test)))
   print("----------------------------------------------------------")
   
   #bubble_model = bm.create_bubble_model2(batch_size, number_of_timesteps, clip_size, appearance_representation_size, number_of_classes)
   
   bubble_model = bm.create_bubble_network(batch_size, 16, new_classes)
   output_label_list_train = keras.utils.to_categorical(output_label_list_train, num_classes=new_classes)
   #for i in range(0, len(output_label_list_train)):
   #    print(str(i) + ' ' + str(output_label_list_train[i]))
   
   #bubble_model = bm.bubble_fit(bubble_model, input_flow_list_train, input_appearance_list_train, number_of_timesteps, clip_size, output_label_list_train, number_of_epochs, batch_size)
   bubble_model = bm.bubble_network_fit(bubble_model, input_flow_list_train, input_appearance_list_train, number_of_timesteps*clip_size, output_label_list_train, number_of_epochs, batch_size, new_classes)
   
   
   input_flow_list_test = np.array([input_flow_list_test])
   input_appearance_list_test = np.array([input_appearance_list_test])
   input_appearance_list_train = np.array([input_appearance_list_train])
   input_flow_list_train = np.array([input_flow_list_train])
   input_flow_list_train = input_flow_list_train[0]
   input_appearance_list_train = input_appearance_list_train[0]
   
   samples = np.shape(input_flow_list_test)[1]
   input_flow_list_test = np.reshape(input_flow_list_test, (samples, number_of_timesteps* clip_size, 112, 112, 3))
   input_appearance_list_test = np.reshape(input_appearance_list_test, (samples, 16, 4096))
   predictions = bubble_model.predict([input_appearance_list_test, input_flow_list_test], batch_size)
   
   #output_label_list_test = utils.uncategorize_array(output_label_list_test)
   predictions = utils.uncategorize_array(predictions)
   output_label_list_train = utils.uncategorize_array(output_label_list_train)
   accuracy = 0
   
   for i in range(0, len(predictions)):
       print(str(i) + ' ' + str(predictions[i]) + " " + str(output_label_list_test[i]))
       if (predictions[i] == output_label_list_test[i]):
           accuracy+=1
           
   accuracy = accuracy/len(predictions)
   print("Final accuracy of the model is: " + str(accuracy))
   