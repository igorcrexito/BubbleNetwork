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
from sklearn.metrics import classification_report, confusion_matrix
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt
from keras.models import Model

batch_size = 20
appearance_representation_size = 225 #size of activations of i3D network
number_of_classes = 20 #number of dataset classes
number_of_frames = 16
load_mode = 1 #set this value to 1 to load a pre-trained model


if __name__ == "__main__":
  
   #you can create a model from scratch or load a trained model
   if load_mode == 0:
        bubble_model = bm.create_bubble_network(batch_size, number_of_frames, appearance_representation_size, new_classes) #creating a model
   else:
        bubble_model = utils.load_weights()
        bubble_model.compile(optimizer=Adam(lr=0.0015, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy') #use this line if u need to compile the model
   
   #fitting model -> parameters are (the created model, the flow input ())
   bubble_model = bm.bubble_network_fit(bubble_model, input_flow_list_train, input_appearance_list_train, number_of_frames, output_label_list_train, number_of_epochs, batch_size, number_of_classes)
   
   #saving model weights
   #utils.save_weights(bubble_model)

   input_flow_list_test = np.array([input_flow_list_test], dtype=np.float32)
   input_appearance_list_test = np.array([input_appearance_list_test], dtype=np.float32)
   input_appearance_list_train = np.array([input_appearance_list_train], dtype=np.float32)
   input_flow_list_train = np.array([input_flow_list_train])
   input_flow_list_train = input_flow_list_train[0]
   input_appearance_list_train = input_appearance_list_train[0]
   
   samples = np.shape(input_flow_list_test)[1]
   
   input_flow_list_test = np.reshape(input_flow_list_test, (samples, number_of_timesteps* clip_size, 112, 112, 3))
   input_appearance_list_test = np.reshape(input_appearance_list_test, (samples, number_of_frames, appearance_representation_size))
   
   predictions = bubble_model.predict([input_appearance_list_test, input_flow_list_test], batch_size)
   #predictions = bubble_model.predict([input_appearance_list_train, input_flow_list_train], batch_size)
   
   #output_label_list_test = utils.uncategorize_array(output_label_list_test)
   predictions = utils.uncategorize_array(predictions)
   output_label_list_train = utils.uncategorize_array(output_label_list_train)
   accuracy = 0
   
   for i in range(0, len(predictions)):
       print(str(i) + ' ' + str(predictions[i]) + " " + str(output_label_list_test[i]))
       #if (predictions[i] == output_label_list_train[i]):
       if (predictions[i] == output_label_list_test[i]):
           accuracy+=1
           
   accuracy = accuracy/len(predictions)
   print("Final accuracy of the model is: " + str(accuracy))
    
   print('Confusion Matrix')
   print(confusion_matrix(output_label_list_test, predictions))
   utils.plot_confusion_matrix(output_label_list_test, predictions, normalize=True,
                      title='Confusion matrix, without normalization')
   
   plt.show()
