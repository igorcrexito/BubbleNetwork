from keras.layers import Input, Lambda, Conv3D, UpSampling3D, MaxPooling3D, concatenate, multiply, add
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, BatchNormalization, Add
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.models import Sequential
from keras.utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt

#two major hyperparameters of the model
bubble_dimension = 12
number_of_bubbles = 36

# Crops (or slices) a Tensor on a given dimension from start to end   
def crop_array(dimension):
    def func(x):
        return x[:,dimension]
    return Lambda(func)

#creating the bubble model
def create_bubble_network(batch_size, number_of_frames, appearance_dimension, number_of_classes):
    
    #first model input (appearance) -> appearance dimension for I3D network is (16,225)
    appearance_input = Input(batch_shape=(batch_size, number_of_frames, appearance_dimension), name='appearance_input')     
    encoding = Flatten()(appearance_input)
    
    #first model input (optical flow) -> frames with 112,112,3 dimension
    of_input = Input(batch_shape=(batch_size, number_of_frames, 112, 112, 3), name='of_input')
    of_encoding = feature_extraction_layers(of_input)
   
    #squeezing OF activations
    of_encoding = Conv3D(number_of_bubbles, (3,3,3), activation='relu', padding='same')(of_encoding)
    of_encoding = Flatten()(of_encoding)
    of_encoding = Dense(number_of_bubbles, activation = 'sigmoid')(of_encoding)
    of_encoding = Dense(number_of_bubbles, activation = 'sigmoid')(of_encoding)
    
    #reshaping both representations
    encoding = Reshape((number_of_frames, appearance_dimension))(encoding)
    of_encoding = Reshape((number_of_bubbles,1))(of_encoding)
    
    #creating a bubble layer
    bubble_layer = []
    for i in range(0, number_of_bubbles):
        bubble_layer.append(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False, name='bubble_'+ str(i)) (encoding))
    
    #duplicating bubble layer to create a branch
    bubbles = bubble_layer
    
    #excitating bubbles
    for i in range(0, number_of_bubbles):
        activation = crop_array(i)(of_encoding)
        bubble_layer[i] = multiply([activation, bubble_layer[i]])
        
    #summation to produce the branch
    for i in range(0, number_of_bubbles):
        if i == 0:
            encoding = add([bubble_layer[i],bubbles[i]])
        else:
            encoding = concatenate([encoding, add([bubble_layer[i],bubbles[i]])], axis=1)
    
    #print(np.shape(encoding))
    encoding = BatchNormalization()(encoding)
    encoding = Dense(1024, activation = 'sigmoid') (encoding)
    encoding = Dropout(0.5)(encoding)
    encoding = Dense(512, activation = 'sigmoid') (encoding)
    encoding = Dropout(0.5)(encoding)
    output = Dense(number_of_classes, activation = 'softmax') (encoding)
    
    bubblenet = Model([appearance_input, of_input], [output])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    
    bubblenet.compile(optimizer=adam, loss='categorical_crossentropy')
    
    #if you want to plot the model and/or show the summary
    #plot_model(bubblenet, to_file='model.png')
    #bubblenet.summary()
    
    return bubblenet

#feature extraction layers to act over optical flow input. Despite using Conv3D, the filters are equivalent to
#2D convolutional layers. The first dimension is used due to the batch dimension of the input
def feature_extraction_layers(input):
    encoding1 = Conv3D(32, (1,7,7), activation='relu', padding='same', strides=(1, 2, 2))(input)
    encoding2 = Conv3D(32, (1,5,5), activation='relu', padding='same', strides=(1, 2, 2))(input)
    encoding3 = Conv3D(32, (1,3,3), activation='relu', padding='same', strides=(1, 2, 2))(input)
    encoding = concatenate([encoding1, encoding2], axis=4)
    encoding = concatenate([encoding, encoding3], axis=4)
    encoding = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='same')(encoding)
    
    #residual module
    pre_encoding = Conv3D(64, (1,1,1), activation='relu', padding='same', strides=(1, 1, 1))(encoding)
    encoding = Conv3D(64, (1,3,3), activation='relu', padding='same')(encoding)
    encoding = Conv3D(64, (1,3,3), activation='relu', padding='same')(encoding)
    encoding = Add()([pre_encoding, encoding])

    #residual module
    pre_encoding = Conv3D(64, (1,1,1), activation='relu', padding='same', strides=(1, 1, 1))(encoding)
    encoding = Conv3D(64, (1,3,3), activation='relu', padding='same')(encoding)
    encoding = Conv3D(64, (1,3,3), activation='relu', padding='same')(encoding)
    encoding = Add()([pre_encoding, encoding])

    #residual module
    encoding = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='same')(encoding)
    pre_encoding = Conv3D(128, (1,1,1), activation='relu', padding='same', strides=(1, 1, 1))(encoding)
    encoding = Conv3D(128, (1,3,3), activation='relu', padding='same')(encoding)
    encoding = Conv3D(128, (1,3,3), activation='relu', padding='same')(encoding)
    encoding = Add()([pre_encoding, encoding])

    #residual module
    pre_encoding = Conv3D(128, (1,1,1), activation='relu', padding='same', strides=(1, 1, 1))(encoding)
    encoding = Conv3D(128, (1,3,3), activation='relu', padding='same')(encoding)
    encoding = Conv3D(128, (1,3,3), activation='relu', padding='same')(encoding)
    encoding = Add()([pre_encoding, encoding])

    #residual module
    encoding = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='same')(encoding)
    pre_encoding = Conv3D(256, (1,1,1), activation='relu', padding='same', strides=(1, 1, 1))(encoding)
    encoding = Conv3D(256, (1,3,3), activation='relu', padding='same')(encoding)
    encoding = Conv3D(256, (1,3,3), activation='relu', padding='same')(encoding)
    encoding = Add()([pre_encoding, encoding])
    
    #residual module
    pre_encoding = Conv3D(256, (1,1,1), activation='relu', padding='same', strides=(1, 1, 1))(encoding)
    encoding = Conv3D(256, (1,3,3), activation='relu', padding='same')(encoding)
    encoding = Conv3D(256, (1,3,3), activation='relu', padding='same')(encoding)
    encoding = Add()([pre_encoding, encoding])

    #residual module
    encoding = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='same')(encoding)
    pre_encoding = Conv3D(512, (1,1,1), activation='relu', padding='same', strides=(1, 1, 1))(encoding)
    encoding = Conv3D(512, (1,2,2), activation='relu', padding='same')(encoding)
    encoding = Conv3D(512, (1,2,2), activation='relu', padding='same')(encoding)
    encoding = Add()([pre_encoding, encoding])
    
    #residual module
    #encoding = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
    #                       border_mode='same')(encoding)
    pre_encoding = Conv3D(512, (1,1,1), activation='relu', padding='same', strides=(1, 1, 1))(encoding)
    encoding = Conv3D(512, (1,2,2), activation='relu', padding='same')(encoding)
    encoding = Conv3D(512, (1,2,2), activation='relu', padding='same')(encoding)
    encoding = Add()([pre_encoding, encoding])

    #residual module
    #encoding = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), border_mode='same')(encoding)
    pre_encoding = Conv3D(512, (1,1,1), activation='relu', padding='same', strides=(1, 1, 1))(encoding)
    encoding = Conv3D(512, (1,2,2), activation='relu', padding='same')(encoding)
    encoding = Conv3D(512, (1,2,2), activation='relu', padding='same')(encoding)
    encoding = Add()([pre_encoding, encoding])
    
    #residual module
    pre_encoding = Conv3D(512, (1,1,1), activation='relu', padding='same', strides=(1, 1, 1))(encoding)
    encoding = Conv3D(512, (1,2,2), activation='relu', padding='same')(encoding)
    encoding = Conv3D(512, (1,2,2), activation='relu', padding='same')(encoding)
    encoding = Add()([pre_encoding, encoding])
    encoding = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), border_mode='same')(encoding)
    
    return encoding
    
    
#fitting the bubble model   
def bubble_network_fit(model, main_input, appearance_input, number_of_frames, video_labels, number_of_epochs, batch_size, number_of_classes):
    
    main_input = np.array([main_input])
    video_labels = np.array([video_labels], dtype = np.int32)
    samples = np.shape(main_input)[1]
    
    input_training = np.reshape(main_input, (samples, number_of_frames, 112, 112, 3))
    appearance_input = np.reshape(appearance_input, (samples, number_of_frames, 225)) #225 is the appearance dimension for I3D
    video_labels = np.reshape(video_labels, (samples, number_of_classes))
    
    model.fit([appearance_input, input_training], [video_labels],
                epochs=number_of_epochs,
                batch_size=batch_size,
                shuffle=True)
                
    return model
