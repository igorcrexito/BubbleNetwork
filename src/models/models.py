from keras.layers import Input, Lambda, Conv3D, UpSampling3D, MaxPooling3D, concatenate, multiply
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, BatchNormalization, Add, AveragePooling3D
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.layers.advanced_activations import ELU
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import LocallyConnected2D
from keras.models import Sequential
from keras.utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt

batch_input_size = 10
bubble_dimension = 32
number_of_bubbles = 32

def create_bubble_model2(batch_size, number_of_timesteps, clip_size, appearance_representation_size, number_of_classes):
    
    #specifying input details -> image size is 4x4x112x112x3
    main_input = Input(batch_shape=(batch_size, number_of_timesteps, clip_size, 112, 112, 3), name='main_input1')
    
    tensor1 = Lambda(slice_tensor1)(main_input)
    tensor2 = Lambda(slice_tensor2)(main_input)
    tensor3 = Lambda(slice_tensor3)(main_input)
    tensor4 = Lambda(slice_tensor4)(main_input)
    
    encoding1 = Lambda(convolutional_blocks)(tensor1)
    encoding2 = Lambda(convolutional_blocks)(tensor2)
    encoding3 = Lambda(convolutional_blocks)(tensor3)
    encoding4 = Lambda(convolutional_blocks)(tensor4)
    
    merge = concatenate([encoding1, encoding2, encoding3, encoding4], axis=1)
    #print(np.shape(merge)) batch, timesteps, width, height, maps
                           
    encoding = Reshape((number_of_timesteps, 4096))(merge)
    secondary_input = Input(batch_shape=(batch_size, number_of_timesteps, appearance_representation_size), name='appearance_input')
    
    secondary = Reshape((number_of_timesteps, 4096))(secondary_input)
    
    merge = concatenate([encoding, secondary], axis = 2)
    
    #print(np.shape(merge))
    
    #bubble layer
    bubble1 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble2 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble3 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble4 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble5 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble6 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble7 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble8 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble9 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble10 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble11 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble12 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble13 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble14 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble15 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble16 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble17 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble18 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble19 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble20 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble21 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble22 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble23 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble24 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble25 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble26 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble27 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble28 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble29 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble30 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble31 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble32 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble33 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble34 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble35 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble36 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble37 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble38 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble39 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble40 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble41 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble42 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble43 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble44 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble45 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble46 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble47 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble48 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble49 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble50 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble51 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble52 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble53 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble54 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble55 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble56 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble57 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble58 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble59 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble60 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble61 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble62 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble63 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble64 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble65 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble66 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble67 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble68 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble69 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble70 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble71 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble72 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble73 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble74 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble75 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble76 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble77 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble78 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble79 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble80 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble81 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble82 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble83 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble84 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble85 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble86 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble87 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble88 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble89 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble90 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble91 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble92 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble93 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble94 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble95 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble96 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble97 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble98 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble99 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble100 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble101 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble102 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble103 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble104 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble105 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble106 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble107 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble108 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble109 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble110 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble111 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble112 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble113 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble114 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble115 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble116 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble117 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble118 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble119 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble120 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble121 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble122 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble123 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble124 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble125 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble126 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble127 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble128 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    
    
    encoding = concatenate([bubble1,bubble2,bubble3,bubble4,bubble5,bubble6,bubble7,bubble8,bubble9,bubble10,bubble11,bubble12,bubble13,bubble14,bubble15, bubble16,
    bubble17,bubble18,bubble19,bubble20,bubble21,bubble22,bubble23,bubble24,bubble25,bubble26,bubble27,bubble28,bubble29,bubble30,bubble31, bubble32, bubble33,
    bubble34, bubble35, bubble36, bubble37, bubble38, bubble39, bubble40, bubble41, bubble42, bubble43, bubble44, bubble45, bubble46, bubble47, bubble48,
    bubble49, bubble50, bubble51, bubble52, bubble53, bubble54, bubble55, bubble56, bubble57, bubble58, bubble59, bubble60, bubble61, bubble62, bubble63, bubble64,
    bubble65, bubble66, bubble67, bubble68, bubble69, bubble70, bubble71, bubble72, bubble73, bubble74, bubble75, bubble76, bubble77, bubble78, bubble79, bubble80,
    bubble81, bubble82, bubble83, bubble84, bubble85, bubble86, bubble87, bubble88, bubble89, bubble90, bubble91, bubble92, bubble93, bubble94, bubble95,
    bubble96, bubble97, bubble98, bubble99, bubble100, bubble101, bubble102, bubble103, bubble104, bubble105, bubble106, bubble107, bubble108, bubble109,
    bubble110, bubble111, bubble112, bubble113, bubble114, bubble115, bubble116, bubble117, bubble118, bubble119, bubble120, bubble121, bubble122, bubble123,
    bubble124, bubble125, bubble126, bubble127, bubble128], axis = 1)
    
    encoding = Dense(2048, activation = 'sigmoid') (encoding)
    encoding = Dense(1024, activation = 'sigmoid') (encoding)
    encoding = Dense(512, activation = 'sigmoid') (encoding)
    output = Dense(number_of_classes, activation = 'softmax') (encoding)
    
    bubblenet = Model([main_input, secondary_input], [output])
    bubblenet.compile(optimizer='nadam', loss='categorical_crossentropy')
    plot_model(bubblenet, to_file='model.png')
    bubblenet.summary()
    
    return bubblenet

def create_bubble_model(batch_size, number_of_timesteps, clip_size, appearance_representation_size, number_of_classes):
    
    #specifying input details -> image size is 8x112x112x3
    main_input = Input(batch_shape=(batch_size, number_of_timesteps, clip_size, 112, 112, 3), name='main_input1')
    
    tensor1 = Lambda(slice_tensor1)(main_input)
    tensor2 = Lambda(slice_tensor2)(main_input)
    tensor3 = Lambda(slice_tensor3)(main_input)
    tensor4 = Lambda(slice_tensor4)(main_input)
    
    encoding1 = Lambda(convolutional_blocks)(tensor1)
    encoding2 = Lambda(convolutional_blocks)(tensor2)
    encoding3 = Lambda(convolutional_blocks)(tensor3)
    encoding4 = Lambda(convolutional_blocks)(tensor4)
    
    merge = concatenate([encoding1, encoding2, encoding3, encoding4], axis=1)
    #print(np.shape(merge)) batch, timesteps, width, height, maps
                           
    encoding = Reshape((number_of_timesteps, 4096))(merge)
    secondary_input = Input(batch_shape=(batch_size, number_of_timesteps, clip_size, appearance_representation_size), name='appearance_input')
    
    secondary = Reshape((number_of_timesteps, 16384))(secondary_input)
    
    merge = concatenate([encoding, secondary], axis = 2)
    
    #print(np.shape(merge))
    
    #bubble layer
    bubble1 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble2 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble3 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble4 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble5 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble6 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble7 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble8 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble9 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble10 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble11 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble12 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble13 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble14 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble15 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble16 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble17 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble18 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble19 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble20 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble21 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble22 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble23 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble24 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble25 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble26 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble27 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble28 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble29 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble30 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble31 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    bubble32 = Bidirectional(SimpleRNN(bubble_dimension, return_sequences=False, stateful=False, return_state=False)) (merge)
    
    encoding = concatenate([bubble1,bubble2,bubble3,bubble4,bubble5,bubble6,bubble7,bubble8,bubble9,bubble10,bubble11,bubble12,bubble13,bubble14,bubble15, bubble16,
    bubble17,bubble18,bubble19,bubble20,bubble21,bubble22,bubble23,bubble24,bubble25,bubble26,bubble27,bubble28,bubble29,bubble30,bubble31, bubble32], axis = 1)
    
    encoding = Dense(2048, activation = 'sigmoid') (encoding)
    encoding = Dense(1024, activation = 'sigmoid') (encoding)
    encoding = Dense(512, activation = 'sigmoid') (encoding)
    output = Dense(number_of_classes, activation = 'softmax') (encoding)
    
    bubblenet = Model([main_input, secondary_input], [output])
    bubblenet.compile(optimizer='nadam', loss='categorical_crossentropy')
    plot_model(bubblenet, to_file='model.png')
    bubblenet.summary()
    
    return bubblenet


def slice_tensor1(input):
    return input[:,0,:]

def slice_tensor2(input):
    return input[:,1,:]

def slice_tensor3(input):
    return input[:,2,:]

def slice_tensor4(input):
    return input[:,3,:]

def crop_array(dimension):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        return x[:,dimension]
    return Lambda(func)


def create_bubble_network(batch_size, number_of_frames, number_of_classes):
    
    #appearance_input = Input(batch_shape=(batch_size, number_of_frames, 112, 112, 3), name='appearance_input') #input 256x256x32
    #encoding = feature_extraction_layers(appearance_input)
    
    appearance_input = Input(batch_shape=(batch_size, number_of_frames, 4096), name='appearance_input') #input 256x256x32
    encoding = Flatten()(appearance_input)
    #encoding = Dense(512, activation = 'sigmoid')(encoding)
    
    of_input = Input(batch_shape=(batch_size, number_of_frames, 112, 112, 3), name='of_input') #input 256x256x32
    of_encoding = feature_extraction_layers(of_input)
    
    #squeezing OF activations
    of_encoding = Conv3D(number_of_bubbles, (3,3,3), activation='relu', padding='same')(of_encoding)
    #of_encoding = AveragePooling3D(pool_size=(number_of_frames, 2, 2), strides=(number_of_frames, 2, 2), border_mode='same', name='pool5')(of_encoding)
    of_encoding = Flatten()(of_encoding)
    of_encoding = Dense(number_of_bubbles, activation = 'sigmoid')(of_encoding)
    
    #print(np.shape(encoding))
    #reshaping both representations
    encoding = Reshape((number_of_frames, 4096))(encoding)
    of_encoding = Reshape((number_of_bubbles,1))(of_encoding)
    #print(np.shape(of_encoding))
    
    #creating a bubble layer
    bubble_layer = []
    for i in range(0, number_of_bubbles):
        bubble_layer.append(LSTM(bubble_dimension, return_sequences=False, stateful=False, return_state=False) (encoding))
    
    #excitating bubbles
    for i in range(0, number_of_bubbles):
        activation = crop_array(i)(of_encoding)
        bubble_layer[i] = multiply([activation, bubble_layer[i]])

    for i in range(0, number_of_bubbles):
        if i == 0:
            encoding = bubble_layer[i]
        else:
            encoding = concatenate([encoding, bubble_layer[i]], axis=1)
    
    #print(np.shape(encoding))
    encoding = Dropout(0.25)(encoding)
    encoding = Dense(512, activation = 'sigmoid') (encoding)
    encoding = Dense(128, activation = 'sigmoid') (encoding)
    output = Dense(number_of_classes, activation = 'softmax') (encoding)
    
    bubblenet = Model([appearance_input, of_input], [output])
    
    #rmsprop = RMSprop(lr=0.001, rho=0.9)
    adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)
    
    bubblenet.compile(optimizer=adam, loss='categorical_crossentropy')
    #plot_model(bubblenet, to_file='model.png')
    bubblenet.summary()
    
    return bubblenet
    
def feature_extraction_layers(input):
    encoding = Conv3D(32, (1,7,7), activation='relu', padding='same', strides=(1, 2, 2))(input)
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
    encoding = Conv3D(512, (1,3,3), activation='relu', padding='same')(encoding)
    encoding = Conv3D(512, (1,3,3), activation='relu', padding='same')(encoding)
    encoding = Add()([pre_encoding, encoding])

    #residual module
    pre_encoding = Conv3D(512, (1,1,1), activation='relu', padding='same', strides=(1, 1, 1))(encoding)
    encoding = Conv3D(512, (1,3,3), activation='relu', padding='same')(encoding)
    encoding = Conv3D(512, (1,3,3), activation='relu', padding='same')(encoding)
    encoding = Add()([pre_encoding, encoding])
    
    encoding = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='same')(encoding)
    
    return encoding

def convolutional_blocks(input):
    encoding = Conv3D(32, (3,3,3), activation='relu', padding='same')(input)
    encoding = Conv3D(32, (3,3,3), activation='relu', padding='same')(encoding)
    encoding = Conv3D(32, (3,3,3), activation='relu', padding='same')(encoding)
    
    encoding = Conv3D(64, (3,3,3), activation='relu', padding='same')(encoding)
    encoding = Conv3D(64, (3,3,3), activation='relu', padding='same')(encoding)
    encoding = Conv3D(64, (3,3,3), activation='relu', padding='same')(encoding)
    encoding = MaxPooling3D(pool_size=(1, 2, 2), strides=(2, 2, 2),
                           border_mode='same', name='pool2')(encoding)
                           
    encoding = Conv3D(128, (3,3,3), activation='relu', padding='same')(encoding)
    encoding = Conv3D(128, (3,3,3), activation='relu', padding='same')(encoding)
    encoding = Conv3D(128, (3,3,3), activation='relu', padding='same')(encoding)
    encoding = BatchNormalization()(encoding)
    encoding = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='same', name='pool3')(encoding)
                           
    encoding = Conv3D(256, (3,3,3), activation='relu', padding='same')(encoding)
    encoding = Conv3D(256, (3,3,3), activation='relu', padding='same')(encoding)
    encoding = Conv3D(256, (3,3,3), activation='relu', padding='same')(encoding)
    encoding = BatchNormalization()(encoding)
    encoding = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='same', name='pool4')(encoding)
                           
    encoding = Conv3D(256, (3,3,3), activation='relu', padding='same')(encoding)
    encoding = Conv3D(256, (3,3,3), activation='relu', padding='same')(encoding)
    encoding = Conv3D(256, (3,3,3), activation='relu', padding='same')(encoding)
    encoding = BatchNormalization()(encoding)
    encoding = MaxPooling3D(pool_size=(1, 4, 4), strides=(1, 4, 4),
                           border_mode='same', name='pool5')(encoding)
    
    return encoding
    
def bubble_fit(model, main_input, appearance_input, number_of_timesteps, clip_size, video_labels, number_of_epochs, batch_size):
    
    main_input = np.array([main_input])
    appearance_input = np.array([appearance_input])
    video_labels = np.array([video_labels], dtype = np.int32)
    samples = np.shape(main_input)[1]
    
    input_training = np.reshape(main_input, (samples, number_of_timesteps, clip_size, 112, 112, 3))
    appearance_input = np.reshape(appearance_input, (samples, number_of_timesteps, 4096))
    video_labels = np.reshape(video_labels, (samples, 51))
    
    model.fit([input_training, appearance_input], [video_labels],
                epochs=number_of_epochs,
                batch_size=batch_size,
                shuffle=True,
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
                
    return model


def bubble_network_fit(model, main_input, appearance_input, number_of_frames, video_labels, number_of_epochs, batch_size, number_of_classes):
    
    main_input = np.array([main_input])
    #appearance_input = np.array([appearance_input])
    video_labels = np.array([video_labels], dtype = np.int32)
    samples = np.shape(main_input)[1]
    
    input_training = np.reshape(main_input, (samples, number_of_frames, 112, 112, 3))
    appearance_input = np.reshape(appearance_input, (samples, number_of_frames, 4096))
    video_labels = np.reshape(video_labels, (samples, number_of_classes))
    
    model.fit([appearance_input, input_training], [video_labels],
                epochs=number_of_epochs,
                batch_size=batch_size,
                shuffle=True)
                
    return model