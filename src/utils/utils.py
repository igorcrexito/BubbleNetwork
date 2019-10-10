import numpy as np
from keras.models import model_from_json
from keras import backend as K
import operator

def uncategorize_array(array):
    array_lenght = len(array) #gathering the array lenght
    new_array = np.zeros((array_lenght,1))
    
    for i in range (0, array_lenght):
        max_index = get_max_index(array[i])
        new_array[i] = max_index
        
    return new_array

def get_max_index(array):
    max_value = 0
    max_index = 0
    
    for i in range(0, len(array)):
        if array[i]>max_value:
            max_value = array[i]
            max_index = i
    
    return max_index