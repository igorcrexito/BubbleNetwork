import numpy as np
from keras.models import model_from_json
from keras import backend as K
import operator
import matplotlib.pyplot as plt

def load_weights(path_to_model):
    json_file = open(path_to_model + "/yup_model_stationary.json", 'r') #loading YUP stationary model, for example
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights(path_to_model+ "/yup_model_stationary.h5")
 
    return loaded_model
