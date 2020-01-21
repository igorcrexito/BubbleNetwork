# BubbleNetwork

This repository contains the files to assemble the BubbleNET architecture associated to the paper "BUBBLENET: A DISPERSE RECURRENT STRUCTURE TO RECOGNIZE ACTIVITIES".

On bubblenetwork.py, you can find details on how to use a trained model (available on folder models) or train a bubble architecture from scratch. On models.py, you can find the implementation of the bubble architecture.

In order to properly use our trained models, the appearance input must be extracted with I3D RGB Network [1], available on models/i3dnetwork. This model uses a 79-frame video as input. So, perform subsample (if needed), extract these features from convolutional layer Convolutional layer 6a; and reshape to (16,225), as shown on the snippet below:

#------------------------------------------------------------------------------------------------------------#
layer_name = 'Conv3d_6a_1x1'
intermediate_layer_model = Model(inputs=rgb_model.layers[0].input, outputs=rgb_model.layers[196].output)
activations = model.predict(subsampled_video) #this video has the following dimensions (1, 79, 224, 224, 3)
activations = np.reshape(activations, (16,225))
#------------------------------------------------------------------------------------------------------------#


[1] J. Carreira and A. Zisserman, “Quo vadis, action recognition? a new model and the kinetics dataset,” in IEEE CVPR, 07 2017, pp. 4724–4733.
