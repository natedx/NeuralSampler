# __________________________________________________________________________________Imports

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import soundfile as sf
import sys

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

'''
Use model_launch in run to launch the learning

define a model to use then use it in model_launch
'''

# _______________________________________________________ convolutionnal autoencoder model

# convolutionnal, 2 layers. Supposed to be good on images but is extremely slow
def convolutionnal_autoencod(nb_buffs):
    input_img_ce = Input(shape=(1, 4*nb_buffs, 256))

    x_ce = Conv2D(4, (3, 3), activation='relu', padding='same', data_format='channels_first')(input_img_ce)
    encoded_ce = MaxPooling2D((4, 4), padding='same', data_format='channels_first')(x_ce)
    x_ce = UpSampling2D((4, 4), data_format='channels_first')(encoded_ce)
    decoded_ce = Conv2D(1, (3, 3), activation='sigmoid', padding='same', data_format='channels_first')(x_ce)

    autoencoder_ce = Model(input_img_ce, decoded_ce)
    encod_ce = Model(input_img_ce, encoded_ce)

    # decod_ce = Model(encoded_ce, decoded_ce)

    return autoencoder_ce, encod_ce


# _______________________________________________________ traditionnal autoencoder model


# _______________________________________________________ learning

def model_launch(image_train, image_test, model, encod, nb_epochs):
    '''
    :param image_train: training dataset
    :param image_test: testing dataset
    :param model: model to compile and train
    :param encod: encoder linked
    :param decod: decoder linked
    :param nb_epochs:
    :return: void
    '''
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    encod.compile(optimizer='adadelta', loss='binary_crossentropy')
    # decod.compile(optimizer='adadelta', loss='binary_crossentropy')
    print(model.summary())
    model.fit(image_train, image_train, shuffle=True, epochs=nb_epochs, validation_data=(image_test, image_test))
