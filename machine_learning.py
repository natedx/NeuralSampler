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
    input = Input(shape=(1, 4*nb_buffs, 256))

    x = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(input)
    x = MaxPooling2D((2, 2), padding='same', data_format='channels_first')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', data_format='channels_first')(x)
    x = MaxPooling2D((2, 2), padding='same', data_format='channels_first')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', data_format='channels_first')(x)
    encoded = MaxPooling2D((2, 2), padding='same', data_format='channels_first')(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same', data_format='channels_first')(encoded)
    x = UpSampling2D((2, 2), data_format='channels_first')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', data_format='channels_first')(x)
    x = UpSampling2D((2, 2), data_format='channels_first')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(x)
    x = UpSampling2D((2, 2), data_format='channels_first')(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', data_format='channels_first')(x)

    autoencoder_ce = Model(input, decoded)
    encod_ce = Model(input, encoded)

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
