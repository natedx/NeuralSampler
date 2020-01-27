# __________________________________________________________________________________Imports

import numpy as np
import sys

import bank_creator as bkc
import sound_out as sdo
import machine_learning as mlg
import matplotlib.pyplot as plt

# __________________________________________________________________________________EXPLANATION

'''
Use the bank_creator commands to create a bank (only on my computer... Maybe you can use it nate)

Use load_solid_bank to load a bank in the directory /banks

learn with model_launch with required parameters

use reconstruct_image to get the out sound
'''

# buffer_size//2 + 1 = multiple of 16 (maxpooling(4,4)) --> 256 height
buffer_size = 511
nb_buffs = 50

# __________________________________________________________________________________Script to create large banks
'''
# path_bank = "D:/01 Musique/01 Programmes/02 NI 2/"
path_bank = "C:/Program Files (x86)/Image-Line/FL Studio 20/Data/Patches/Packs/Drums (ModeAudio)"
path_bank = "C:\Program Files (x86)\Image-Line\FL Studio 20\Data\Patches\Packs"

# words = ["Snare", "Snares", "snare", "snares"]
# words = ["Perc", "Percs", "perc", "percs"]

words = ["kick", "Kick"]

num_max = 20

# use triple quotes to pass these 3lines creating the library
filename = 'bank_n8_snares_'
path_out = './banks/' + filename
bkc.create_solid_bank(np.array(bkc.load_image_bank(path_bank, words, num_max, buffer_size, nb_buffs)), buffer_size,
                      nb_buffs, path_out)

# NB : in the banks names, there's info on the parameters to set:
# you should set nb_buffs to nb= in the name, then use in the ML section a size of 4*nb_buffs for the second dimension
# you should set buffer_size to size= in the name

'''

# __________________________________________________________________________________Import the large banks

path_selected_bank = 'banks/bank_percssize=511nb=50.npy'
bank = bkc.load_solid_bank(path_selected_bank)
print('----------- LOADED')

plt.matshow(bank[0,0], cmap="magma")

plt.show()

# __________________________________________________________________________________learning


# you can load several banks to learn on more data

print(np.array(bank).shape)
# keep 80% for training, 20% for test
from math import floor
nb_images = len(bank)
print(sys.getsizeof(bank))
selector = floor(nb_images * 0.8)
image_train = np.array(bank[:selector])
image_test = np.array(bank[selector:])

# use the selected encoder
autoencoder_ce, encod_ce = mlg.convolutionnal_autoencod(nb_buffs)

# actual learning
nb_epochs =5
mlg.model_launch(image_train, image_test, autoencoder_ce, encod_ce, nb_epochs)


# __________________________________________________________________________________test_learning quality

# small audio test on the sample i of the dataset train
path_out = './results'
samples = [i for i in range(20)]


def test_audio(path_out):
    for i in samples:
        test_image_learn = np.array([image_train[i]])
        test_image = np.array(image_train[i][0])
        predicted_image = autoencoder_ce.predict(test_image_learn)[0][0]
        print(predicted_image.shape)
        print(test_image.shape)
        lrec = sdo.reconstruct_image(test_image, nb_buffs, path_out, 'test' + str(i))
        rrec = sdo.reconstruct_image(predicted_image, nb_buffs, path_out, 'learned' + str(i))
    print('------------TEST OK')


test_audio(path_out)

