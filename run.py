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



# __________________________________________________________________________________learning

# creating function that splits banks into 4 subsets.
def splitter(db, nb_buffs):
    split = []
    for i in range(4):
        split.append(db[:,:,i*nb_buffs:(i+1)*nb_buffs])
    return split

def merger(a):
    return np.concatenate((a[0], a[1], a[2], a[3]), axis=2)

# you can load several banks to learn on more data

print(np.array(bank).shape)
# keep 80% for training, 20% for test
from math import floor
nb_images = len(bank)
print(sys.getsizeof(bank))
selector = floor(nb_images * 0.8)


# image_train = np.abs(bank[:selector])
# image_test = np.abs(bank[selector:])

image_train = (bank[:selector]+1)/2
image_test = (bank[selector:]+1)/2

image_test_bis = bank[:selector]
image_train_bis = bank[selector:]

# image_train = bank[:selector]
# image_test = bank[selector:]


image_train_split = splitter(image_train, 50)
image_test_split = splitter(image_test, 50)

# # Print the first image of the data bank
# for i in range(10):
#     plt.matshow(image_test[i,0])
#     plt.colorbar()
#     plt.show()


# use the selected encoder
# please note that the autoencoder in mlg now takes 4 inputs and gives back 4 outputs
autoencoder_ce, encod_ce = mlg.convolutionnal_autoencod(nb_buffs)




# actual learning
nb_epochs = 10
mlg.model_launch(image_train_split, image_test_split, autoencoder_ce, encod_ce, nb_epochs)


# __________________________________________________________________________________test_learning quality

# small audio test on the sample i of the dataset train
path_out = './results'
samples = [i for i in range(20)]

image_test_split = np.array(image_test_split)


def test_audio(path_out):
    for i in samples:
        test_image_learn = np.expand_dims(image_test_split[:,i], 1)
        test_image = np.array(image_test_bis[i][0])
        predicted_image = autoencoder_ce.predict(list(test_image_learn))

        for j in range(4):
            predicted_image[j] = (predicted_image[j] - np.average(predicted_image[j]))*20

        predicted_image = merger(predicted_image)[0][0]

        # avg = np.average(predicted_image)
        # pmin = np.amin(predicted_image)
        # pmax = np.amax(predicted_image)
        # predicted_image = (predicted_image - avg)*50

        # Viewing function
        if i < 10 and i%2 == 0:
            plt.matshow(predicted_image)
            plt.colorbar()
            plt.show()
            plt.matshow(test_image)
            plt.colorbar()
            plt.show()

        print(predicted_image.shape)
        print(test_image.shape)
        lrec = sdo.reconstruct_image(test_image, nb_buffs, path_out, 'test' + str(i))
        rrec = sdo.reconstruct_image(predicted_image, nb_buffs, path_out, 'learned' + str(i))



    print('------------TEST OK')


test_audio(path_out)

