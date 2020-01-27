# __________________________________________________________________________________Imports

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import soundfile as sf
import sys


'''
use the solid banks functions to create and load banks of images
'''

# __________________________________________________________________________________Read files

def test_string(seq, words):
    '''
    checks if the string seq contains at least one of the words in words
    :param words: array of strings
    :return: boolean
    '''
    if 'Battery 4 Factory Library' in seq:
        return False
    for word in words:
        if word in seq:
            return True
    return False


def loading_banks(root_path, words, num_max):
    num_current = 0
    loaded_bank = []
    for root, dirs, files in os.walk(root_path):
        for name in dirs:
            if num_current >= num_max:
                print("______________EXIT : max number reached_________")
                return loaded_bank
            current = os.path.join(root, name)
            # only keeping the end of the string to catch only the kick banks
            if test_string(current, words):
                loaded_bank.append(current)
                print(loaded_bank[-1])
                num_current += 1
                print(num_current)
    print("____________EXIT : everything read__________")
    return loaded_bank


# we have to search for wav document and link their path:
# the path will be used to convert directly the proper sound into image to save memory
# we discard all sounds that are not in .wav format


def load_files(path_bank, words, num_max):
    loaded_bank = loading_banks(path_bank, words, num_max)
    loaded_files = []
    for bank_path in loaded_bank:
        for root, dirs, files in os.walk(bank_path):
            for name in files:
                current = os.path.join(bank_path, name)
                if ".wav" in current[-4:]:
                    loaded_files.append(current)
    return loaded_files


# __________________________________________________________________________________Refactor the audio files

def reader(path):
    '''
    :param path: path to read
    :return: the object read : [array [[l,r] ....], freq_spl]
    '''
    return sf.read(path, always_2d=True)


def read_wav(readedfile, frame_length):
    '''
    :param readedfile: file read with reader function
    :param frame_length: number of samples kept
    :return: [left, right] channels of the sound
    '''
    left = []
    right = []
    file_length = len(readedfile[0])
    size = min(frame_length, file_length)
    for i in range(size):
        left.append(readedfile[0][i][0])
        right.append(readedfile[0][i][1])
    # the case frame_length <= file_length is already handled
    if frame_length > file_length:
        left.extend([0] * (frame_length - file_length))
        right.extend([0] * (frame_length - file_length))
    return [left, right]


# __________________________________________________________________________________Turn the audio into frequencies

def generate_3d_image(sound, buffer_size, nb_buffs):
    '''
    computes the FFT and extract the 4 interesting parts
    :param sound: [left, right] where L and R are sounds in mono
    :param buffer_size: size of the buffer to compute FFT
    :param nb_buffs: number of buffers
    :return: array of 4 2d images : (l_r, l_c, r_r, r_c)
    '''
    (left, right) = (sound[0][:buffer_size * nb_buffs], sound[1][:buffer_size * nb_buffs])
    spec_l_r = []
    spec_r_r = []
    spec_l_c = []
    spec_r_c = []
    for buffer_i in range(nb_buffs):
        buffer_current_l = left[buffer_i * buffer_size:(buffer_i + 1) * buffer_size]
        buffer_fft_l = np.fft.rfft(buffer_current_l)
        buffer_current_r = right[buffer_i * buffer_size:(buffer_i + 1) * buffer_size]
        buffer_fft_r = np.fft.rfft(buffer_current_r)
        height = len(buffer_fft_l)
        buf_l_r = []
        buf_l_c = []
        buf_r_r = []
        buf_r_c = []
        for i in range(height):
            buf_l_r.append(buffer_fft_l[i].real)
            buf_l_c.append(buffer_fft_l[i].imag)
            buf_r_r.append(buffer_fft_r[i].real)
            buf_r_c.append(buffer_fft_r[i].imag)
        spec_l_r.append(buf_l_r)
        spec_l_c.append(buf_l_c)
        spec_r_r.append(buf_r_r)
        spec_r_c.append(buf_r_c)

    return [spec_l_r, spec_l_c, spec_r_r, spec_r_c]


# ______________________________________________________ PLOT the maps

def print_maps(specs):
    '''
    plots the imag, real for l and r for the sound passed in
    :param specs: out of the generate_3d_image
    :return: void
    '''
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(specs[0], interpolation='nearest', cmap=cm.gist_rainbow)
    axs[0, 0].set_title('l_r')
    axs[0, 1].imshow(specs[2], interpolation='nearest', cmap=cm.gist_rainbow)
    axs[0, 1].set_title('r_r')
    axs[1, 0].imshow(specs[1], interpolation='nearest', cmap=cm.gist_rainbow)
    axs[1, 0].set_title('l_c')
    axs[1, 1].imshow(specs[3], interpolation='nearest', cmap=cm.gist_rainbow)
    axs[1, 1].set_title('r_c')
    plt.show()


# _______________________________________________________ file to image

def file_to_image(path, buffer_size, nb_buffs):
    '''
    :param path: str path of the file to change into image
    :param buffer_size:
    :param nb_buffs:
    :return: the 3D image of the stereo sound
    '''
    total_sample_length = buffer_size * nb_buffs
    try:
        audio = reader(path)
        extracted_audio = read_wav(audio, total_sample_length)
        return generate_3d_image(extracted_audio, buffer_size, nb_buffs)
    except:
        return "error"


def normalize(image):
    '''
    :param image: 3D list
    :return: the array noralized
    '''
    c_max = 0
    for i in range(len(image)):
        for j in range(len(image[i])):
            for k in range(len(image[i][j])):
                current = abs(image[i][j][k])
                if current > c_max:
                    c_max = current
    for i in range(len(image)):
        for j in range(len(image[i])):
            for k in range(len(image[i][j])):
                image[i][j][k] /= c_max
    return image


def file_to_image_norm(path, buffer_size, nb_buffs):
    '''
    same as file_to_image but normalizes the image
    :param path:
    :param buffer_size:
    :param nb_buffs:
    :return:
    '''
    total_sample_length = buffer_size * nb_buffs
    try:
        audio = reader(path)
        extracted_audio = read_wav(audio, total_sample_length)
        not_norm = generate_3d_image(extracted_audio, buffer_size, nb_buffs)
        return normalize(not_norm)
    except:
        return "error"


def load_image_bank(path_bank, words, num_max, buffer_size, nb_buffs):
    '''
    :param path_bank: directory to search into
    :param words: words to search in the databases
    :param num_max: number of selected banks
    :param buffer_size:
    :param nb_buffs:
    :return: an image bank of sounds
    '''
    image_bank = []
    file_bank = load_files(path_bank, words, num_max)
    for path in file_bank:
        image_current = file_to_image_norm(path, buffer_size, nb_buffs)
        print(path, '    ---------- okay')
        if type(image_current) == str or len(image_current) != 4:
            pass
        else:
            # The algorithm will have to learn representations with their complex
            # parts (or the result produced won't be coherent even if the learning is perfect)
            image_bank.append([image_current[0] + image_current[1] + image_current[2] + image_current[3]])
            image_bank.append([image_current[2] + image_current[3] + image_current[0] + image_current[1]])
    return image_bank


# _______________________________________________________ solid banks

def create_solid_bank(bank, buffer_size, nb_buffs, path):
    '''
    saves the bank into the path via np.save method
    :param bank: array to be saved
    :param buffer_size: int
    :param nb_buffs: int
    :param path: path for out file
    :return: void
    '''
    path_save = path + 'size=' + str(buffer_size) + 'nb=' + str(nb_buffs)
    np.save(path_save, bank)
    print('BANK CREATED AT : ' + path_save)


def load_solid_bank(file):
    return np.load(file)

'''
# _______________________________________________________ test load_bank

# buffer_size//2 + 1 = multiple of 16 (maxpooling(4,4)) --> 256 height
buffer_size = 511
nb_buffs = 100
# au total 44100 samples donc 1 seconde
path_bank = "D:/01 Musique/01 Programmes/02 NI 2/"
words = ["Snare", "Snares", "snare", "snares"]
num_max = 1


image_bank = load_image_bank(path_bank, words, num_max, buffer_size, nb_buffs)
print(np.array(image_bank).shape)

# keep 80% for training, 20% for test
from math import floor

nb_images = len(image_bank)
print(sys.getsizeof(image_bank))
selector = floor(nb_images * 0.8)
image_train = np.array(image_bank[:selector])
image_test = np.array(image_bank[selector:])
'''

# _______________________________________________________ test load solid bank
'''
filename = 'bank_sn_test_1'
path_out = './banks/' + filename
create_solid_bank(np.array(load_image_bank(path_bank, words, num_max, buffer_size, nb_buffs)), buffer_size, nb_buffs, path_out)
path_in = 'banks/bank_sn_ta=1size=511nb=100.npy'
loaded_bank = load_solid_bank(path_in)
print(loaded_bank[0])
'''