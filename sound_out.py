# __________________________________________________________________________________Imports

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

import bank_creator as bkc

'''
Use reconstruct_image in run to generate sounds with images
'''


# _____________________________________________________ reformat given images

def reconstruct_image(image, nb_buffs, path, filename):
    [l_r, l_c, r_r, r_c] = [image[:nb_buffs], image[nb_buffs:2*nb_buffs], image[2*nb_buffs:3*nb_buffs], image[3*nb_buffs:]]
    (left_p, right_p) = (generate_sound_back(l_r, l_c), generate_sound_back(r_r, r_c))
    reconstruct_test(left_p, right_p, path, filename)
    print('OK -- reconstructed')
    return True


# _____________________________________________________ iFFT : back to sound

def generate_sound_back(real, imag):
    '''
    :param real: list : real part of the FFT of the mono sound
    :param imag: list : imaginary part of the FFT of the mono sound
    :return: array
    '''
    sound = []
    #print(len(real), '\n' * 10)
    for i in range(len(real)):
        #print(i, len(real[i]), len(imag[i]))
        complex_buffer = [real[i][j] + 1j * imag[i][j] for j in range(len(real[i]))]
        sound_buffer = np.fft.irfft(complex_buffer)
        sound.extend(sound_buffer)
    return sound


# _____________________________________________________ compare audios real/FFTed with the example

def plot_sounds(recomp, original):
    '''
    plots original vs recomposed sounds
    :param recomp: recomposed sound (mono)
    :param original: original sound (mono)
    :return: void
    '''
    original_n = original[:len(recomp)]
    xlin = np.linspace(0, len(recomp) - 1, len(recomp))
    fig, axs = plt.subplots(2)
    axs[0].plot(xlin, original_n)
    axs[1].plot(xlin, recomp)
    plt.show()


# _______________________________________________________ recreate an audio file

def reconstruct_test(left, right, path, filename):
    '''
    reconstruct the audio file
    :param left: list, audio canal left
    :param right: list, audio canal right
    :param path: the output path
    :param filename: the output filename
    :return: void
    '''
    path_full = path + '/' + filename + '.wav'
    sizer = 2 ** 31
    sound_recomposed_stereo = [[int(left[i] * sizer), int(right[i] * sizer)] for i in range(len(left))]
    sf.write(path_full, sound_recomposed_stereo, 44100)

# _______________________________________________________ normalizer_test

# path_test = 'D:/01 Musique/01 Programmes/02 NI 2/Amplified Funk\Samples\Drums\Snare\Snare Equinox 2.wav'
# image_current = bkc.file_to_image_norm(path_test, 511, 100)
# left = generate_sound_back(image_current[0], image_current[1])
# right = generate_sound_back(image_current[2], image_current[3])
# reconstruct_test(left, right, 'results', 'test_reconstructed_norm')