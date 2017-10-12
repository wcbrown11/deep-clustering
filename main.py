# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:09:46 2016

@author: valterf

Main code with examples for the most important function calls. None of this
will work if you haven't prepared your train/valid/test file lists.
"""
from visualization import print_examples
from nnet import train_nnet, load_model
from predict import separate_sources
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()

config.gpu_options.allow_growth=True

set_session(tf.Session(config=config))
import os
import glob
def generate_file_list(path):
    file_list = glob.glob(path)
    result = []
    for ele in file_list:
        for j in range(1,7):
            name = ele.replace('CH1','CH%d'%j)
            result.append(os.path.basename(name))
    return result

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

def main():
    train_path = '/scratch/near/2speakers_6channel/wav16k/min/tr/mix/*CH1.wav'
    train_list = generate_file_list(train_path)
    valid_list = '/scratch/near/2speakers_6channel/wav16k/min/cv/mix/*CH1.wav'
    valid_list = generate_file_list(valid_list)
    train_nnet(train_list, valid_list)
    model = load_model('model')
    egs = []
    current_spk = ""

    # From here on, all the code does is get 2 random speakers from the test
    # set and visualize the outputs and references. You need to have matplotlib
    # installed for this to work.
    for line in open('test'):
        line = line.strip().split()
        if len(line) != 2:
            continue
        w, s = line
        if s != current_spk:
            egs.append(w)
            current_spk = s
            if len(egs) == 2:
                break
    print_examples(egs, model, db_threshold=40, ignore_background=True)
    
    # If you wish to test source separation, generate a mixed 'mixed.wav'
    # file and test with the following line
    # separate_sources('mixed.wav', model, 2, 'out')


if __name__ == "__main__":
    main()
