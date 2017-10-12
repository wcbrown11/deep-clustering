# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:23:31 2016

@author: jcsilva
STFT/ISTFT derived from Basj's implementation[1], with minor modifications,
such as the replacement of the hann window by its root square, as specified in
the original paper from Hershey et. al. (2015)[2]

[1] http://stackoverflow.com/a/20409020
[2] https://arxiv.org/abs/1508.04306
"""
import numpy as np
import random
import soundfile as sf
from config import FRAME_LENGTH, FRAME_SHIFT, FRAME_RATE
from config import TIMESTEPS, DB_THRESHOLD


def sqrt_hann(M):
    return np.sqrt(np.hanning(M))


def stft(x, fftsize=int(FRAME_LENGTH*FRAME_RATE),
         overlap=FRAME_LENGTH//FRAME_SHIFT):
    """
    Short-time fourier transform.
        x:
        input waveform (1D array of samples)

        fftsize:
        in samples, size of the fft window

        overlap:
        should be a divisor of fftsize, represents the rate of
        window superposition (window displacement=fftsize/overlap)

        return: linear domain spectrum (2D complex array)
    """
    hop = int(np.round(fftsize / overlap))
    w = sqrt_hann(fftsize)
    out = np.array([np.fft.rfft(w*x[i:i+fftsize])
                    for i in range(0, len(x)-fftsize, hop)])
    return out


def istft(X, overlap=FRAME_LENGTH//FRAME_SHIFT):
    """
    Inverse short-time fourier transform.
        X:
        input spectrum (2D complex array)

        overlap:
        should be a divisor of (X.shape[1] - 1) * 2, represents the rate of
        window superposition (window displacement=fftsize/overlap)

        return: floating-point waveform samples (1D array)
    """
    fftsize = (X.shape[1] - 1) * 2
    hop = int(np.round(fftsize / overlap))
    w = sqrt_hann(fftsize)
    x = np.zeros(X.shape[0]*hop)
    wsum = np.zeros(X.shape[0]*hop)
    for n, i in enumerate(range(0, len(x)-fftsize, hop)):
        x[i:i+fftsize] += np.real(np.fft.irfft(X[n])) * w   # overlap-add
        wsum[i:i+fftsize] += w ** 2.
    pos = wsum != 0
    x[pos] /= wsum[pos]
    return x


def get_egs(wavlist, path, min_mix=2, max_mix=3, batch_size=1):
    """
    Generate examples for the neural network from a list of wave files with
    speaker ids. Each line is of type "path speaker", as follows:

    path/to/mix.wav

    and so on.
    min_mix and max_mix are the minimum and maximum number of examples to
    be mixed for generating a training example
    """
    speaker_wavs = {}
    batch_x = []
    batch_y = []
    batch_count = 0

    while True:  # Generate examples indefinitely
        wavsum = None
        sigs = []

        # Pop wav files from random speakers, store them individually for
        # dominant spectra decision and generate the mixed input
        file_name = wavlist.pop(0)
        wavlist.append(file_name)
        s1, rate = sf.read('/scratch/near/2speakers_6channel/wav16k/min/'+path+'/s1/'+file_name)
        if rate != FRAME_RATE:
            raise Exception("Config specifies " + str(FRAME_RATE) +
                            "Hz as sample rate, but file " + str(p) +
                            "is in " + str(rate) + "Hz.")
        s1 = s1 - np.mean(s1)
        s1 = s1/np.max(np.abs(s1))
        s1 *= (np.random.random()*1/4 + 3/4)
        s2, rate = sf.read('/scratch/near/2speakers_6channel/wav16k/min/'+path+'/s2/'+file_name)
        if rate != FRAME_RATE:
            raise Exception("Config specifies " + str(FRAME_RATE) +
                            "Hz as sample rate, but file " + str(p) +
                            "is in " + str(rate) + "Hz.")
        s2 = s2 - np.mean(s2)
        s2 = s1/np.max(np.abs(s2))
        s2 *= (np.random.random()*1/4 + 3/4)

        wavsum = s1[:min(len(s1),len(s2))] + s2[:min(len(s1),len(s2))]
        sigs.append(s1)
        sigs.append(s2)


        # STFT for mixed signal
        def get_logspec(sig):
            return np.log10(np.absolute(stft(sig)) + 1e-7)

        X = get_logspec(wavsum)
        if len(X) <= TIMESTEPS:
            continue

        # STFTs for individual signals
        specs = []
        for sig in sigs:
            specs.append(get_logspec(sig[:len(wavsum)]))
        specs = np.array(specs)

        nc = max_mix

        # Get dominant spectra indexes, create one-hot outputs
        Y = np.zeros(X.shape + (nc,))
        vals = np.argmax(specs, axis=0)
        for i in range(2):
            t = np.zeros(nc)
            t[i] = 1
            Y[vals == i] = t

        # Create mask for zeroing out gradients from silence components
        m = np.max(X) - DB_THRESHOLD/20.  # From dB to log10 power
        z = np.zeros(nc)
        Y[X < m] = z

        # Generating sequences
        i = 0
        while i + TIMESTEPS < len(X):
            batch_x.append(X[i:i+TIMESTEPS])
            batch_y.append(Y[i:i+TIMESTEPS])
            i += TIMESTEPS//2

            batch_count = batch_count+1

            if batch_count == batch_size:
                inp = np.array(batch_x).reshape((batch_size,
                                                 TIMESTEPS, -1))
                out = np.array(batch_y).reshape((batch_size,
                                                 TIMESTEPS, -1))
                yield({'input': inp},
                      {'kmeans_o': out})
                batch_x = []
                batch_y = []
                batch_count = 0


if __name__ == "__main__":
    x, y = next(get_egs(train_list, batch_size=50))
    print(x['input'].shape)
    print(y['kmeans_o'].shape)
