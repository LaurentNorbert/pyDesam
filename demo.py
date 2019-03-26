import numpy as np
import matplotlib.pyplot as plt
import soundfile as snd  # pip install !
from scipy.io import loadmat

import os.path

import HRaudio as hr
import STFT

"""
Demo script : adaptive High Resolution spectral analysis applied to an audio signal
"""

audioPath = 'audio'
filename = 'glockenspielShort.wav'
# filename = 'clar.wav'
# filename = 'notepiano2.wav'

###################
# read audio file #
###################

fullFilename = os.path.join(audioPath, filename)
(s, Fs) = snd.read(fullFilename)

print('Audio filename : ', fullFilename)
lenFic = s.shape[0]

print('Duration of the file : %.1f seconds' % float(lenFic/Fs))
print('Sampling frequency : ', Fs)

##########################
# HR analysis parameters #
##########################
# filter bank
D = 8     # number of positive frequency subbands
L = 129   # length of the analysis filters - MUST BE ODD!
#                                            ------------
# data
n = 64     # dimension of the data vectors
l = 65     # number of data vectors for each analysis window
N = n+l-1   # total length of the observation window

# ## number of poles in each subband ; automatic determination if 0
rank = [20, 20, 20, 1, 1, 1, 1, 1]
# rank = np.zeros((D,))  # [0, 0, 0, 0, 0, 0]
rmax = 25  # maximum rank in the ESM model (i.e. number of poles) if rank[k] == 0 (with the ESTER criterion)

print('Parameters:')
print('D ', D, ' L ', L, ' N ', N, ' n ', n, ' l ', l)
print('rank = ', rank)
print('rmax = ', rmax)

Df = Fs/2/D  # width of the subbands in Hz
print('Width of the subbands : %d Hz' % int(np.round(Df)))

###############
# HR analysis #
###############
The_z, The_alpha = hr.analyse(s, Fs, D, L, rank, n, l, rmax)

#####################
# plot sepctrogram  #
#####################
nfft = 4096
hopSize = 512
winSize = 3000  # 68 ms at 44100 Hz

plt.subplot(2, 1, 1)
plt.title('spectrogram')
Xs, f, t = STFT.spectrogram(s, winSize, hopSize, nfft, Fs, plot=True)
# plt.show()

##################
# plot HRogramme #
##################
d = loadmat('colors.mat')  # load Roland Badeau's colormap

plt.subplot(2, 1, 2)
hr.HRogram(int(N*D/8), Fs, The_z, The_alpha, n, d['colors'])
plt.title('HRogram')
plt.tight_layout()


#######################################
# re-synthesize the sinusoidal signal #
#######################################
print("Synthesis")
x = 2*np.real(hr.synthese(The_z, The_alpha, D, n, l, 1))
print("Done")

# creation of the output folder
outPath = 'output'
if not os.path.isdir(outPath):
    os.mkdir(outPath)

###################
# save the audios #
###################
outFilename = "%s_sin.wav" % filename[:-4]
snd.write(os.path.join(outPath, outFilename), x, Fs, format='wav')
print('')
print('Output filenames :')
print(os.path.join(outPath, outFilename))

# residual
ll = min((len(s), len(x)))
r = s[:ll]-x[:ll]
outFilename = "%s_noise.wav" % filename[:-4]
snd.write(os.path.join(outPath, outFilename), r, Fs, format='wav')
print(os.path.join(outPath, outFilename))
# show figure
plt.show()
