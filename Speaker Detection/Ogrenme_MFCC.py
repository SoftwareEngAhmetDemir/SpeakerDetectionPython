import numpy as np
from hmmlearn import hmm
import python_speech_features
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import pickle
from ReadFiles  import *
from sklearn import mixture



def Ogrenme_MFCC_HMM(model_name, sound_file,model_Type): ##Make A model for HMM using MFCC and GMM

    (rate, sig) = wav.read(sound_file)

    mfcc_feat = mfcc(sig, samplerate=rate, winlen=0.0025, winstep=0.01, numcep=2000,
                         nfilt=26, nfft=1024, lowfreq=0,
                         highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True)

    d1 = python_speech_features.base.delta(mfcc_feat, 9999)  # calculate delta and do framing

    combined1 = np.hstack((mfcc_feat, d1))
    if model_Type=='hmm':
        model = hmm.GaussianHMM(n_components=16, covariance_type='full',
                                min_covar=1e-3,
                                startprob_prior=1.0, transmat_prior=0.1,
                                means_prior=0.1, means_weight=0.1,
                                covars_prior=1e-2, covars_weight=0.1,
                                algorithm="viterbi", random_state=None,
                                n_iter=100, tol=1e-2, verbose=True,
                                params="stmc", init_params="stmc")

    else:
        if model_Type=='gmm':
            model = mixture.GaussianMixture(n_components=24, max_iter=200, covariance_type='diag', n_init=3)

        if model_Type!='gmm' and model_Type!='hmm':
            print('error!!!!!')


    Train = model.fit(combined1)

    pickle.dump(Train, open(model_name + "", 'wb'))  # write model

Model_Type = input('hmm OR gmm')


Gender = input('Gender? Male Or Female')


Type = input('Type? newYork Or newEngland')

file_place = 'sounds/'+Gender+'/'+Type

Sound_Files = get_Files('sounds/'+Gender+'/'+Type)

for fileName in Sound_Files:
    print(fileName)

choiced_file = input('which file you want to choice it for training from upper files')
#
file_place+='/'+choiced_file
# print('file you choiced '+file_place)

Ogrenme_MFCC_HMM('models/MFCC/'+Model_Type+'/'+Gender+Type+'.'+Model_Type,file_place,Model_Type)