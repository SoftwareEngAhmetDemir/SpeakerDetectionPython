import numpy as np
from hmmlearn import hmm
import python_speech_features
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import pickle
from ReadFiles  import *


print('MFCC HMM Model ile!!! ')


while(True):
    Gender = input('Male Or Female ???')
    Nationality = input('newYork Or newEngland')


    files = get_Files('sounds/'+Gender+'/'+Nationality)


    print('sounds file for '+ Gender+' '+Nationality)
    print('-------------------------------------------------')
    count = 0
    for file in files:
        print(count ,' - ' ,file)
        count+=1

    print('--------------------------------------------------')

    choiced_file = input('Choice a sound wave file by its number from the upper list')

    choiced_file = int(choiced_file)
    (rate, sig) = wav.read(files[choiced_file])
#####################################################
    model_Male_NewYork_MFCC_HMM = 'models/MFCC/HMM/MalenewYork'


    model_Female_NewYork_MFCC_HMM = 'models/MFCC/HMM/FemalenewYork'

    model_Male_NewEngland_MFCC_HMM = 'models/MFCC/HMM/MalenewEngland'

    model_Female_NewEngland_MFCC_HMM = 'models/MFCC/HMM/FemalenewEngland'
#######################################################

    array_of_models = [model_Female_NewEngland_MFCC_HMM,
                       model_Female_NewYork_MFCC_HMM,
                       model_Male_NewYork_MFCC_HMM,
                       model_Male_NewEngland_MFCC_HMM]





    mfcc_feat = mfcc(sig,samplerate= rate,winlen=0.0025, winstep=0.01, numcep=2000,
                 nfilt=26,nfft=1024,lowfreq=0,
                 highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)

    d1= python_speech_features.base.delta(mfcc_feat,9999) # calculate delta and do framing

    combined1 = np.hstack((mfcc_feat,d1))

    model = hmm.GaussianHMM(n_components=16, covariance_type='full',
                 min_covar=1e-3,
                 startprob_prior=1.0, transmat_prior=0.1,
                 means_prior=0.1, means_weight=0.1,
                 covars_prior=1e-2, covars_weight=0.1,
                 algorithm="viterbi", random_state=None,
                 n_iter=100, tol=1e-2, verbose=True,
                 params="stmc", init_params="stmc")

    f_result = -99999999999999999
    f_model = ''
    count = 0
    for model in array_of_models:
        loaded_model = pickle.load(open(model+".hmm", 'rb')) # ideal model
        result1 = loaded_model.score(combined1) # (self,MFCC)
        if f_result<result1:
            f_result = result1
            f_model=model
            count = array_of_models.index(model)
        # print(model,' ',result1)
    name_models = ['FemaleEngland', 'FemaleNewYork', 'MaleNewYork', 'MaleNewEngland']


    print('En Yakin Kounsan : ===> '+name_models[count])
