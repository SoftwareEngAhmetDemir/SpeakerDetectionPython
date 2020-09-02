# # -*- coding: utf-8 -*-
# """
# Created on Sat Jul 18 21:28:15 2020
#
# @author: Damer
# """
#
from Correlation import correlation_with_normalize as corf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import python_speech_features
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import pickle
from sklearn import mixture, preprocessing
from sklearn import svm
from python_speech_features import delta
import ReadFiles
import scipy.stats
from ReadFiles import *
# r = ReadFiles.get_Files('sounds/Male')
#
# for i in r:
#     print(i)

while(True):
    Gender = input('Male Or Female ???')
    Nationality = input('newYork Or newEngland')

    files = get_Files('sounds/'+Gender+'/'+Nationality)
    print('sounds file for ' + Gender + ' ' + Nationality)
    print('-------------------------------------------------')
    count = 0
    for file in files:
        print(count, ' - ', file)
        count += 1

    print('--------------------------------------------------')

    choiced_file = input('Choice a sound wave file by its number from the upper list')

    choiced_file = int(choiced_file)
    (rate, sig) = wav.read(files[choiced_file])


#####################################################
    model_Male_NewYork_MFCC_HMM = 'models/MFCC/GMM/MalenewYork'


    model_Female_NewYork_MFCC_HMM = 'models/MFCC/GMM/FemalenewYork'

    model_Male_NewEngland_MFCC_HMM = 'models/MFCC/GMM/MalenewEngland'

    model_Female_NewEngland_MFCC_HMM = 'models/MFCC/GMM/FemalenewEngland'
#######################################################


    array_of_models = [model_Female_NewEngland_MFCC_HMM,
                   model_Female_NewYork_MFCC_HMM,
                   model_Male_NewYork_MFCC_HMM,
                   model_Male_NewEngland_MFCC_HMM]

    name_models = ['FemaleEngland','FemaleNewYork','MaleNewYork','MaleNewEngland']

# print(rate)

    mfcc_feat = mfcc(sig,samplerate= rate,winlen=0.0025, winstep=0.01, numcep=2000,
                 nfilt=26,nfft=1024,lowfreq=0,
                 highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)
    fbank_feat = logfbank(sig, rate).flatten()



    d1= python_speech_features.base.delta(mfcc_feat,9999) # calculate delta and do framing



    combined1 = np.hstack((mfcc_feat,d1))


# combined2 = np.hstack((mfcc_feat2,d2))


# res = len(set(combined1.flatten()) & set(combined2.flatten())) / float(len(set(combined1.flatten()) | set(combined2.flatten()))) * 100

    gmm = mixture.GaussianMixture(n_components=24, max_iter=200, covariance_type='diag', n_init=3)

# Train = gmm.fit(combined1,len(combined1))

#dumping the trained gaussian model
# picklefile = 'd:/deneme/ahmed' + ".gmm"

# pickle.dump(Train, open('MS.gmm', 'wb')) # write model
    f_result = -99999999999999999
    f_model = ''
    count=0
    for model in array_of_models:
        loaded_model = pickle.load(open(model+'.gmm', 'rb')) # ideal model
        result1 = loaded_model.score(combined1) # (self,MFCC)
        if f_result < result1:
            f_result = result1
            f_model = model
            count = array_of_models.index(model)


    ff_model = ''

    print(name_models[count],' ',f_result)



# lable = gmm.predict(combined1)

# result2 = loaded_model.score(combined2)
#
# # p1 = loaded_model.predict(combined1)
#
# Predict = loaded_model.predict(combined1)
# result3 =loaded_model.score_samples(combined1)
# print(result1,'%')
#
#
# print(result2,'%')
#
# combined1.resize(10000,1)
# combined2.resize(10000,1)
#
# # f = scipy.stats.pearsonr(combined1.flatten(), combined2.flatten())
#
#
# # print('en son ',np.correlate(combined1.flatten(),combined2.flatten()) )
#
#
#
# print('corelation' ,corf([1,2,3],[10,30,40])*100,'%')

# cor  = 1*100+2*300+3*400 = 100+600+1200 = 1900

# norm = (1+2+3)^2 = 36 , (10+30+40)^2 = 6400 , sqrt (36 * 6400) = 15,178.9327
# acor = 1900/15179.933

