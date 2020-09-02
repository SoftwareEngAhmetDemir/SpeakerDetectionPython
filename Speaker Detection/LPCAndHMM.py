from hmmlearn import hmm

from LPCDenklemleri  import *
from ReadFiles import *


print('LPC & HMM ')
print('-----------------------')
while(True):

    Gender = input('Male Or Female ???')
    Nationality = input('newYork Or newEngland')


    files = get_Files('sounds/'+Gender+'/'+Nationality)
    ##################

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

    ##############

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
    model_Male_NewYork_LPC_HMM = 'models/LPC/HMM/MalenewYork'


    model_Female_NewYork_LPC_HMM = 'models/LPC/HMM/FemalenewYork'

    model_Male_NewEngland_LPC_HMM = 'models/LPC/HMM/MalenewEngland'

    model_Female_NewEngland_LPC_HMM = 'models/LPC/HMM/FemalenewEngland'
    #######################################################
    array_of_models = [model_Female_NewEngland_LPC_HMM,
                       model_Female_NewYork_LPC_HMM,
                       model_Male_NewYork_LPC_HMM,
                       model_Male_NewEngland_LPC_HMM]
    ####################################################
    name_models = ['FemaleEngland','FemaleNewYork','MaleNewYork','MaleNewEngland']


######################3


    # print ('array ',lpc(sig,rate,300))
    a = lpc(sig, rate, 100)
    # print('first , = ',a)


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
        loaded_model = pickle.load(open(model + '.hmm', 'rb'))  # ideal model
        result1 = loaded_model.score(a)  # (self,MFCC)
        if f_result < result1:
            f_result = result1
            f_model = model
            count = array_of_models.index(model)

    print('konusan ====> ' + name_models[count])