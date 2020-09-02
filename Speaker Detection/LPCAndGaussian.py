from LPCDenklemleri import *
from ReadFiles import *

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
    model_Male_NewYork_LPC_GMM = 'models/LPC/GMM/MalenewYork'


    model_Female_NewYork_LPC_GMM = 'models/LPC/GMM/FemalenewYork'

    model_Male_NewEngland_LPC_GMM = 'models/LPC/GMM/MalenewEngland'

    model_Female_NewEngland_LPC_GMM = 'models/LPC/GMM/FemalenewEngland'
    #######################################################
    array_of_models = [model_Female_NewEngland_LPC_GMM,
                       model_Female_NewYork_LPC_GMM,
                       model_Male_NewYork_LPC_GMM,
                       model_Male_NewEngland_LPC_GMM]
    ####################################################
    name_models = ['FemaleEngland','FemaleNewYork','MaleNewYork','MaleNewEngland']

    #####################################################
    # print ('array ',lpc(sig,rate,300))
    a = lpc(sig,rate,100)
    #print('first , = ',a)


    gmm = mixture.GaussianMixture(
    n_components=24,  covariance_type='diag', tol=0.0001, reg_covar=1e-06, max_iter=20000,
        n_init=1,
        init_params='kmeans', weights_init=None, means_init=None,
        precisions_init=None, random_state=None, warm_start=True,
        verbose=1, verbose_interval=99999
    )


    f_result = -99999999999999999
    f_model = ''
    count=0
    for model in array_of_models:
        loaded_model = pickle.load(open(model+'.gmm', 'rb')) # ideal model
        result1 = loaded_model.score(a) # (self,MFCC)
        if f_result < result1:
            f_result = result1
            f_model = model
            count = array_of_models.index(model)


    print('konusan ====> '+name_models[count])