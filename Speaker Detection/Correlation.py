import math

def correlation_with_normalize(sig1, sig2):
    sum = 0

    sum_sqr_sig1 = 0

    sum_sqr_sig2 = 0

    ###################################

    for x in range(len(sig1)):
        sum += sig1[x] * sig2[x]
        sum_sqr_sig1 += sig1[x] ** 2
        sum_sqr_sig2 += sig2[x] ** 2




    payda = math.sqrt((sum_sqr_sig1 * sum_sqr_sig2))
    print('payda = ', payda)
    pay = sum


    print('pay=' ,+pay)

    return (pay / payda)