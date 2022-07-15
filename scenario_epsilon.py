# originally from Risk and complexity in scenario optimization,
# https://doi.org/10.1007/s10107-019-01446-4
# matlab code, translated for python and improved
import timeit

import numpy as np
from scipy.special import betaincinv as bii
from scipy.special import betainc as bi


def epsLU(k, N, beta):

    # alphaL = betaaincinv(beta,k,N-k+1)
    # scipy and matlab have scrambled inputs, careful!
    # further, BII doesnt like a == 0, so add epsilon
    epsilon_precision = np.finfo(float).eps
    alphaL = bii(k+epsilon_precision, N-k+1, beta)
    # alphaU = 1-betaaincinv(beta,N-k+1,k)
    alphaU = 1-bii(N-k+1, k+epsilon_precision, beta)

    m1 = np.arange(start=k, stop=N+1, step=1)

    """
    the matlab code generates a square matrix, takes the upper (or lower, later on) triangular, 
    and sums over the values. whilst this works, it uses a lot of memory
    """
    # start_old_method = timeit.default_timer()
    # aux1 = np.sum(np.triu(np.log(np.ones((N-k+1, 1)) * m1), k=1), axis=1)
    # aux2 = np.sum(np.triu(np.log(np.ones((N-k+1, 1)) * (m1-k)), k=1), axis=1)
    # coeffs1 = aux2-aux1
    # end_old_method = timeit.default_timer()
    # time_old = end_old_method-start_old_method

    # instead of generating a matrix, use np.cumsum
    aux1 = np.log(m1)
    aux1[0] = 0.
    aux1 = np.flip(np.cumsum(np.flip(aux1)))
    aux1[0:-1] = aux1[1:]
    aux1[-1] = 0.
    # aux 2
    aux2 = np.log(m1-k)
    aux2[0] = 0.
    aux2 = np.flip(np.cumsum(np.flip(aux2)))
    aux2[0:-1] = aux2[1:]
    aux2[-1] = 0.
    coeffs1 = aux2-aux1

    # start_time_old_method = timeit.default_timer()
    # m2 = np.arange(start=N+1, stop=4*N+1, step=1)
    # aux3 = np.sum(np.tril(np.log(np.ones((3*N, 1)) * m2)), axis=1)
    # aux4 = np.sum(np.tril(np.log(np.ones((3*N, 1)) * (m2-k))), axis=1)
    # coeffs2 = aux3-aux4
    # end_time_old_method = timeit.default_timer()
    #
    # time_old = end_time_old_method-start_time_old_method

    m2 = np.arange(start=N + 1, stop=4 * N + 1, step=1)
    aux3 = np.cumsum((np.log(m2)))
    aux4 = np.cumsum((np.log(m2-k)))
    coeffs2 = aux3 - aux4

    # when to stop the bisection method
    bisection_precision = 1e-8

    t1 = 1. - alphaL
    t2 = 1.

    poly1 = 1 + beta / (2 * N) - beta / (2 * N) * np.sum(np.exp(coeffs1 - (N - m1.T) * np.log(t1))) \
            -beta / (6 * N) * np.sum(np.exp(coeffs2 + (m2.T-N) * np.log(t1)))

    poly2 = 1 + beta / (2 * N) - beta / (2 * N) * np.sum(np.exp(coeffs1 - (N - m1.T) * np.log(t2))) \
            -beta / (6 * N) * np.sum(np.exp(coeffs2 + (m2.T-N) * np.log(t2)))

    if (poly1 * poly2) > 0:
        epsL = 0.
    else:
        epsL = 1. - (t2+t1)/2
        while t2 - t1 > bisection_precision:
            t = (t1 + t2) / 2
            # print(f'Bisection precision 1 : {t2 - t1}')
            polyt = 1 + beta / (2 * N) - beta / (2 * N) * np.sum(np.exp(coeffs1 - (N - m1.T) * np.log(t))) \
                    -beta / (6 * N) * np.sum(np.exp(coeffs2 + (m2.T-N) * np.log(t)))
            if polyt > 0:
                t1 = t
            else:
                t2 = t

            epsL = 1. - t2

    t1 = 0.
    t2 = 1. - alphaU

    poly1 = 1 + beta / (2 * N) - beta / (2 * N) * np.sum(np.exp(coeffs1 - (N - m1.T) * np.log(t1))) \
            -beta / (6 * N) * np.sum(np.exp(coeffs2 + (m2.T-N) * np.log(t1)))
    poly2 = 1 + beta / (2 * N) - beta / (2 * N) * np.sum(np.exp(coeffs1 - (N - m1.T) * np.log(t2))) \
            -beta / (6 * N) * np.sum(np.exp(coeffs2 + (m2.T-N) * np.log(t2)))

    if (poly1 * poly2) > 0:
        epsU = 0.
    else:
        while t2 - t1 > bisection_precision:
            # print(f'Bisection precision 2 : {t2 - t1}')
            t = (t1 + t2) / 2
            polyt = 1 + beta / (2 * N) - beta / (2 * N) * np.sum(np.exp(coeffs1 - (N - m1.T) * np.log(t))) \
                    -beta / (6 * N) * np.sum(np.exp(coeffs2 + (m2.T-N) * np.log(t)))
            if polyt > 0:
                t2 = t
            else:
                t1 = t

        epsU = 1. - t1

    return epsL, epsU


if __name__ == '__main__':
    k = 7
    N = 900
    beta = 1e-3
    print(epsLU(k=k, N=N, beta=beta))
