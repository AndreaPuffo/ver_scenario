# originally from The Risk of Making Decisions from data through the lens of the scenario approach,
# https://marco-campi.unibs.it/pdf-pszip/2021-SYSID.pdf
# inspired by the matlab code found at the end of
# Wait and Judge Scenario Optimisation,
# https://marco-campi.unibs.it/pdf-pszip/Wait-and-judge.PDF, or https://doi.org/10.1007/s10107-016-1056-9
import timeit
import numpy as np
import scipy.special
from decimal import Decimal as dc


###########################
# important note: this function is numerically unstable for values N > 1000
# use the Decimal one in that case, slower but accurate
###########################
def eps_general_smallN(k, N, beta):

    """
    adapted from equation (3)
    (n k) * t**(N-k) - beta/N * sum_{i=k}^{N-1} (i k) * t**(i-k) = 0 ,
    which is

    1 - beta/N * (N k)**(-1) * sum_{i=k}^{N-1} (i k) * t**(i-N) = 0

    """
    n_over_k_minus_one = dc(scipy.special.comb(N, k, exact=True)) ** -1
    n_over_k_minus_one = float(n_over_k_minus_one)

    i_over_k = np.zeros((N-k, ))
    for idx in range(k, N):
        i_over_k[idx-k] = float(dc.ln( dc(scipy.special.comb(idx, k, exact=True)) ))
    i_minus_n = np.arange(start=k, stop=N) - N

    # when to stop the bisection method
    bisection_precision = 1e-12

    t1 = 0.
    t2 = 1.

    while t2 - t1 > bisection_precision:
        t = (t1 + t2) / 2
        # print(f'Bisection precision 1 : {t2 - t1}')
        polyt = 1. - n_over_k_minus_one * (beta / N) * np.sum(np.exp(i_over_k + i_minus_n * np.log(t)))
        if polyt > 0:
            t2 = t
        else:
            t1 = t

        eps = 1. - t1

    return eps


def eps_general(k, N, beta):

    """
    adapted from equation (3)
    (n k) * t**(N-k) - beta/N * sum_{i=k}^{N-1} (i k) * t**(i-k) = 0 ,
    which is

    1 - beta/N * (N k)**(-1) * sum_{i=k}^{N-1} (i k) * t**(i-N) = 0

    """
    print(k)
    n_over_k_minus_one = dc(scipy.special.comb(N, k, exact=True)) ** -1

    i_over_k = np.empty((N-k, ), dtype=np.object)
    for idx in range(k, N):
        i_over_k[idx-k] = dc.ln(dc(scipy.special.comb(idx, k, exact=True)))
    i_minus_n = np.arange(start=k, stop=N) - N

    # when to stop the bisection method
    bisection_precision = 1e-12

    t1 = dc(0.)
    t2 = dc(1.)

    while t2 - t1 > bisection_precision:
        t = (t1 + t2) / 2
        # print(f'Bisection precision 1 : {t2 - t1}')
        tmp = i_over_k + i_minus_n * dc.ln(t)
        sumexptmp = sum([dc.exp(i) for i in tmp])
        polyt = dc(1.) - n_over_k_minus_one * dc(beta / N) * sumexptmp
        # polyt = 1. - n_over_k_minus_one * dc((beta / N) * np.sum(dc.exp(i_over_k + i_minus_n * dc.ln(t))))
        if polyt > 0:
            t2 = t
        else:
            t1 = t

        eps = dc(1.) - t1

    return float(eps)


if __name__ == '__main__':
    k = 7
    N = 900
    beta = 1e-3
    print(eps_general(k=k, N=N, beta=beta))
