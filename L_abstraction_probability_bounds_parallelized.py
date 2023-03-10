import numpy as np
#import matplotlib.pyplot as plt
import itertools
from scipy.special import comb
import scipy.io
from tqdm import tqdm
from multiprocessing import Pool

# Load the measure (probability) of the events, such as the measure of the k_bar-sequences' equivalence classes
ell_seq_measure = scipy.io.loadmat(
    './data/H_sequences_measure.mat')
# Load the label associated to the measure of the k_bar-sequences
u = []
with open('./data/H_sequences.txt') as f:
    for line in f:
        u.append(line.strip())

v = list(zip(u, ell_seq_measure["ell_sequences_measure"][0]))
# Load ell-sequences
u = []
with open('./data/ell_sequences.txt') as f:
    for line in f:
        u.append(line.strip())

# Number of i.c.
N = 30
H = len(v[0][0].split('-'))
l = len(u[0].split('-'))
M = len(v)
#print("Possible outcomes ", M)
V_support = np.array(())
Pmf_V = np.array(())
Pmf_seen = np.zeros(M)
Pmf_exact_symbols = {}


# For parallelization


def compute_violation(chosen, V_support, Pmf_V, Pmf_exact_symbols, Pmf_seen):
    global u
    V_support_temp = np.array(())
    Pmf_V_temp = np.array(())
    Pmf_seen_temp = np.zeros(M)
    Pmf_exact_symbols_temp = {}
    i = len(chosen)
    # tuple of i elements: compute probability of seeing exactly these elements
    P_chosen = sum([v[q][1] for q in chosen])
    chosen_H_sequences = [v[q][0] for q in chosen]
    violation = 0
    # Find l-sequences not present in the N k_bar-sequences
    unseen_ell_sequences = u.copy()
    violating_H_sequences = [v[q] for q in range(M) if q not in chosen]
    # Split k_bar-sequences in l-sequences
    for H_sequence in chosen_H_sequences:
        for c in range(H - l):
            ell_sequence = H_sequence[2 * c: 2 * c + 2 * (l - 1) + 1]
            if ell_sequence in unseen_ell_sequences:
                unseen_ell_sequences.remove(ell_sequence)
    for z in unseen_ell_sequences:
        violating_H_sequences_copy = violating_H_sequences.copy()
        for y in violating_H_sequences:
            if z in y[0]:
                violation += y[1]
                violating_H_sequences_copy.remove(y)
        violating_H_sequences = violating_H_sequences_copy.copy()
    # Probability of seeing all the possible N outcomes with sequences made up of any combination of the chosen,
    # including combinations where some elements of the i-tuple are missing which need to be subtracted
    outcome_probability = np.power(P_chosen, N)
    Pmf_exact_symbols_temp[chosen] = outcome_probability
    # Subtract repeated outcomes to obtain all sequences that contain at least one element of each chosen ones
    for k in range(1, i):
        sub_combinations = list(itertools.combinations(chosen, k))
        for sub_comb in sub_combinations:
            Pmf_exact_symbols_temp[chosen] -= Pmf_exact_symbols[sub_comb]
    if violation not in V_support:
        V_support_temp = np.append(V_support_temp, violation)
        Pmf_V_temp = np.append(Pmf_V_temp, Pmf_exact_symbols_temp[chosen])
    else:
        idx = np.where(V_support_temp == violation)
        Pmf_V_temp[idx] = Pmf_V_temp[idx] + Pmf_exact_symbols_temp[chosen]
    Pmf_seen_temp[i - 1] += outcome_probability
    return V_support_temp, Pmf_V_temp, Pmf_exact_symbols_temp, Pmf_seen_temp


# Main

if __name__ == '__main__':
    for i in range(1, M + 1):
        # This blows up quickly: at every iteration computes M-choose-i tuples
        index_mat_temp = list(itertools.combinations(range(M), i))
        index_mat = [(q, V_support, Pmf_V, Pmf_exact_symbols, Pmf_seen) for q in index_mat_temp]
        with Pool() as pool:
            for result in tqdm(pool.starmap(compute_violation, index_mat)):
                [V_support_temp, Pmf_V_temp, Pmf_exact_symbols_temp, Pmf_seen_temp] = result
                for t in range(len(V_support_temp)):
                    if V_support_temp[t] not in V_support:
                        V_support = np.append(V_support, V_support_temp[t])
                        Pmf_V = np.append(Pmf_V, Pmf_V_temp[t])
                    else:
                        idx = np.where(V_support == V_support_temp[t])
                        Pmf_V[idx] = Pmf_V[idx] + Pmf_V_temp[t]
                Pmf_exact_symbols.update(Pmf_exact_symbols_temp)
                Pmf_seen[i-1] = Pmf_seen_temp[i-1]
        if i > 1:
            for j in range(1, i):
                Pmf_seen[i - 1] -= Pmf_seen[j - 1] * comb(M - j, i - j)
        print("Probability of seeing exactly ", i, " symbols ", Pmf_seen[i - 1])

    print("Pmf_seen ", sum(Pmf_seen))
    print("Pmf_exact_symbols ", sum(Pmf_exact_symbols.values()))
    print("Pmf_V ", sum(Pmf_V))
    # Sort arrays
    p = V_support.argsort()
    V_support = V_support[p]
    Pmf_V = Pmf_V[p]
    # Plot stuff
    print("Support of the Violation")
    print(V_support)
    print("Probability mass function of the Violation")
    print(Pmf_V)
    print("Probability V > 0 is ", sum(Pmf_V[1:]))
    '''fig, (ax1, ax2) = plt.subplots(2)
    ax1.stem(V_support, Pmf_V, label="P.M.F. of the violation")
    Cmf_V = [Pmf_V[0]]
    for i in range(1, len(Pmf_V), 1):
        Cmf_V.append(Cmf_V[i - 1] + Pmf_V[i])
    ax2.plot(V_support, Cmf_V, label="C.M.F. of the violation")
    plt.setp((ax1, ax2), xlim=[0, 1], ylim=[0, 1])
    plt.xlabel("V")
    ax1.legend()
    ax2.legend()
    plt.show()'''
