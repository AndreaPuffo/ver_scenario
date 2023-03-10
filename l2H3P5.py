import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.special import comb

# Number of i.c.
N = 12
# gamma < 1/32
gamma = 1/64
#
ABC = 1/2
BCD = 1/4
CDE = 1/8
DEE = 1/16
EEE = 1/16-2*gamma
EEA = gamma
EAB = gamma
### Assume uniform distribution over [0,1] with dynamics x+ = 1/2x for x in [gamma,1] and x+ = 1/2(x+1) elsewhere
v = (("ABC",ABC),("BCD",BCD), ("CDE",CDE), ("DEE",DEE),("EEE",EEE),("EEA",EEA),("EAB",EAB))
u = ("AB","BC","CD","DE","EE","EA")
M = len(v)
V_support = np.array(())
Pmf_V = np.array(())
Pmf_seen = np.zeros(M)
Pmf_exact_symbols = {}
for i in range(1,M+1):
    index_mat = list(itertools.combinations(range(M), i))
    for chosen in index_mat:
        P_chosen = sum([v[u][1] for u in chosen])
        unchosen_sequences = [v[u][0] for u in chosen]
        violation = 0
        violating_short_strings = []
        violating_long_strings = []
        ### Find l-sequences not present in the N k_bar-sequences
        for z in u:
            ### Flag = 0 if the l-sequences is missing
            flag = 0
            for x in unchosen_sequences:
                if z in x:
                    flag = 1
                    break
            if flag == 0:
                ### Find k_bar-sequences that contain the l-sequence
                for y in v:
                    if z in y[0]:
                        ### To prevent counting duplicate k_bar-sequences
                        if y not in violating_long_strings:
                            violating_long_strings.append(y)
                            violation += y[1]
        ### Probability of seeing all the possible N long outcomes with sequences made up of any combination of the chosen k_bar-sequences
        outcome_probability = np.power((P_chosen), N)
        Pmf_exact_symbols[chosen] = outcome_probability
        ### Subtract repeated outcomes to obtain all sequences that contain at least one element of each chosen ones
        for k in range(1,i):
            sub_combinations = list(itertools.combinations(chosen, k))
            for sub_comb in sub_combinations:
                Pmf_exact_symbols[chosen] -= Pmf_exact_symbols[sub_comb]

        if violation not in V_support:
            V_support = np.append(V_support, violation)
            Pmf_V = np.append(Pmf_V, Pmf_exact_symbols[chosen])
        else:
            idx = np.where(V_support == violation)
            Pmf_V[idx] = Pmf_V[idx] + Pmf_exact_symbols[chosen]

        Pmf_seen[i-1] += outcome_probability
    if i > 1:
        for j in range(1, i):
            Pmf_seen[i-1] -= Pmf_seen[j-1]*comb(M-j, i-j)
    print("Probability of seeing exactly ", i, " symbols ", Pmf_seen[i - 1])

print("Pmf_seen sums to ", sum(Pmf_seen))
print("Pmf_exact_symbols sums to ", sum(Pmf_exact_symbols.values()))
print("Pmf_V sums to ", sum(Pmf_V))
### Sort arrays
p = V_support.argsort()
V_support = V_support[p]
Pmf_V = Pmf_V[p]
### Plot stuff
print("Support of the Violation")
print(V_support)
print("Probability mass function of the Violation")
print(Pmf_V)
fig, (ax1,ax2) = plt.subplots(2)
ax1.stem(V_support, Pmf_V, label="P.M.F. of the violation")
Cmf_V = [Pmf_V[0]]
for i in range(1,len(Pmf_V),1):
    Cmf_V.append(Cmf_V[i-1]+Pmf_V[i])
ax2.plot(V_support, Cmf_V, label="C.M.F. of the violation")
plt.setp((ax1,ax2), xlim = [0,1], ylim = [0,1])
plt.xlabel("V")
ax1.legend()
ax2.legend()
plt.show()
