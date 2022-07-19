import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.special import comb

# Number of i.c.
N = 10e5
# gamma < 1/32
gamma = 1/64
#
AB = 1/2
BC = 1/4
CD = 1/8
DE = 1/16
EE = 1/16-gamma
EA = gamma
# Assume uniform distribution over [0,1] with dynamics x+ = 1/2x for x in [gamma,1] and x+ = 1/2(x+1) elsewhere
# A = [1/2,1], B = [0,1/2)
v = (("AB",AB),("BC",BC), ("CD",CD), ("DE",DE),("EE",EE),("EA",EA))
u = ("AB","BC","CD","DE","EE","EA")
M = len(v)
# Generate all possible outcomes of length N
V_support = np.array(())
Pmf_V = np.array(())
Pmf_seen = np.zeros(M)
Pmf_exact_symbols = {}
for i in range(1,M+1):
    index_mat = list(itertools.combinations(range(M), i))
    for chosen in index_mat:
        #print(chosen)
        violation = sum([v[u][1] for u in range(M) if u not in chosen])
        #print(violation)
        ### Probability of seeing all the possible N outcomes with sequences made up of any combination of the chosen
        outcome_probability = np.power((1 - violation), N)
        Pmf_exact_symbols[chosen] = outcome_probability
        ### Subtract repeated outcomes to obtain all sequences that contain at least one element of each chosen ones
        for k in range(1,i):
            sub_combinations = list(itertools.combinations(chosen, k))
            #print(sub_combinations)
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

print("Pmf_seen ", sum(Pmf_seen))
print("Pmf_exact_symbols ", sum(Pmf_exact_symbols.values()))
print("Pmf_V ", sum(Pmf_V))
#Sort arrays
p = V_support.argsort()
V_support = V_support[p]
Pmf_V = Pmf_V[p]

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
