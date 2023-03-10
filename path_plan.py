import matplotlib.patches
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scenario_epsilon import eps_general
from ZigZagPath import ZigZagPath
from utils import set_cover, plot_border_walls


#######################
# TRAINING
#######################
SEED = 167
np.random.seed(SEED)

episodes = 15000
explore_thr = 0.2
lr = 0.1
gamma = 0.5
actions = ['u', 'd', 'e', 'w']
# account for 10*10 positions + 1 signaling 'out of bounds'
Q = np.zeros((10*10 + 1, 4))
Q[100, :] = -1*np.ones((4,))

print('Start Training Agent')

for i in tqdm.tqdm(range(episodes)):

    # initial state
    x0 = np.random.randint(low=0, high=9, size=(2, 1))
    state = x0.copy()
    done = False
    # creates environment
    zzp = ZigZagPath(x0=x0)

    # print sort of training loss
    # if i % (episodes//100) == 0:
    #     print(np.sum(Q))

    trajectory = []
    iters = 0

    # actual training
    while not done and iters < 20:
        # idx on the Q matrix for the current state
        idx_q = state[1]*10+state[0]
        # action selection
        if np.random.rand() < explore_thr:
            # random action
            act = np.random.randint(low=0, high=4)
        else:
            # suggested action
            act = np.argmax(Q[idx_q, :])

        # update
        next_state, reward, done = zzp.step(actions[act])
        # if the agent is out of bounds, send it to an extra state with index 100
        if zzp.out_of_bounds(next_state):
            idx_q_next = 100
        else:
            idx_q_next = next_state[1] * 10 + next_state[0]

        # update Q function
        Q[idx_q, act] = (1 - lr) * Q[idx_q, act] + lr * (reward + gamma * np.max(Q[idx_q_next, :]))

        state = next_state
        iters += 1

        trajectory.append(state)


#############################
# ACTUAL SAMPLING
#############################
print('-'*80)
print('Start Trajectory Sampling')
new_experiments = 10000
BETA = 1e-12
H = 25
# plot only some trajectories
plot_thr = 10./new_experiments

# collect all trajectories
all_trajectories = np.zeros((1, H, new_experiments))

# loop over trajectories
for i in tqdm.tqdm(range(new_experiments)):

    # initial state
    x0 = np.random.randint(low=0, high=9, size=(2, 1))
    done = False

    # ensure we get a valid x0
    while zzp.into_walls(x0):
        x0 = np.random.randint(low=0, high=9, size=(2, 1))

    # create environment
    zzp = ZigZagPath(x0=x0)
    state = x0.copy()

    # plot random trajectory
    plot_flag = False
    if np.random.rand() < plot_thr:
        plot_flag = True

    if plot_flag:
        plot_border_walls()

    iters = 0
    all_trajectories[0, 0, i] = state[1]*10+state[0]
    # collect k_bar time steps per trajectory
    while iters < H-1:

        if plot_flag:
            plt.scatter(state[0], state[1], c='b')

        # idx on the Q matrix for the current state
        idx_q = state[1]*10+state[0]

        # action selct
        act = np.argmax(Q[idx_q, :])

        # update state
        next_state, reward, done = zzp.step(actions[act])
        if (zzp.into_walls(next_state) or zzp.out_of_bounds(next_state)) and not zzp.into_walls(state):
            print(f'Current State: {state.T}, Next State: {next_state.T}')
            raise ValueError('Learning Error')
        else:
            idx_q_next = next_state[1] * 10 + next_state[0]

        if done and not zzp.in_target(next_state) and iters > 0:
            print('Learning Error: Target Not Reached.')

        state = next_state
        all_trajectories[0, iters+1, i] = next_state[1] * 10 + next_state[0]
        iters += 1


##################
# SCENARIO
##################

# create set of ell_seq
ell = 2
unique_ell_seq = set()
for expm in range(new_experiments):
    # single trajectory
    for t in range(H-ell):
        # create tuple of ell-seq
        ell_seq = tuple(all_trajectories[:, t:t+ell, expm][0])
        # store in a set for uniqueness
        unique_ell_seq.add(ell_seq)

# Set cover problem to find complexity
subsets = []
for j in range(all_trajectories.shape[-1]):
    H_seq = np.squeeze(all_trajectories[:,:,j])
    seq_of_ell_seq = []
    for i in range(0,len(H_seq)-ell+1):
        seq_of_ell_seq.append(tuple(H_seq[i:i+ell]))
    subsets.append(set(seq_of_ell_seq))

cover = set_cover(unique_ell_seq, subsets)
print("Complexity (smallest cardinality of subset of k_bar-sequences that return the same solution): ", len(cover))

print(f'Number of unique ell sequences: {len(unique_ell_seq)}')
print(f'Unique ell sequences: \n{unique_ell_seq}')

print('-'*80)
epsi_up = eps_general(k=len(unique_ell_seq), N=new_experiments, beta=BETA)
print(f'Epsilon Bound With Unique l-Sequences: {epsi_up}')
epsi_up = eps_general(k=len(cover), N=new_experiments, beta=BETA)
print(f'Epsilon Bound With Set Cover: {epsi_up}')
print('-'*80)


print('Plotting...')
plt.show()
