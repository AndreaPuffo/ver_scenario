import sys
import numpy as np
import tqdm
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pendulum_utils import plot_init_conditions, partitions_1D, get_sequences_stats, \
    distribution_subplots, generate_vanderpol_traj
from scenario_epsilon import epsLU


def contractive_system(x0, N_steps, epsilon):
    # generates N trajectories starting from x0
    traj = np.zeros((N_steps, ))
    traj[0] = x0
    for i in range(N_steps-1):
        xi = traj[i]
        if xi > epsilon:
            traj[i+1] = 0.5 * xi
        else:
            traj[i+1] = 0.5 * xi + 0.5

    return traj


# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 10, 1
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
N_traj = 20000
time_steps = tmax + 1
n_vars = 1


traj_domain_bounds = [0, 1]
all_positions = np.zeros((N_traj, time_steps))
lamb = 1e-2
for i in range(N_traj):
    x0 = np.random.uniform(low=0., high=1., size=(1,))
    all_positions[i, :] = contractive_system(x0, time_steps, lamb)

# boundaries for system are [0,1]
x_bounds = [0., 1.]
parts_x_idx = [1., 0.5, 0.25, 0.125, 0.0625, 0.]
parts_per_axis = len(parts_x_idx)-1

all_trajectory_parts = partitions_1D(all_positions, parts_x_idx, N_traj, time_steps)

print(f'Partitions: {all_trajectory_parts}')

# find all ell-sequences from a trajectory
ell = 2

assert ell <= time_steps

print(f'Total trajectories: {N_traj}. \nVisitable ell-sequences: {(parts_per_axis)**(n_vars*ell)}. ')

ell_seq_trajectory, ell_seq_init, ell_seq_rnd = get_sequences_stats(all_trajectory_parts, ell, time_steps)

if len(ell_seq_trajectory) > len(ell_seq_init) and len(ell_seq_trajectory) > len(ell_seq_rnd):
    print(f'Visited ell-sequences are more than the initial ones: \n'
          f'visited {len(ell_seq_trajectory)}, initial: {len(ell_seq_init)}, '
          f'random: {len(ell_seq_rnd)}.')
elif len(ell_seq_trajectory) == len(ell_seq_rnd) and len(ell_seq_trajectory) > len(ell_seq_init):
    print(f'Randomly picked ell-sequences == visited partitions: \n'
          f'visited {len(ell_seq_trajectory)}, initial: {len(ell_seq_init)}, '
          f'random: {len(ell_seq_rnd)}.')
else:
    print(f'Same number of seen and initial sequences.')


print('-'*80)
epsi_lo, epsi_up = epsLU(k=len(ell_seq_trajectory), N=N_traj, beta=1e-12)
print(f'Epsilon Bound: {epsi_up}')
print('-'*80)

if epsi_up < lamb:
    print(f'Abstraction *should* simulate system')


# new data
tmax = 10
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
time_steps = tmax + 1
all_positions = np.zeros((N_traj, time_steps))
for i in range(N_traj):
    x0 = np.random.uniform(low=0., high=1., size=(1,))
    all_positions[i, :] = contractive_system(x0, time_steps, lamb)

# sequences of partitions
all_trajectory_parts = partitions_1D(all_positions, parts_x_idx, N_traj, time_steps)

# print(f'Sequences of partitions: {all_trajectory_parts}')
# find all ell-sequences from a trajectory
print(f'Total trajectories: {N_traj}. \nVisitable Partitions: {(parts_per_axis)**(n_vars*ell)}. ')

new_ell_seq_trajectory, new_ell_seq_init, new_ell_seq_rnd = get_sequences_stats(all_trajectory_parts, ell, time_steps)

tot_ell_visited = ell_seq_trajectory.union(new_ell_seq_trajectory)
tot_init_visited = ell_seq_init.union(new_ell_seq_init)
tot_rnd_visited = ell_seq_rnd.union(new_ell_seq_rnd)

print(f'New sequences seen: {len(tot_ell_visited)-len(ell_seq_trajectory)}')
print(f'New initial seen: {len(tot_init_visited)-len(ell_seq_init)}')
print(f'New random seen: {len(tot_rnd_visited)-len(ell_seq_rnd)}')

print(f'Old sequences: {ell_seq_trajectory}')
print(f'New sequences: {tot_ell_visited}')

distribution_subplots(all_trajectory_parts, time_steps=6+ell,  # time_steps,
                      parts_per_axis=parts_per_axis+1, n_vars=n_vars, plot_every=1, ell=ell)


plt.show()
