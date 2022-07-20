import sys
import numpy as np
import tqdm
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scenario_epsilon import eps_general
from utils import plot_init_conditions, partitions_sequences, get_sequences_stats, \
    distribution_subplots, generate_traj_doublepend, get_partition_stats


SEED = 5190
np.random.seed(SEED)

# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 0.3, 0.1
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
N_traj = 10000
time_steps = round(tmax/dt) + 1
friction = False
# generate trajectories for the double pendulum system
all_positions = generate_traj_doublepend(N_traj, t, time_steps, friction_flag=friction)

if N_traj <= 1000:
    # for t in range(time_steps//10):
    fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
    ax = fig.add_subplot(111)
    plot_init_conditions(all_positions, ax)


# boundaries for pendulum are [-2, -2], [2, 2] with top part actually empty
parts_per_axis = 3+1
x_bounds, y_bounds = [0., 2*np.pi], [0., 2*np.pi]
# computes partition boundaries for theta 1 and theta2
parts_x_idx = np.linspace(x_bounds[0], x_bounds[1], parts_per_axis)
parts_y_idx = np.linspace(y_bounds[0], y_bounds[1], parts_per_axis)

# computes the partitions for all trajectories
all_trajectory_parts = partitions_sequences(all_positions, [parts_x_idx, parts_y_idx], N_traj,
                                            time_steps, parts_per_axis)

n_vars = 2
print(f'Partitions: {all_trajectory_parts}')

# find all ell-sequences from a trajectory
ell = 2
print(f'Total trajectories: {N_traj}. \nVisitable ell-sequences: {(parts_per_axis-1)**(n_vars*ell)}. ')

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
epsi_up = eps_general(k=len(ell_seq_trajectory), N=N_traj, beta=1e-12)
print(rf'Epsilon Bound with All $\ell$-sequences: {epsi_up}')
epsi_up = eps_general(k=len(ell_seq_init), N=N_traj, beta=1e-12)
print(rf'Epsilon Bound with Initial $\ell$-sequences: {epsi_up}')
print('-'*80)


# new data
tmax = 0.6
N_traj = 100000
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
time_steps = round(tmax/dt) + 1
all_positions = generate_traj_doublepend(N_traj, t, time_steps, friction_flag=friction)
# sequences of partitions
all_trajectory_parts = partitions_sequences(all_positions, [parts_x_idx, parts_y_idx], N_traj,
                                            time_steps, parts_per_axis)

print(f'Sequences of partitions: {all_trajectory_parts}')
# find all ell-sequences from a trajectory
print(f'Total trajectories: {N_traj}. \nVisitable Partitions: {(parts_per_axis-1)**(n_vars*ell)}. ')

new_ell_seq_trajectory, new_ell_seq_init, new_ell_seq_rnd = get_sequences_stats(all_trajectory_parts, ell, time_steps)

ell_seq_stats, tot_ell_sequences = get_partition_stats(all_trajectory_parts, ell)

tot_ell_visited = ell_seq_trajectory.union(new_ell_seq_trajectory)
tot_init_visited = ell_seq_init.union(new_ell_seq_init)
tot_rnd_visited = ell_seq_rnd.union(new_ell_seq_rnd)

only_new_sequences = tot_ell_visited.difference(ell_seq_trajectory)

empirical_witness_new_seq = {}
for ell_seq in only_new_sequences:
    empirical_witness_new_seq[ell_seq] = ell_seq_stats[ell_seq]


print(f'New sequences seen: {len(tot_ell_visited)-len(ell_seq_trajectory)}')
empirical_gamma = empirical_witness_new_seq[ell_seq]/tot_ell_sequences
for ell_seq in only_new_sequences:
    print(f'Empirical frequency of {ell_seq}: {empirical_gamma}')

# find N to match the empirical gamma
print('Computing N to match gamma...')
while epsi_up > empirical_gamma:
    N_traj = 10*N_traj
    print(f'Currently at {N_traj}')
    epsi_up = eps_general(k=len(ell_seq_trajectory), N=N_traj, beta=1e-12)

if epsi_up <= empirical_gamma:
    print(f'Need {N_traj} to match the empirical gamma!')


distribution_subplots(all_trajectory_parts, time_steps=4+ell, parts_per_axis=parts_per_axis, n_vars=n_vars,
                      plot_every=2, ell=ell)


plt.show()
