import sys
import numpy as np
import tqdm
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scenario_epsilon import eps_general
from utils import plot_init_conditions, partitions_sequences, get_sequences_stats, \
    distribution_subplots, generate_vanderpol_traj


# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 5.8, 0.2
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
N_traj = 20000
BETA = 1e-12
time_steps = round(tmax/dt) + 1

# generates trajectories of the van der pol system
traj_domain_bounds = [-2.5, 2.5]
all_positions = generate_vanderpol_traj(N_traj, t, time_steps, traj_domain_bounds)


# boundaries for van der pol are [-3, -3], [3, 3]
parts_per_axis = 4+1
x_bounds, y_bounds = [-4, 4], [-4, 4]
# computes partition boundaries per axis x and y
parts_x_idx = np.linspace(x_bounds[0], x_bounds[1], parts_per_axis)
parts_y_idx = np.linspace(y_bounds[0], y_bounds[1], parts_per_axis)

# computes the partitions for all trajectories
all_trajectory_parts = partitions_sequences(all_positions, [parts_x_idx, parts_y_idx], N_traj,
                                            time_steps, parts_per_axis, reset_angles=False)

# find all ell-sequences from a trajectory
ell = 2
print(f'Total trajectories: {N_traj}. \nVisitable ell-sequences: {(parts_per_axis-1)**(2*ell)}. ')

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
epsi_up = eps_general(k=len(ell_seq_trajectory), N=N_traj, beta=BETA)
print(f'Epsilon Bound: {epsi_up}')
print('-'*80)


# new data
tmax, dt = 19.8, 0.2
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
time_steps = round(tmax/dt) + 1
all_positions = generate_vanderpol_traj(N_traj, t, time_steps, traj_domain_bounds)
# sequences of partitions
all_trajectory_parts = partitions_sequences(all_positions, [parts_x_idx, parts_y_idx], N_traj,
                                            time_steps, parts_per_axis, reset_angles=False)

print(f'Sequences of partitions: {all_trajectory_parts}')
# find all ell-sequences from a trajectory
print(f'Total trajectories: {N_traj}. \nVisitable Partitions: {(parts_per_axis-1)**(2*ell)}. ')

new_ell_seq_trajectory, new_ell_seq_init, new_ell_seq_rnd = get_sequences_stats(all_trajectory_parts, ell, time_steps)

tot_ell_visited = ell_seq_trajectory.union(new_ell_seq_trajectory)
tot_init_visited = ell_seq_init.union(new_ell_seq_init)
tot_rnd_visited = ell_seq_rnd.union(new_ell_seq_rnd)

print(f'New sequences seen: {len(tot_ell_visited)-len(ell_seq_trajectory)}')
print(f'New initial seen: {len(tot_init_visited)-len(ell_seq_init)}')
print(f'New random seen: {len(tot_rnd_visited)-len(ell_seq_rnd)}')

n_vars = 2
distribution_subplots(all_trajectory_parts, time_steps=6+ell,
                      parts_per_axis=parts_per_axis+1, n_vars=n_vars, plot_every=1, ell=ell)

plt.show()
