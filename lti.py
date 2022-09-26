import sys
import numpy as np
import tqdm
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from utils import plot_init_conditions, partitions_grid, get_sequences_stats, \
    distribution_subplots, generate_vanderpol_traj
from scenario_epsilon import eps_general


def contractive_system(A, x0, N_steps):
    # generates N trajectories starting from x0

    traj = np.zeros((N_steps, x0.shape[0]))
    traj[0] = x0
    for i in range(N_steps-1):
        xi = traj[i]
        traj[i+1] = A @ xi

    return traj


A = 1./3. * np.array([
        [1., 2.],
        [-1., 1.]
    ])

eigs = np.linalg.eigvals(A)
norm_eigs = np.absolute(eigs)
contraction = np.max(norm_eigs)
print(f'Model Eigenvalues: {eigs}, contraction rate: {contraction}')


# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 3, 1
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
N_traj = 10000
time_steps = tmax + 1
n_vars = 2

domain_bounds = [-1., 1.]
all_positions = np.zeros((N_traj, time_steps, n_vars))
for i in tqdm.tqdm(range(N_traj)):
    x0 = np.random.uniform(low=domain_bounds[0], high=domain_bounds[1], size=(n_vars,))
    all_positions[i, :] = contractive_system(A, x0, time_steps)

# boundaries for system are [0,1]
x_bounds = domain_bounds
y_bounds = domain_bounds
n_partitions = 9
parts_x_idx = np.linspace(x_bounds[0], x_bounds[1], n_partitions+1)
parts_y_idx = np.linspace(y_bounds[0], y_bounds[1], n_partitions+1)
boundaries = [parts_x_idx, parts_y_idx]

parts_per_axis = len(parts_x_idx)-1

all_trajectory_parts = partitions_grid(all_positions, boundaries, N_traj, time_steps)

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
    print(f'Same number of seen and initial sequences: ({len(ell_seq_init)}).')


print('-'*80)
epsi_up = eps_general(k=len(ell_seq_trajectory), N=N_traj, beta=1e-12)
print(f'Epsilon Bound: {epsi_up}')
print('-'*80)

###################################
#  infinite horizon
###################################

# find H as the number of steps to exit the domain
# not really accurate, but works
idx = np.where(parts_x_idx > 0)[0][0]
start = parts_x_idx[idx]

H = 1
while start < x_bounds[1]:
    H = H+1
    start = start * contraction**-1

k = np.minimum(time_steps, H)
gamma_bar = ( (1 - contraction**(k-H-1)) / (1 - contraction**-1) ) * epsi_up

print(f'Infinite Horizon PAC bounds: {gamma_bar}')

#################################
# new data
#################################
tmax = 10
N_traj = 100000
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
time_steps = tmax + 1
all_positions = np.zeros((N_traj, time_steps, n_vars))
for i in range(N_traj):
    x0 = np.random.uniform(low=domain_bounds[0], high=domain_bounds[1], size=(n_vars,))
    all_positions[i, :] = contractive_system(A, x0, time_steps)

# sequences of partitions
all_trajectory_parts = partitions_grid(all_positions, boundaries, N_traj, time_steps)

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