import sys
import time
import numpy as np
import tqdm
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from utils import plot_init_conditions, partitions_grid, get_sequences_stats, \
    distribution_subplots, generate_vanderpol_traj, set_cover, greedy_set_cover, \
    match_rows
from scenario_epsilon import eps_general
import multiprocessing as mp



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
        [-1.8, 1.]
    ])



eigs = np.linalg.eigvals(A)
norm_eigs = np.absolute(eigs)
contraction = np.max(norm_eigs)
print(f'Model Eigenvalues: {eigs}, contraction rate: {contraction}')


# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 8, 1
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
N_traj = 1000000
H = tmax + 1
n_vars = 2

domain_bounds = [-1., 1.]
all_positions = np.zeros((N_traj, H, n_vars))
for i in tqdm.tqdm(range(N_traj)):
    x0 = np.random.uniform(low=domain_bounds[0], high=domain_bounds[1], size=(n_vars,))
    all_positions[i, :] = contractive_system(A, x0, H)

# boundaries for system are [0,1]
x_bounds = domain_bounds
y_bounds = domain_bounds
n_partitions = 82-1
parts_x_idx = np.linspace(x_bounds[0], x_bounds[1], n_partitions+1)
parts_y_idx = np.linspace(y_bounds[0], y_bounds[1], n_partitions+1)
boundaries = [parts_x_idx, parts_y_idx]


print(f'Partition intervals: {len(parts_x_idx)-1}')

assert np.mod(len(parts_x_idx)-1, 2) == 1, "Number of partitions must be odd (for the center)."

parts_per_axis = len(parts_x_idx)-1

all_trajectory_parts = partitions_grid(all_positions, boundaries, N_traj, H)

print(f'Partitions: {all_trajectory_parts}')

# find all ell-sequences from a trajectory
ell = H

assert ell <= H

print(f'Total trajectories: {N_traj}. \nVisitable ell-sequences: {(parts_per_axis)**(n_vars*ell)}. ')

ell_seq_trajectory, ell_seq_init, ell_seq_rnd = get_sequences_stats(all_trajectory_parts, ell, H)

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

num_sets = 0
if ell < H:
    # Recast for set cover problem
    subsets = []
    for H_seq in all_trajectory_parts:
        seq_of_ell_seq = []
        for i in range(0, len(H_seq)-ell+1):
            seq_of_ell_seq.append(tuple(H_seq[i:i+ell]))
        subsets.append(set(seq_of_ell_seq))
    tic = time.perf_counter()
    num_sets = greedy_set_cover(subsets,ell_seq_trajectory)
    toc = time.perf_counter()
    print(f"Time elapsed: {toc - tic:0.4f} seconds")
else:
    num_sets = len(ell_seq_trajectory)


print("Upper bound of complexity ", num_sets)

print('-'*80)
epsi_up = eps_general(k=num_sets, N=N_traj, beta=1e-12)
print(f'Epsilon Bound using complexity: {epsi_up}')
print('-'*80)
'''
# Comparison with ell = H
ell_1 = H


print(f'Total trajectories: {N_traj}. \nVisitable ell-sequences: {(parts_per_axis)**(n_vars*ell_1)}. ')

ell_seq_trajectory, ell_seq_init, ell_seq_rnd = get_sequences_stats(all_trajectory_parts, ell_1, H)

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
epsi_up_1 = eps_general(k=len(ell_seq_trajectory), N=N_traj, beta=1e-12)
print(f'Epsilon Bound, based on unique ell-sequences: {epsi_up_1}')
print('-'*80)
'''
###################################
#  infinite horizon
###################################

# find k_bar as the number of steps to exit the domain
# not really accurate, but works
idx = np.where(parts_x_idx > 0)[0][0]
start = parts_x_idx[idx]

# Compute inverse determinant
rho = np.abs(np.linalg.det(A))**(-1)
print(f'Inverse determinant: {rho}')
rho = np.ceil(rho)
print(f'Upperbound rho: {rho}')
# Compute the induced 2-norm of matrix A
alfa = np.linalg.norm(A, ord=2)
print(f'Induced 2-norm: {alfa}')
alfa = np.ceil(alfa*10)/10
print(f'Upperbound alfa: {alfa}')
# Compute d_min and d_max for the computation of \bar{k}
d_max = 0
if x_bounds == y_bounds and np.abs(x_bounds[0]) == x_bounds[1]:
    d_max = x_bounds[1]
else:
    print("Adjust bounds (or code)!")
# Compute the size of the interval of parts_x_idx
#d_min = (parts_x_idx[1] - parts_x_idx[0])/2
# Find closest point to 0 in parts_x_idx
d_min = np.abs(parts_x_idx - 0).min()
assert d_min != 0
print(f'd_min: {d_min}')
k_bar = np.ceil(np.log(d_min / d_max) / np.log(alfa))
print(f'Number of steps to exit the domain: {k_bar}')
assert k_bar >= (H-ell)
phi = np.arange(0, k_bar+1)
for i in range(0, len(phi)):
    z = np.ceil((k_bar+1)/(i + 1))-1
    phi[i] = 1
    t = 0
    while t <= z-1:
        phi[i] += rho**(k_bar-t*(i+1) - i)
        t = t + 1
    phi[i] = 1/phi[i]
#plt.plot(phi)
print(f'H-ell: {H-ell}')
print(f'phi[H-ell]: {phi[H-ell]}')
gamma_bar = epsi_up/phi[H-ell]
print(f'Infinite Horizon PAC bounds: {gamma_bar}')

#################################
# Domino Completion
#################################

# Domino Minimal completion
# Check if the first ell-1 entries of every element in ell_seq_trajectory concide with the last ell-1 entries of any other element
# If not, then check if the first ell-2 entries of every element in ell_seq_trajectory concide with the last ell-2 entries of any other element
#ell = 5
#ell_seq_trajectory = set([(1,1,1),(1,2,3),(2,3,1),(2,2,2)])
#ell_seq_trajectory = set([(1,1,1,1,1),(1,1,1,1,2), (2,1,2,1,1), (3,1,1,1,2)])
'''
# Parallelized version, but not fast after all...
print(f'Number of unique ell-sequences before domino completion: {len(ell_seq_trajectory)}')
tic = time.perf_counter()
missing = None
flag_missing = 0
ell_seq_trajectory = np.array(list(l_seq for l_seq in ell_seq_trajectory), dtype=int)
new_ell_seq_trajectory = ell_seq_trajectory.copy()
u = 1
L = len(ell_seq_trajectory)
tic = time.perf_counter()
while True:
    L_complete = len(new_ell_seq_trajectory)

    # Check if any suffix is missing in the prefix and remove complete suffixes
    complete_suffixes = []
    for i in range(len(ell_seq_trajectory)):
        # Select l-sequence to be completed
        matched_suffix_flag = 0
        for j in range(L_complete):
            # There exists a prefix that matches the suffix
            if np.array_equal(ell_seq_trajectory[i][-ell+1:], new_ell_seq_trajectory[j][0:ell - 1]):
                matched_suffix_flag = 1
                complete_suffixes.append(i)
                break
        # No match is found
        if matched_suffix_flag == 0:
            missing = ell_seq_trajectory[i]
            flag_missing = 1
            break
        else:
            flag_missing = 0
    # Remove complete suffixes
    ell_seq_trajectory = np.delete(ell_seq_trajectory, complete_suffixes, axis=0)
    #print("Missing ", missing)
    concatenated_sequence = None

    # Count the number of CPU cores
    num_processes = mp.cpu_count()

    # Split new_ell_seq_trajectory into chunks to be processed by each process
    chunk_size = int(np.ceil(len(new_ell_seq_trajectory) / num_processes))

    # Create a multiprocessing queue to receive results
    result_queue = mp.Queue()

    # Minimal completion
    if flag_missing == 1:

        # Find the largest l resulting in a match
        for l in reversed(range(1, ell - 1)):

            # Create a list of processes
            processes = []

            # Assign a chunk of new_ell_seq_trajectory to each process
            for i in range(num_processes):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(new_ell_seq_trajectory))
                p = mp.Process(target=match_rows, args=(missing, new_ell_seq_trajectory, l, start_idx, end_idx, result_queue))
                processes.append(p)

            # Start the processes
            for p in processes:
                p.start()

            # Wait for all processes to complete
            for p in processes:
                p.join()

            # Check if any matches were found
            if result_queue.empty():
                match = None
            else:
                # Get the first match
                match = result_queue.get()

            # Terminate the remaining processes
            for p in processes:
                if p.is_alive():
                    p.terminate()

            # Concatenate matching sequences
            if match is not None:
                concatenated_sequence = np.concatenate((missing, match[l:]))
                break

        # No minimal completion found
        if concatenated_sequence is None:
            print("No minimal completion found, Error!")
            assert concatenated_sequence is not None
            break
        # Obtain every subsequence of length ell from concatenated_sequence except the last one and first one and append
        # to new_ell_seq_trajectory, sinde they are already completed
        for i in range(len(concatenated_sequence)-ell-1):
            #print(concatenated_sequence[1+i:1+i+ell])
            new_ell_seq_trajectory = np.append(new_ell_seq_trajectory, [concatenated_sequence[1 + i:1 + i + ell]], axis=0)
    else:
        break
    # Print every now and then
    if (L_complete-L)/L > u/1000:
        toc = time.perf_counter()
        print(f'Elapsed time: {toc-tic:0.4f} seconds')
        print(f'Remaining ell-sequences: {len(ell_seq_trajectory)}')
        print(f'Number of unique ell-sequences after domino completion: {L_complete}')
        print(f'Percentage of ell-sequences added wrt to the original set: {u/10}%')
        u = u + 1
        tic = toc
toc = time.perf_counter()
print(f'Elapsed time: {toc-tic:0.4f} seconds')
print(f'Number of unique ell-sequences after domino completion: {len(new_ell_seq_trajectory)}')
exit()
ell_seq_trajectory = new_ell_seq_trajectory.copy()
'''

print(f'Number of unique ell-sequences before domino completion: {len(ell_seq_trajectory)}')
tic = time.perf_counter()
missing = None
flag_missing = 0
ell_seq_trajectory = np.array(list(l_seq for l_seq in ell_seq_trajectory), dtype=int)
new_ell_seq_trajectory = ell_seq_trajectory.copy()
u = 1
L = len(ell_seq_trajectory)
tic = time.perf_counter()
while True:
    L_complete = len(new_ell_seq_trajectory)

    # Check if any suffix is missing in the prefix and remove complete suffixes
    complete_suffixes = []
    for i in range(len(ell_seq_trajectory)):
        # Select l-sequence to be completed
        matched_suffix_flag = 0
        for j in range(L_complete):
            # There exists a prefix that matches the suffix
            if np.array_equal(ell_seq_trajectory[i][-ell+1:], new_ell_seq_trajectory[j][0:ell - 1]):
                matched_suffix_flag = 1
                complete_suffixes.append(i)
                break
        # No match is found
        if matched_suffix_flag == 0:
            missing = ell_seq_trajectory[i]
            flag_missing = 1
            break
        else:
            flag_missing = 0
    # Remove complete suffixes
    ell_seq_trajectory = np.delete(ell_seq_trajectory, complete_suffixes, axis=0)
    #print("Missing ", missing)
    concatenated_sequence = None
    # Minimal completion
    if flag_missing == 1:
        for l in reversed(range(1, ell - 1)):
            for i in range(L_complete):
                if np.array_equal(missing[-l:], new_ell_seq_trajectory[i][0:l]):
                    #print("Candidate for merge:", new_ell_seq_trajectory[i])
                    concatenated_sequence = np.concatenate((missing, new_ell_seq_trajectory[i][l:]))
                    #print(concatenated_sequence)
                    break
            if concatenated_sequence is not None:
                break
        # No minimal completion found
        if concatenated_sequence is None:
            #print("No minimal completion found, Error!")
            assert concatenated_sequence is not None
            break
        # Obtain every subsequence of length ell from concatenated_sequence except the last one and first one and append
        # to new_ell_seq_trajectory, sinde they are already completed
        for i in range(len(concatenated_sequence)-ell-1):
            #print(concatenated_sequence[1+i:1+i+ell])
            new_ell_seq_trajectory = np.append(new_ell_seq_trajectory, [concatenated_sequence[1 + i:1 + i + ell]], axis=0)
    else:
        break
    # Print every now and then
    if (L_complete-L)/L > u/100:
        toc = time.perf_counter()
        print(f'Elapsed time: {toc-tic:0.4f} seconds')
        print(f'Remaining ell-sequences: {len(ell_seq_trajectory)}')
        print(f'Number of unique ell-sequences after domino completion: {L_complete}')
        print(f'Percentage of ell-sequences added wrt to the original set: {u}%')
        u = u + 1
        tic = toc
toc = time.perf_counter()
print(f'Elapsed time: {toc-tic:0.4f} seconds')
print(f'Number of unique ell-sequences after domino completion: {len(new_ell_seq_trajectory)}')

ell_seq_trajectory = new_ell_seq_trajectory.copy()

ell_seq_trajectory = set(tuple(s) for s in ell_seq_trajectory)
'''
# Double check that set is domino complete



print(f'Number of unique ell-sequences before domino completion: {len(ell_seq_trajectory)}')
prefixes = set()
suffixes = set()
u = 1
missing = set()
outer_tic = time.perf_counter()
while True:
    for l_seq in ell_seq_trajectory:
        prefixes.add(l_seq[0:ell-1])
        suffixes.add(l_seq[1:ell])

    for suff in suffixes:
        if suff not in prefixes:
            missing.add(suff)
            break
    if len(missing) == 0:
        break
    else:
        pref = missing.pop()
        for i in range(n_partitions**n_vars):
            temp = list(pref)
            temp.append(i+1)
            #print("Adding ", tuple(temp), " to ell_seq_trajectory")
            ell_seq_trajectory.add(tuple(temp))

    if len(ell_seq_trajectory)/u > 10000:
        toc = time.perf_counter()
        print(f'Elapsed time: {toc-tic:0.4f} seconds')
        print(f'Number of unique ell-sequences after domino completion: {len(ell_seq_trajectory)}')
        u = u + 1
        tic = toc
outer_toc = time.perf_counter()
print(f'Elapsed time: {outer_toc-outer_tic:0.4f} seconds')
print(f'Number of unique ell-sequences after domino completion: {len(ell_seq_trajectory)}')
'''
#################################
# new data
#################################

tmax = 8
N_traj = 100000
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
H = tmax + 1
all_positions = np.zeros((N_traj, H, n_vars))
for i in range(N_traj):
    x0 = np.random.uniform(low=domain_bounds[0], high=domain_bounds[1], size=(n_vars,))
    all_positions[i, :] = contractive_system(A, x0, H)

# sequences of partitions
all_trajectory_parts = partitions_grid(all_positions, boundaries, N_traj, H)

# print(f'Sequences of partitions: {all_trajectory_parts}')
# find all ell-sequences from a trajectory
print(f'Total trajectories: {N_traj}. \nVisitable Partitions: {(parts_per_axis)**(n_vars*ell)}. ')

# Divide every H-sequence into ell-sequences
subsets = []
for H_seq in all_trajectory_parts:
    seq_of_ell_seq = []
    for i in range(0, len(H_seq)-ell+1):
        seq_of_ell_seq.append(tuple(H_seq[i:i+ell]))
    subsets.append(set(seq_of_ell_seq))
N_Viol = 0
for H_seq in subsets:
    if not H_seq.issubset(ell_seq_trajectory):
        N_Viol += 1

print(f'Percentage of trajectories violating the solution: {N_Viol/N_traj*100}%')
'''
new_ell_seq_trajectory, new_ell_seq_init, new_ell_seq_rnd = get_sequences_stats(all_trajectory_parts, ell, H)

tot_ell_visited = ell_seq_trajectory.union(new_ell_seq_trajectory)
tot_init_visited = ell_seq_init.union(new_ell_seq_init)
tot_rnd_visited = ell_seq_rnd.union(new_ell_seq_rnd)

print(f'New sequences seen: {len(tot_ell_visited)-len(ell_seq_trajectory)}')
print(f'New initial seen: {len(tot_init_visited)-len(ell_seq_init)}')
print(f'New random seen: {len(tot_rnd_visited)-len(ell_seq_rnd)}')

print(f'Old sequences: {ell_seq_trajectory}')
print(f'New sequences: {tot_ell_visited}')

distribution_subplots(all_trajectory_parts, time_steps=6+ell,  # H,
                      parts_per_axis=parts_per_axis+1, n_vars=n_vars, plot_every=1, ell=ell)


plt.show()
'''