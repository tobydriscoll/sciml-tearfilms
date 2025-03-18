import os
import h5py
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats.qmc import Halton

# File paths
old_file_path = "C:/Users/arnab/OneDrive/Desktop/Study material/summer 24/Research/Old equations/datasets/trials_many_alldata_v3.h5"
new_file_path = "C:/Users/arnab/OneDrive/Desktop/Study material/summer 24/Research/New equations/datasets/trials_many_alldata_new_eqution_halton.h5"

# Ensure the directory exists
new_dir = os.path.dirname(new_file_path)
os.makedirs(new_dir, exist_ok=True)

# Constants
P_c = 0.0653
phi = 0.279
h_e_range = [0.5, 1]
rho = 1e3
d_range = [2e-6, 5e-6]
l_range = [0.138e-3, 0.412e-3]
U_range = [0.056e-3, 0.099e-3]
ts_range = [1.75, 6.6]


# Function to calculate I
def calculate_I(h, f, I_0=1.0):
    return I_0 * (1 - np.exp(-phi * h * f)) / (1 + f ** 2)

# Differential equation system
def differential_system(t, y, J_e, b_1, b_2, h_e):
    h, c = y
    g_t = b_1 * np.exp(b_2 * t)
    dh_dt = -g_t * h - J_e * (1 - (h_e / h) ** 3) + P_c * (c-1)
    dc_dt = (J_e * (1 - (h_e / h) ** 3) - P_c * (c - 1)) * c / h
    return [dh_dt, dc_dt]

with h5py.File(old_file_path, 'r') as old_data:
    # Extract parameter values
    P_old = old_data['P'][:]
    J_e_primes = P_old[:, 0]
    b_1_primes = P_old[:, 1]
    b_2_primes = P_old[:, 2]
    f_0_primes = P_old[:, 6]

    # Find the shortest length
    min_length = min(len(J_e_primes), len(b_1_primes), len(b_2_primes), len(f_0_primes))
    # Function to randomly reduce the length of a list
    def reduce_length(arr, target_length):
        if len(arr) > target_length:
            indices = np.random.choice(len(arr), target_length, replace=False)
            return arr[indices]
        return arr

    # Reduce lengths to the shortest length
    J_e_primes = reduce_length(J_e_primes, min_length)
    b_1_primes = reduce_length(b_1_primes, min_length)
    b_2_primes = reduce_length(b_2_primes, min_length)
    f_0_primes = reduce_length(f_0_primes, min_length)
    f_cr = 5.3e-3
    f_0_values = f_0_primes/f_cr
    h_0_values = P_old[:, 4]

time_grid = np.linspace(0, 1, 601)

halton_sampler = Halton(d=1, scramble=False, seed=42)
halton_sequence = halton_sampler.random(n=111563).flatten()
# Rescale to be in the range 0.5 to 1.0
h_e_halton = h_e_range[0] + (h_e_range[1] - h_e_range[0]) * halton_sequence

d_halton = d_range[0] + (d_range[1] - d_range[0]) * halton_sequence

l_halton = l_range[0] + (l_range[1] - l_range[0]) * halton_sequence

U_halton = U_range[0] + (U_range[1] - U_range[0]) * halton_sequence

ts_halton = ts_range[0] + (ts_range[1] - ts_range[0]) * halton_sequence

J_e_values = J_e_primes * l_halton / (rho * d_halton * U_halton)

b_1_values = b_1_primes * ts_halton

b_2_values = b_2_primes * ts_halton

h_e_values = np.array(h_e_halton) / np.array(h_0_values)

# Differential system (P_c is a constant and does not need to be passed as an argument)
def differential_system(t, y, J_e, b_1, b_2, h_e):
    h, c = y
    g_t = b_1 * np.exp(b_2 * t)
    dh_dt = -g_t * h - J_e * (1 - (h_e / h) ** 3) + P_c * (c - 1)
    dc_dt = (J_e * (1 - (h_e / h) ** 3) - P_c * (c - 1)) * c / h
    return [dh_dt, dc_dt]

def generate_and_store_dataset():
    valid_h, valid_c, valid_f, valid_I, valid_p = [], [], [], [], []
    
    for i, (J_e, b_1, b_2, h_e, f_0) in enumerate(zip(J_e_values, b_1_values, b_2_values, h_e_values, f_0_values)):
        y0 = [1.0, 1.0]  # Initial conditions: h(0) = 1.0, c(0) = 1.0
        sol = solve_ivp(differential_system, [0., 1.], y0, t_eval=time_grid, 
                        args=(J_e, b_1, b_2, h_e), method='RK45')
        
        if np.max(sol.y[0]) > 1.1:
            # print(f"{i}'th parameter set is rejected.")
            continue  # Discard solutions where h exceeds 1.1
        
        # print(f"{i}'th parameter set is accepted.")
        h_values = sol.y[0]
        c_values = sol.y[1]
        f_values = f_0 * c_values
        I_values = calculate_I(h_values, f_values)
        
        valid_h.append(h_values)
        valid_c.append(c_values)
        valid_f.append(f_values)
        valid_I.append(I_values)
        valid_p.append([J_e, b_1, b_2, h_e * h_0_values[i]])  # Store h_e_prime instead of h_e

    # Convert lists to NumPy arrays
    valid_h = np.array(valid_h)
    valid_c = np.array(valid_c)
    valid_f = np.array(valid_f)
    valid_I = np.array(valid_I)
    valid_p = np.array(valid_p)

    # Save data in HDF5 format
    with h5py.File(new_file_path, "w") as hdf_file:
        hdf_file.create_dataset("h", data=valid_h)
        hdf_file.create_dataset("c", data=valid_c)
        hdf_file.create_dataset("f", data=valid_f)
        hdf_file.create_dataset("I", data=valid_I)
        hdf_file.create_dataset("p", data=valid_p)
        
    print(f"Total accepted parameter sets: {valid_h.shape[0]}")

generate_and_store_dataset()
print("Dataset generation completed.")