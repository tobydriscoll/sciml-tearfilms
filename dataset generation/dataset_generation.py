# %%
import os
import h5py
import numpy as np
from scipy.integrate import solve_ivp

# File paths
old_file_path = "C:/Users/arnab/OneDrive/Desktop/Study material/summer 24/Research/Old equations/datasets/trials_many_alldata_v3.h5"
new_file_path = "C:/Users/arnab/OneDrive/Desktop/Study material/summer 24/Research/New equations/datasets/trials_many_alldata_new_eqution.h5"

# Ensure the directory exists
new_dir = os.path.dirname(new_file_path)
os.makedirs(new_dir, exist_ok=True)

# Constants
P_c = 0.0653
phi = 0.279
h_e_range = (1e-5, 1e-3)

# Function to calculate I
def calculate_I(h, f, I_0=1.0):
    return I_0 * (1 - np.exp(phi * h * f)) / (1 + f ** 2)

# Differential equation system
def differential_system(t, y, J_e, b_1, b_2, h_e):
    h, c, f = y
    g_t = b_1 * np.exp(b_2 * t)
    dh_dt = -g_t * h - J_e * (1 - (h_e / h) ** 3)
    dc_dt = (J_e - P_c * (c - 1)) * c / h
    df_dt = (J_e - P_c * (c - 1)) * f / h
    return [dh_dt, dc_dt, df_dt]

# Main function
def generate_dataset():
    with h5py.File(old_file_path, 'r') as old_data, h5py.File(new_file_path, 'w') as new_data:
        # Extract parameter values
        P_old = old_data['P'][:]
        J_e_values = P_old[:, 0]
        b_1_values = P_old[:, 1]
        b_2_values = P_old[:, 2]
        f_0_values = P_old[:, 6]
        time_grid = np.linspace(0, 1, 201)

        # Create datasets in the new file
        new_h = new_data.create_dataset("h", (0, len(time_grid)), maxshape=(None, len(time_grid)))
        new_c = new_data.create_dataset("c", (0, len(time_grid)), maxshape=(None, len(time_grid)))
        new_f = new_data.create_dataset("f", (0, len(time_grid)), maxshape=(None, len(time_grid)))
        new_I = new_data.create_dataset("I", (0, len(time_grid)), maxshape=(None, len(time_grid)))
        new_P = new_data.create_dataset("P", (0, P_old.shape[1] + 1), maxshape=(None, P_old.shape[1] + 1))

        # Iterate over parameter sets
        for i, (J_e, b_1, b_2, f_0) in enumerate(zip(J_e_values, b_1_values, b_2_values, f_0_values)):
            while True:  # Repeat until acceptable data is generated
                # Randomly choose h_e
                h_e = np.random.uniform(*h_e_range)

                # Solve the differential equations
                y0 = [1, 1, f_0]
                sol = solve_ivp(
                    differential_system,
                    [time_grid[0], time_grid[-1]],
                    y0,
                    t_eval=time_grid,
                    args=(J_e, b_1, b_2, h_e)
                )

                h, c, f = sol.y
                I = calculate_I(h, f)

                # Check acceptability
                if np.all(h <= 1.1) and np.all(c <= 1.1) and np.all(I <= 1.1):
                    # Data is acceptable, save it
                    new_h.resize(new_h.shape[0] + 1, axis=0)
                    new_c.resize(new_c.shape[0] + 1, axis=0)
                    new_f.resize(new_f.shape[0] + 1, axis=0)
                    new_I.resize(new_I.shape[0] + 1, axis=0)
                    new_P.resize(new_P.shape[0] + 1, axis=0)

                    new_h[-1] = h
                    new_c[-1] = c
                    new_f[-1] = f
                    new_I[-1] = I

                    # Append parameter values including h_e
                    new_P[-1] = np.append(P_old[i], h_e)
                    break

# Run the function
generate_dataset()



