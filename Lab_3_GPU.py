# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hyperparameters import hyperparameters # Assuming this exists and works
# Try importing CuPy
try:
    import cupy as cp
    xp = cp
    gpu_available = True
    print("CuPy found, using GPU.")
except ImportError:
    xp = np
    gpu_available = False
    print("CuPy not found, using NumPy on CPU.")

from Digital_twin import DigitalTwin # Assuming this exists, though it's less used in the vectorized version

hp = hyperparameters()

# Path to the CSV file
csv_file_path = 'filtered_low-pass_data-units.csv'
df = pd.read_csv(csv_file_path)

sample_rate = 40  # Hz
start_time = 0
# start_time = int(round(7.67 * sample_rate))
end_time = start_time + 30 * sample_rate

highlighted_data = df[start_time:end_time]
highlighted_data.loc[:, 'low-pass_filtered'] += 8 # Adjusted for the offset

# Plot highlighted_data['time'] and highlighted_data['xAccl']
plt.figure(figsize=(10, 6))
plt.plot(highlighted_data['time'], highlighted_data['low-pass_filtered'], label='xAccl', color='blue')
plt.xlabel('Time (microseconds)')
plt.ylabel('xAccl')
plt.title('Highlighted Data: Time vs xAccl')
plt.legend()
plt.grid(True)
plt.show()

#%%
# Ensure time is relative and in seconds
df_time_pd = (highlighted_data['time'] - highlighted_data['time'].iloc[0]) / 1_000_000

# Sensor properties
sensor_max = 1024
sensor_min = -1024
sensor_range = sensor_max - sensor_min

######################
# 3.1: THE SENSOR DATA IS TRANSFORMED TO RADIANS
######################
offset_radians = 0.022
df_theta_radians_pd = (highlighted_data['low-pass_filtered'] - sensor_min) * (np.pi / sensor_range) - 0.5 * np.pi - offset_radians

# --- Convert relevant data to NumPy/CuPy arrays ---
df_theta_radians = xp.asarray(df_theta_radians_pd.values, dtype=xp.float64)
df_time = xp.asarray(df_time_pd.values, dtype=xp.float64) # Keep time for plotting

# Calculate average time difference (can use pandas or numpy/cupy)
# Using pandas before converting to xp array is simpler here
average_time_diff = df_time_pd.diff().mean()
print("Average time difference between samples:", average_time_diff)

######################
# 3.3: FIND THE INITIAL CONDITIONS
######################
# Use the first value from the processed radians data
theta_init = df_theta_radians[0].item() # Get scalar value
theta_dot_init = 0.0 # Initial angular velocity assumption

print(f"Initial theta: {theta_init}, Initial theta_dot: {theta_dot_init}")

# Simulation parameters
delta_t = average_time_diff # Use the actual average time step from data
# delta_t = 0.025 # Or keep fixed if preferred
print(f"Using delta_t: {delta_t}")
sim_steps = len(df_theta_radians)

# Define the parameter ranges for the grid search
c_air_range_np = np.linspace(0.000, 0.001, 100, dtype=np.float64)  # High resolution grid
c_c_range_np = np.linspace(0.000, 0.01, 100, dtype=np.float64)   # High resolution grid
l_range_np = np.linspace(0.230, 0.235, 50, dtype=np.float64) # Single value for l

# --- Create parameter grid on GPU/CPU using xp ---
c_air_grid, c_c_grid, l_grid = xp.meshgrid(
    xp.asarray(c_air_range_np),
    xp.asarray(c_c_range_np),
    xp.asarray(l_range_np),
    indexing='ij' # Use matrix indexing ij, results c_air changes fastest
)

# Flatten the grids to get 1D arrays of all combinations
num_combinations = c_air_grid.size
c_air_flat = c_air_grid.ravel()
c_c_flat = c_c_grid.ravel()
l_flat = l_grid.ravel() # Will be constant if l_range has one element

print(f"Total parameter combinations: {num_combinations}")

# Constants
g_const = xp.asarray(hp.GRAVITY, dtype=xp.float64)
m_const = xp.asarray(hp.PENDULUM_MASS, dtype=xp.float64)


def get_theta_double_dot_vectorized(theta, theta_dot, l, m, c_air, c_c, g):
    """
    Computes the angular acceleration (theta_double_dot) for the pendulum - VECTORIZED.
    Accepts and returns CuPy/NumPy arrays.
    """
    # Ensure inputs are arrays if they aren't already matching the grid size
    # (theta and theta_dot will be arrays of size num_combinations)
    # l, m, c_air, c_c, g need to be compatible for broadcasting

    # Gravity contribution
    gravity_torque = -(g * xp.sin(theta)) / l

    # Coulomb friction
    # Note: Ensure m * l**2 is not zero
    denominator_cl = m * l**2
    # Avoid division by zero if m or l could be zero (unlikely for pendulum)
    # Add a small epsilon if necessary: denominator_cl = m * l**2 + 1e-9
    coulomb_friction = -((c_c * xp.sign(theta_dot)) / denominator_cl)

    # Air friction term
    air_friction = - (c_air * theta_dot) / denominator_cl

    # Total angular acceleration
    theta_double_dot = gravity_torque + coulomb_friction + air_friction
    
    return theta_double_dot

def simulate_potential_model_vectorized(theta_init, theta_dot_init, c_air_flat, c_c_flat, l_flat, theta_measurements, delta_t, m, g):
    """
    Simulates the pendulum for all parameter combinations in parallel.
    """
    num_combinations = len(c_air_flat)
    num_steps = len(theta_measurements)

    # Initialize state arrays for all simulations
    theta = xp.full(num_combinations, theta_init, dtype=xp.float64)
    theta_dot = xp.full(num_combinations, theta_dot_init, dtype=xp.float64)

    sum_sq_errors = xp.zeros(num_combinations, dtype=xp.float64)

    for i in range(num_steps):
        # Calculate acceleration for all combinations
        theta_double_dot = get_theta_double_dot_vectorized(theta, theta_dot, l_flat, m, c_air_flat, c_c_flat, g)

        # Update state using Euler method for all combinations
        theta_dot += theta_double_dot * delta_t
        theta += theta_dot * delta_t

        # Store result (if using sim_measurements_all)
        # sim_measurements_all[:, i] = theta

        # Accumulate squared error incrementally
        sum_sq_errors += (theta - theta_measurements[i])**2

    # Calculate RMSE from sum of squared errors
    mean_sq_error = sum_sq_errors / num_steps
    rmse_values = xp.sqrt(mean_sq_error)

    # Find the index of the minimum error
    min_error_idx = xp.argmin(rmse_values)

    # Get the minimum error and corresponding best parameters
    lowest_error = rmse_values[min_error_idx].item() # .item() moves scalar GPU->CPU
    best_c_air = c_air_flat[min_error_idx].item()
    best_c_c = c_c_flat[min_error_idx].item()
    best_l = l_flat[min_error_idx].item() # Will be the constant l

    # For plotting the best simulation later, we need its trajectory.
    # We can either re-simulate with best params, or extract if sim_measurements_all was stored.
    # Re-simulating is often easier and less memory intensive:
    best_sim_measurements,_ = run_single_simulation(theta_init, theta_dot_init, best_c_air, best_c_c, best_l, num_steps, delta_t, m, g)

    return lowest_error, (best_c_air, best_c_c, best_l), best_sim_measurements

def run_single_simulation(theta_init, theta_dot_init, c_air, c_c, l, num_steps, delta_t, m, g):
    """ Runs a single simulation, returns the theta trajectory (using NumPy/CPU is fine here) """
    theta = theta_init
    theta_dot = theta_dot_init
    sim_meas = np.zeros(num_steps)

    # Ensure parameters are scalar numpy types for this function if needed
    l_s, m_s, c_air_s, c_c_s, g_s = map(float, (l, m, c_air, c_c, g))

    sum_sq_errors = 0
    for i in range(num_steps):
         # Using a non-vectorized numpy version for single sim clarity
         # Or adapt get_theta_double_dot_vectorized to handle scalars
        gravity_torque = -(g_s * np.sin(theta)) / l_s
        denominator_cl = m_s * l_s**2
        coulomb_friction = -((c_c_s * np.sign(theta_dot)) / denominator_cl) if denominator_cl != 0 else 0
        air_friction = - (c_air_s * theta_dot) / denominator_cl if denominator_cl != 0 else 0
        theta_double_dot = gravity_torque + coulomb_friction + air_friction

        theta_dot += theta_double_dot * delta_t
        theta += theta_dot * delta_t
        sim_meas[i] = theta
        sum_sq_errors += (theta - df_theta_radians[i])**2

    mean_sq_error = sum_sq_errors / num_steps
    rmse_values = xp.sqrt(mean_sq_error)

    return sim_meas, rmse_values

# --- Run the vectorized simulation and parameter search ---
print("Starting vectorized grid search...")
import time
start_gpu_time = time.time()

lowest_error, best_params, sim_measurements_best = simulate_potential_model_vectorized(
    theta_init,
    theta_dot_init,
    c_air_flat,
    c_c_flat,
    l_flat,
    df_theta_radians, # Pass the measurements (already on GPU/CPU)
    delta_t,
    m_const,
    g_const
)

end_gpu_time = time.time()
print(f"Vectorized search finished in {end_gpu_time - start_gpu_time:.2f} seconds.")

print("Best Parameters (c_air, c_c, l):", best_params)
print("Lowest Error (RMSE):", lowest_error)


# --- Comparison and Plotting ---
# Convert data back to NumPy for plotting if it was on GPU
if xp == cp:
    df_time_np = cp.asnumpy(df_time)
    df_theta_radians_np = cp.asnumpy(df_theta_radians)
    # sim_measurements_best is already NumPy array from run_single_simulation
else:
    df_time_np = df_time
    df_theta_radians_np = df_theta_radians
    # sim_measurements_best is already NumPy array

#%%

first_estimate, error = run_single_simulation(
    theta_init,
    theta_dot_init,
    0.0004564646464646465,
    0.0071535353535353,
    0.2358244897959183,  # Use the best l found
    sim_steps,
    delta_t,
    m_const,
    g_const
)
starting = 25*40
ending = starting+5*40
print("First Estimate Error:", error)
# Plot the simulated measurements and the actual measurements
plt.figure(figsize=(12, 7))
plt.plot(df_time_np[starting:ending], df_theta_radians_np[starting:ending], label='Actual Measurements', color='blue', linewidth=1.5)
# plt.plot(df_time_np[starting:ending], sim_measurements_best[starting:ending], label=f'Simulated (Best Params: c_air={best_params[0]:.5f}, c_c={best_params[1]:.5f})', color='green', linestyle='--', linewidth=1.5)
plt.plot(df_time_np[starting:ending], first_estimate[starting:ending], label=f'Simulated (First Estimate)', color='red', linestyle='--', linewidth=1.5)
# plt.plot(df_time.iloc[window_size - 1:].reset_index(drop=True), estimated_measurements, label='Estimated Measurements', color='orange', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Theta (radians)')
plt.title('Comparison of Simulated (Best Fit) and Actual Measurements')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 7))
plt.plot(df_time_np, df_theta_radians_np, label='Actual Measurements', color='blue', linewidth=1.5)
# plt.plot(df_time_np, sim_measurements_best, label=f'Simulated (Best Params: c_air={best_params[0]:.5f}, c_c={best_params[1]:.5f})', color='green', linestyle='--', linewidth=1.5)
plt.plot(df_time_np, first_estimate, label=f'Simulated (First Estimate)', color='red', linestyle='--', linewidth=1.5)
# plt.plot(df_time.iloc[window_size - 1:].reset_index(drop=True), estimated_measurements, label='Estimated Measurements', color='orange', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Theta (radians)')
plt.title('Comparison of Simulated (Best Fit) and Actual Measurements')
plt.legend()
plt.grid(True)
plt.show()

# %%