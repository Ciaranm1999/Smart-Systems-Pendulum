#%%
import numpy as np
from Digital_twin import DigitalTwin
import pandas as pd
import matplotlib.pyplot as plt
from hyperparameters import hyperparameters
hp = hyperparameters()

# Path to the CSV file
csv_file_path = 'data_points_free_fall_40Hz.csv'
df = pd.read_csv(csv_file_path)


sample_rate = 40  # Hz
start_time = int(round(7.67 * sample_rate))
end_time = start_time + 10 * sample_rate
# start_time = int(round(70*50))
# end_time = start_time + 1 * 50

highlighted_data = df[start_time:end_time]
df_time = (highlighted_data['time'] - highlighted_data['time'].iloc[0]) /1_000_000
df_theta = highlighted_data['xAccl']
# Apply a moving average filter to the 'theta' column
window_size = 5  # Define the window size for the moving average

# Calculate the average difference between samples in df_time
average_time_diff = df_time.diff().mean()
print("Average time difference between samples:", average_time_diff)
#######################
# 3.2: THE DATA IS FILTERED
#######################
df_theta_filtered = highlighted_data['xAccl'].rolling(window=window_size).mean()

sensor_max = 1024 # Assuming this is by design
sensor_min = -1024
sensor_range = sensor_max - sensor_min

######################
# 3.1: THE SENSOR DATA IS TRANSFORMED TO RADIANS
######################

# Convert sensor values to radians
offset_radians = 0.022
df_theta_radians = (df_theta_filtered - sensor_min) * (np.pi / sensor_range) - 0.5 * np.pi - offset_radians
df_theta_radians = df_theta_radians.iloc[window_size - 1:].reset_index(drop=True)
# df_time = df_time.iloc[window_size - 1:].reset_index(drop=True)
# Plot the filtered theta values against the time column
# plt.figure(figsize=(10, 6))
# plt.plot(df_time, df_theta_radians, label='xAccl')
# plt.xlabel('Time (s)')
# plt.ylabel('xAccl')
# plt.title('xAccl vs Time')
# plt.legend()
# plt.grid()
# plt.show()

######################
# 3.3: FIND THE INITIAL CONDITIONS
######################



# digital_twin.get_theta_double_dot()

#Process dt_theta sush that it is translated to radians.

def find_initial_state(df_theta):
     # Find the initial condions of theta and theta_dot in the data
     # start = 30
     # for i in range(start, start + 10):
     #      print(df_theta.iloc[i + 1] - df_theta.iloc[i])

     theta = df_theta.iloc[0]  # Initial angle in radians
     theta_dot = 0.1
     return theta, theta_dot

#Initial conditions based on the recorded data
theta, theta_dot = find_initial_state(df_theta_radians)

#should be the same as your recording, sim_time is equal to total time of the recording

delta_t = 0.025
sim_time = 0.
digital_twin = DigitalTwin()
digital_twin.delta_t = delta_t

# Define the ranges for each parameter and the 'resolution' of the search space
c_air_range = np.linspace(0.00, 0.01, 500)  # Example range for c_air
c_c_range = np.linspace(0.0, 0.001, 500)    # Example range for c_c
l_range = [0.25]    


def get_theta_double_dot(theta, theta_dot, l, m, c_air, c_c, g=hp.GRAVITY):
        """
        Computes the angular acceleration (theta_double_dot) for the pendulum
        - Gravity
        - Coulomb friction
        - Air friction
        """
     
        
        # Gravity contribution
        gravity_torque = -(g * np.sin(theta)) / l
        
        # Coulomb friction
        coulomb_friction = -((c_c * np.sign(theta_dot)) / (m * l**2))

        # **New: Air friction term**
        air_friction = - (c_air * theta_dot) / (m * l**2)
        
        # Total angular acceleration
        theta_double_dot = gravity_torque + coulomb_friction + air_friction
        
        return theta_double_dot

def simulate_potential_model(theta, theta_dot, c_air, c_c, l, theta_measurements):
     digital_twin.c_air =c_air
     digital_twin.c_c = c_c
     digital_twin.l = l
     sim_measurements = []
     for i in range(len(theta_measurements)):
        theta_double_dot = get_theta_double_dot(theta, theta_dot, l, hp.PENDULUM_MASS, c_air, c_c)
        theta_dot += theta_double_dot * delta_t
        theta += theta_dot * delta_t
        
        sim_measurements.append(theta)
     rmse = np.sqrt(np.mean((np.array(theta_measurements) - np.array(sim_measurements))**2))
     return rmse, sim_measurements

#example usage
exp_err = 1000
# Initialize variables to store the best parameters and the lowest error found
best_params = None
lowest_error = float('inf')
index = 0

# Nested loops to go through each combination of parameters
for c_air in c_air_range:
     for c_c in c_c_range:
          for l in l_range:
               error, _ = simulate_potential_model(theta, theta_dot, c_air, c_c, l, df_theta_radians)
               if error < lowest_error:
                    lowest_error = error
                    # print(error, "found a better error")
                    best_params = (c_air, c_c, l)
     index+=1
     print(f'Finished {index * 10} iterations')

print("Best Parameters:", best_params)
print("Lowest Error:", lowest_error)
# print("expected Error: ", exp_err)

err_estimated, estimated_measurements = simulate_potential_model(theta, theta_dot, hp.AIR_FRICTION, hp.COLOUMB_FRICTION, best_params[2], df_theta_radians)
err_simulated, sim_measurements = simulate_potential_model(theta, theta_dot, best_params[0], best_params[1], best_params[2], df_theta_radians)
print("Estimated Error:", err_estimated)
print("Simulated Error:", err_simulated)
# Plot the simulated measurements and the actual measurements
plt.figure(figsize=(10, 6))
plt.plot(df_time.iloc[window_size - 1:].reset_index(drop=True), df_theta_radians, label='Actual Measurements', color='blue')
plt.plot(df_time.iloc[window_size - 1:].reset_index(drop=True), sim_measurements, label='Simulated Measurements', color='red', linestyle='--')
# plt.plot(df_time.iloc[window_size - 1:].reset_index(drop=True), estimated_measurements, label='Estimated Measurements', color='orange', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Theta (radians)')
plt.title('Comparison of Simulated and Actual Measurements')
plt.legend()
plt.grid()
plt.show()

# %%
