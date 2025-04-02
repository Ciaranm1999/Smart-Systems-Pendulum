import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the test data
csv_file_path = 'filtered_sensor_output.csv'
data = pd.read_csv(csv_file_path)

# Convert time to relative seconds
data['time_seconds'] = data['time'] - data['time'].iloc[0]

# Pendulum parameters
m = 0.3  # Mass of the pendulum (kg)
g = 9.81  # Acceleration due to gravity (m/s^2)
l = 0.4  # Length of the pendulum (m)

# Clamp theta values to a realistic range
data['ema'] = np.clip(data['ema'], -180, 180)

# Convert theta from degrees to radians
data['theta_radians'] = np.radians(data['ema'])

# Calculate angular velocity (theta_dot) numerically
data['theta_dot'] = np.gradient(data['theta_radians']) / np.gradient(data['time_seconds'])

# Smooth theta_dot to reduce spikes
alpha_dot = 0.1  # Smoothing factor
theta_dot_smoothed = [data['theta_dot'].iloc[0]]  # Initialize with the first value
for i in range(1, len(data['theta_dot'])):
    smoothed_value = alpha_dot * data['theta_dot'].iloc[i] + (1 - alpha_dot) * theta_dot_smoothed[-1]
    theta_dot_smoothed.append(smoothed_value)
data['theta_dot'] = theta_dot_smoothed

# Calculate Potential Energy (PE)
data['PE'] = m * g * l * (1 - np.cos(data['theta_radians']))

# Calculate Kinetic Energy (KE)
data['KE'] = 0.5 * m * (l * data['theta_dot'])**2

# Calculate Total Energy (TE)
data['TE'] = data['PE'] + data['KE']
print(data['theta_radians'].describe())

# Replot angular velocity
plt.figure(figsize=(12, 6))
plt.plot(data['time_seconds'], data['theta_dot'], label='Angular Velocity (theta_dot)', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.title('Angular Velocity Over Time')
plt.legend()
plt.grid(True)
plt.show()


# Use maximum total energy as reference for relative error
te_reference = data['TE'].max()
data['TE_relative_error'] = (data['TE'] - te_reference) / te_reference

# Smooth the relative error
alpha_error = 0.1  # Smoothing factor
relative_error_smoothed = [data['TE_relative_error'].iloc[0]]
for i in range(1, len(data['TE_relative_error'])):
    smoothed_value = alpha_error * data['TE_relative_error'].iloc[i] + (1 - alpha_error) * relative_error_smoothed[-1]
    relative_error_smoothed.append(smoothed_value)
data['TE_relative_error'] = relative_error_smoothed


# Define an exponential decay function
def exponential_decay(t, a, b, c):
    return a * np.exp(-b * t) + c

# Filter steady-state data (t > 2)
steady_state_data = data[data['time_seconds'] > 2]

# Provide initial guesses for the parameters
initial_guesses = [steady_state_data['TE'].max() - steady_state_data['TE'].min(), 0.1, steady_state_data['TE'].min()]

# Fit the decay curve with initial guesses and increased maxfev
popt, _ = curve_fit(exponential_decay, steady_state_data['time_seconds'], steady_state_data['TE'], p0=initial_guesses, maxfev=5000)

# Print the fitted parameters
a, b, c = popt
print(f"Exponential Decay Parameters: a={a}, b={b}, c={c}")


# Plot KE, PE, and TE together
plt.figure(figsize=(12, 6))
plt.plot(data['time_seconds'], data['KE'], label='Kinetic Energy (KE)', color='blue')
plt.plot(data['time_seconds'], data['PE'], label='Potential Energy (PE)', color='red')
plt.plot(data['time_seconds'], data['TE'], label='Total Energy (TE)', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('Kinetic, Potential, and Total Energy Over Time')
plt.legend()
plt.grid(True)
plt.show()

alpha_dot = 0.05  # Reduce the smoothing factor for stronger smoothing
theta_dot_smoothed = [data['theta_dot'].iloc[0]]
for i in range(1, len(data['theta_dot'])):
    smoothed_value = alpha_dot * data['theta_dot'].iloc[i] + (1 - alpha_dot) * theta_dot_smoothed[-1]
    theta_dot_smoothed.append(smoothed_value)
data['theta_dot'] = theta_dot_smoothed

steady_state_data = data[data['time_seconds'] > 2]
plt.figure(figsize=(12, 6))
plt.plot(steady_state_data['time_seconds'], steady_state_data['KE'], label='Kinetic Energy (KE)', color='blue')
plt.plot(steady_state_data['time_seconds'], steady_state_data['PE'], label='Potential Energy (PE)', color='red')
plt.plot(steady_state_data['time_seconds'], steady_state_data['TE'], label='Total Energy (TE)', color='green')
plt.plot(data['time_seconds'], exponential_decay(data['time_seconds'], *popt), label='Fitted Decay Curve', color='orange', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('Kinetic, Potential, and Total Energy Over Time (t > 2)')
plt.legend()
plt.grid(True)
plt.show()
