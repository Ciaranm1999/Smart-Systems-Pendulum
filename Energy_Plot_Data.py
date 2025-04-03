import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the test data
csv_file_path = 'filtered_free_fall_output.csv'
data = pd.read_csv(csv_file_path)

# Convert time to relative seconds and scale to seconds
data['time_seconds'] = (data['time'] - data['time'].iloc[0]) / 1e6

# Pendulum parameters
m = 0.3  # Mass of the pendulum (kg)
g = 9.81  # Acceleration due to gravity (m/s^2)
l = 0.24  # Length of the pendulum (m)

# Convert theta from degrees to radians and normalize
data['theta_radians'] = np.radians(data['ema'])
data['theta_radians'] = data['theta_radians'] / data['theta_radians'].max() * np.pi

# Calculate angular velocity (theta_dot) numerically
data['theta_dot'] = np.gradient(data['theta_radians']) / np.gradient(data['time_seconds'])

# Calculate Potential Energy (PE)
data['PE'] = m * g * l * (1 - np.cos(data['theta_radians']))

# Calculate Kinetic Energy (KE)
data['KE'] = 0.5 * m * (l * data['theta_dot'])**2

# Calculate Total Energy (TE)
data['TE'] = data['PE'] + data['KE']

# Define an exponential decay function
def exponential_decay(t, a, b, c):
    return a * np.exp(-b * t) + c

# Filter steady-state data (t > 6)
steady_state_data = data[data['time_seconds'] > 6]

# Provide initial guesses for the parameters
initial_guesses = [steady_state_data['TE'].max() - steady_state_data['TE'].min(), 0.1, steady_state_data['TE'].min()]

# Filter data to start at time_seconds > 6
filtered_data = data[data['time_seconds'] > 6]

# Plot Angular Displacement Over Time (starting at time > 6)
plt.figure(figsize=(12, 6))
plt.plot(filtered_data['time_seconds'], filtered_data['theta_radians'], label='Angular Displacement (theta_radians)', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Angular Displacement (rad)')
plt.title('Angular Displacement Over Time (t > 6)')
plt.legend()
plt.grid(True)
plt.show()

# Plot PE, KE, and TE Over Time (starting at time > 6)
plt.figure(figsize=(12, 6))
plt.plot(filtered_data['time_seconds'], filtered_data['KE'], label='Kinetic Energy (KE)', color='blue')
plt.plot(filtered_data['time_seconds'], filtered_data['PE'], label='Potential Energy (PE)', color='red')
plt.plot(filtered_data['time_seconds'], filtered_data['TE'], label='Total Energy (TE)', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('Kinetic, Potential, and Total Energy Over Time (t > 6)')
plt.legend()
plt.grid(True)
plt.show()

# Fit the exponential decay function to the Total Energy (TE)
popt, pcov = curve_fit(exponential_decay, steady_state_data['time_seconds'], steady_state_data['TE'], p0=initial_guesses)

# Extract the fitted parameters
a, b, c = popt
print(f"Fitted parameters: a={a}, b={b}, c={c}")

# Plot the Total Energy with the fitted decay curve
plt.figure(figsize=(12, 6))
plt.plot(steady_state_data['time_seconds'], steady_state_data['TE'], label='Total Energy (TE)', color='green')
plt.plot(steady_state_data['time_seconds'], exponential_decay(steady_state_data['time_seconds'], *popt), label='Fitted Decay Curve', color='orange', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('Total Energy with Fitted Exponential Decay Curve (t > 6)')
plt.legend()
plt.grid(True)
plt.show()

# Find turning points where KE is near zero
turning_points = steady_state_data[steady_state_data['KE'] < 1e-3]

# Calculate energy losses at turning points
energy_losses = turning_points['TE'].diff().abs()
average_coulomb_loss = energy_losses.mean()
print(f"Estimated average energy loss due to Coulomb friction: {average_coulomb_loss:.4f} J")
print(f"Air Friction = {b}")

# Calculate total energy dissipated by air friction
time_interval = steady_state_data['time_seconds'].iloc[-1] - steady_state_data['time_seconds'].iloc[0]
energy_dissipated_air = a * (1 - np.exp(-b * time_interval))

# Calculate total energy dissipated by Coulomb friction
num_turning_points = len(turning_points)
energy_dissipated_coulomb = average_coulomb_loss * num_turning_points

# Calculate total energy dissipated by friction
total_energy_dissipated = energy_dissipated_air + energy_dissipated_coulomb

# Print energy dissipation results
print(f"Energy dissipated by air friction: {energy_dissipated_air:.4f} J")
print(f"Energy dissipated by Coulomb friction: {energy_dissipated_coulomb:.4f} J")
print(f"Total energy dissipated by friction: {total_energy_dissipated:.4f} J")

# Calculate the coefficient of air friction (gamma)
coefficient_air_friction = b * m
print(f"Coefficient of air friction (gamma): {coefficient_air_friction:.4f} kg/s")

# Calculate the coefficient of Coulomb friction (mu)
coefficient_coulomb_friction = average_coulomb_loss / (m * g * l)
print(f"Coefficient of Coulomb friction (mu): {coefficient_coulomb_friction:.4f}")