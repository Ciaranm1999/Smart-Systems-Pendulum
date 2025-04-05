#%%
import pandas as pd
from hyperparameters import hyperparameters
from Digital_twin import DigitalTwin
import numpy as np
hp = hyperparameters()

# Load the CSV file
file_path = 'data_points_movement_40Hz.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the data
print(data.head())
import matplotlib.pyplot as plt
data['adjustedtime'] = (data['time'] - data['time'].min())/1_000_000
# Plot the newPosition column against the time column
data['realPosition'] = data['newPosition'] * hp.TRACK_LENGTH / hp.range_encoder
data['realPosition'] -= data['realPosition'].min()
data['velocity'] = data['realPosition'].diff() / data['adjustedtime'].diff()
data['acceleration'] = data['velocity'].diff() / data['adjustedtime'].diff()
# plt.plot(data['adjustedtime'], data['acceleration'], label='acceleration')
# plt.plot(data['adjustedtime'], data['realPosition'] * 100, label='position')
# plt.xlabel('Time')
# plt.ylabel('New Position')
# plt.title('New Position vs Time')
# plt.legend()
# plt.grid()
# plt.show()

real_pos_200 = data.loc[(data['adjustedtime'] - 3).abs().idxmin(), 'realPosition']
real_pos_150 = data.loc[(data['adjustedtime'] - 5).abs().idxmin(), 'realPosition']
real_pos_100 = data.loc[(data['adjustedtime'] - 7).abs().idxmin(), 'realPosition']
real_pos_50 = data.loc[(data['adjustedtime'] - 9).abs().idxmin(), 'realPosition']

def get_positions(relative_50, relative_100, relative_150, relative_200):
    out_50 =np.abs(relative_50 - relative_100)
    out_100 =np.abs(relative_100 - relative_150)
    out_150 =np.abs(relative_150 - relative_200)

    return out_50, out_100, out_150, relative_200

move_50, move_100, move_150, move_200 = get_positions(real_pos_50, real_pos_100, real_pos_150, real_pos_200)

filtered_data = data[(data['adjustedtime'] >= 2) & (data['adjustedtime'] <= 2.4)]

digital_twin = DigitalTwin()

def check_parameters(digital_twin, move_50, move_100, move_150, move_200):
    times = [0.2, 0.15, 0.1, 0.05]
    
    reals = []
    for time in times:
        digital_twin.update_motor_accelerations_2('right', time)
        reals.append(np.sum(digital_twin.future_motor_positions))
        digital_twin.future_motor_positions = []

    real_50, real_100, real_150, real_200 = reals[::-1]

    relative_50, relative_100, relative_150, relative_200 = get_positions(real_50, real_100, real_150, real_200)
    error = (move_50 - relative_50) ** 2 + (move_100 - relative_100) ** 2 + (move_150 - relative_150) ** 2 + (move_200 - relative_200) ** 2
    return error

Ke_range = np.linspace(0.03 , 0.07, 50)       #0.02625
J_range = np.linspace(1e-3 , 2e-4, 50) # 9e-5
B_range = np.linspace(1e-7 , 1e-5, 50) # 1e-6

MOTOR_R = 7  # Resistance in ohms
MOTOR_K_t = 0.00375 * MOTOR_R  /2# Torque constant in Nm/A
MOTOR_K_e = 0.00375 * MOTOR_R  /2# Back EMF constant in V/(rad/s)
MOTOR_J = 9e-5  # Rotor inertia in kg.m^2
MOTOR_B = 1e-6  # Viscous friction coefficient in Nm.s/rad

error = check_parameters(digital_twin, move_50, move_100, move_150, move_200)
print(f"Initial Error: {error}")

lowest_error = float('inf')
best_params = None
for Ke in Ke_range:
    for J_value in J_range:
        for B_value in B_range:
            digital_twin.K_e = Ke
            digital_twin.K_t = Ke
            digital_twin.J = J_value
            digital_twin.B = B_value

            error = check_parameters(digital_twin, move_50, move_100, move_150, move_200)
            if error < lowest_error:
                lowest_error = error
                best_params = (Ke, J_value, B_value)

print("Best Parameters:")
print("K_e:", best_params[0])
print("J:", best_params[1])
print("B:", best_params[2])
print("Lowest Error:", lowest_error)
