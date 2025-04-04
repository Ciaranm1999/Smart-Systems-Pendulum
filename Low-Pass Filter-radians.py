import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_squared_error



# Sensor conversion parameters
sensor_max = 1024
sensor_min = -1024
sensor_range = sensor_max - sensor_min
offset_radians = 0.022

# Load the data
df = pd.read_csv("data_points_free_fall_40Hz - Copy.csv")
df_sim = pd.read_csv("theta_history.csv")
df_other_filters = pd.read_csv("filtered_free_fall_output-radians.csv")


# Parameters
sampling_rate = 40  # Hz
samples_to_use = 1500

theta_history = df_sim['Theta'][:1235]
ema_f = df_other_filters['ema'][:samples_to_use]
median_f = df_other_filters['median'][:samples_to_use]
kalman_f = df_other_filters['kalman'][:samples_to_use]


# Convert to radians
raw_signal = df['xAccl'][:samples_to_use] #start from 265 to 1500 for simulation comparison
signal = ((raw_signal - sensor_min) * (np.pi / sensor_range)) - (0.5 * np.pi) - offset_radians
n = len(signal)
timestep = 1 / sampling_rate

# Detrend the signal (remove mean)
signal = signal - np.mean(signal)
signal_np = signal.to_numpy()

# FFT to find dominant frequency
fft_values = fft(signal_np)
frequencies = fftfreq(n, timestep)
positive_freqs = frequencies[:n // 2]
magnitude = np.abs(fft_values)[:n // 2]

# Find dominant frequency under 5 Hz
signal_band = positive_freqs < 5.0
dominant_freq = positive_freqs[signal_band][np.argmax(magnitude[signal_band])]
cutoff_freq = dominant_freq + 3.0  # Better 3

# Design low-pass Butterworth filter
b, a = butter(N=1, Wn=cutoff_freq / (sampling_rate / 2), btype='low') # N=1
filtered_signal = filtfilt(b, a, signal_np)

print(f"Dominant frequency detected: {dominant_freq:.2f} Hz")
print(f"Cutoff frequency for low-pass filter: {cutoff_freq:.2f} Hz")

# --- RMSE calculation ---
theta_filtered = filtered_signal[:len(theta_history)]  # Ensure both arrays are the same length

rmse = np.sqrt(mean_squared_error(theta_history, theta_filtered))
print(f"RMSE between filtered signal and simulated theta: {rmse:.6f} radians")


# # --- Plotting the results ---
# plt.figure(figsize=(12, 6))
# # plt.plot(theta_history, label='Simulated theta', linewidth=1, color='green' , linestyle='--')
# plt.plot(signal_np, label='Raw theta', linewidth=1, color='blue', linestyle='-.')
# plt.plot(filtered_signal, label='Filtered theta', linewidth=1, color='red')
# plt.title(f'Raw vs Low-Pass Filtered Theta')
# plt.xlabel('Sample Index')
# plt.ylabel('Theta (radians)')
# plt.legend()
# plt.grid(True)
# # plt.xlim(130, 400)
# plt.tight_layout()

# plt.show()

# --- Plotting the results with other filters---
plt.figure(figsize=(12, 6))
# plt.plot(theta_history, label='Simulated theta', linewidth=1, color='green' , linestyle='--')
plt.plot(signal_np, label='Raw theta', linewidth=1, color='blue', linestyle='-.')
plt.plot(filtered_signal, label='Low-Pass', linewidth=1, color='red')
plt.plot(ema_f, label='EMA', linewidth=1, color='orange')
plt.plot(median_f, label='Median', linewidth=1, color='purple')
plt.plot(kalman_f, label='Kalman', linewidth=1, color='brown')
plt.title(f'Raw vs All Filtered Theta')
plt.xlabel('Sample Index')
plt.ylabel('Theta (radians)')
plt.legend()
plt.grid(True)
plt.xlim(260, 330)
plt.tight_layout()

plt.show()




# # Add the filtered signal to the original DataFrame
# df_filtered = df.copy()
# df_filtered.loc[265:samples_to_use - 1, 'low-pass_filtered-radians'] = filtered_signal

# # Save to a new CSV file
# df_filtered.to_csv("filtered_low-pass_data-radians.csv", index=False)

# print("Filtered data saved to 'filtered_pendulum_data.csv'")

