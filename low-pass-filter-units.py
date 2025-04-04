import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt

# Load the data
df = pd.read_csv("data_points_free_fall_40Hz - Copy.csv")

# Parameters
sampling_rate = 40  # Hz
samples_to_use = 1500
signal = df['xAccl'][265:samples_to_use]
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

# Plot the result
plt.figure(figsize=(12, 6))
plt.plot(signal_np, label='Raw theta', linewidth=1)
plt.plot(filtered_signal, label='Filtered theta', linewidth=1.5, linestyle='--')
plt.title(f'Low-pass Filter Zoomed In (First 500 Samples)')
plt.xlabel('Sample Index')
plt.ylabel('Acceleration')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# Add the filtered signal to the original DataFrame
df_filtered = df.copy()
df_filtered.loc[265:samples_to_use - 1, 'low-pass_filtered'] = filtered_signal

# Save to a new CSV file
df_filtered.to_csv("filtered_low-pass_data-units.csv", index=False)

print("Filtered data saved to 'filtered_pendulum_data.csv'")