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
signal = df['xAccl'][:samples_to_use]
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
cutoff_freq = dominant_freq + 1.0  # Better 3

# Design low-pass Butterworth filter
b, a = butter(N=4, Wn=cutoff_freq / (sampling_rate / 2), btype='low') # N=1
filtered_signal = filtfilt(b, a, signal_np)

# Plot the result
plt.figure(figsize=(12, 6))
plt.plot(signal_np[0:500], label='Raw xAccl')
plt.plot(filtered_signal[0:500], label='Filtered xAccl (Low-pass)', linewidth=2)
plt.title(f'Low-pass Filter Applied (Cutoff = {cutoff_freq:.2f} Hz)')
plt.xlabel('Sample Index')
plt.ylabel('Acceleration')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: print the detected dominant frequency
print(f"Dominant frequency detected: {dominant_freq:.2f} Hz")


# Add the filtered signal to the original DataFrame
df_filtered = df.copy()
df_filtered.loc[:samples_to_use - 1, 'low-pass_filtered'] = filtered_signal

# Save to a new CSV file
df_filtered.to_csv("filtered_low-pass_data.csv", index=False)

print("Filtered data saved to 'filtered_pendulum_data.csv'")
