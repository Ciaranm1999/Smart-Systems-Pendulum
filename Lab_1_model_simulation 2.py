import time
import pygame
from Digital_twin import DigitalTwin
import csv
import pandas as pd
import matplotlib.pyplot as plt

# Clear the contents of the recording.csv file
with open('recording.csv', mode='w', newline='') as file:
    file.truncate()

digital_twin = DigitalTwin()

if __name__ == '__main__':
    running = True
    while running:
        # Perform a simulation step
        theta, theta_dot, x_pivot = digital_twin.step()

        # Calculate energies
        PE, KE, TE = digital_twin.calculate_energies()

        # Render the pendulum
        digital_twin.render(theta, x_pivot)
        time.sleep(digital_twin.delta_t)

        # Save the theta, theta_dot, x_pivot, and energies to the CSV file
        with open('recording.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([time.time(), theta, theta_dot, x_pivot, PE, KE, TE])

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in digital_twin.actions:
                    direction, duration = digital_twin.actions[event.key]
                    digital_twin.perform_action(direction, duration)
                elif event.key == pygame.K_r:
                    digital_twin = DigitalTwin()  # Restart the system
                    print("System restarted")
                elif event.key == pygame.K_ESCAPE:
                    running = False  # Quit the simulation

    pygame.quit()

# Plot the data
# Load the data from the CSV file
data = pd.read_csv('recording.csv', header=None, names=['time', 'theta', 'theta_dot', 'x_pivot', 'PE', 'KE', 'TE'])

# Create figure and axis for energies
fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot Potential Energy (PE)
ax1.plot(data['time'], data['PE'], label='Potential Energy (PE)', color='blue')

# Plot Kinetic Energy (KE)
ax1.plot(data['time'], data['KE'], label='Kinetic Energy (KE)', color='red')

# Plot Total Energy (TE)
ax1.plot(data['time'], data['TE'], label='Total Energy (TE)', color='green', linestyle='--')

# Labels and title
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Energy (J)')
plt.title('Energy of the Pendulum Over Time')
ax1.legend()
ax1.grid(True)

# Show plot
plt.show()