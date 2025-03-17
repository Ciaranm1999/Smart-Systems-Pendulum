import time
import pygame
from Digital_twin import DigitalTwin
import csv
from reinforcement import PendulumRlAgent
import numpy as np
# Before starting run pip install -r requirements.txt

# Clear the contents of the recording.csv file
with open('recording.csv', mode='w', newline='') as file:
    file.truncate()
    names=['time', 'theta', 'theta_dot', 'theta_double_dot', 'x_pivot']
    writer = csv.writer(file)
    writer.writerow(names)

digital_twin = DigitalTwin()
sample_time = 0.2 # seconds

history_buffer = []
replay_buffer = []

render = True
max_run_time = 60.0 # seconds
action = ['left', 0]
Agent = PendulumRlAgent()

if __name__=='__main__':
        running = True
        time_in_current_state = 0.
        run_time = 0.
        while running:
            run_time += digital_twin.delta_t
            time_in_current_state += digital_twin.delta_t
            theta, theta_dot, theta_double_dot, x_pivot = digital_twin.step()

            # the history buffer will never get really big, unless we run for hours.
            history_buffer.append([run_time, theta, theta_dot, theta_double_dot, x_pivot])
            
            if render:
                digital_twin.render(theta, x_pivot)
                time.sleep(digital_twin.delta_t)

            if time_in_current_state >= sample_time:
                time_in_current_state = 0.
                # code to call the RL agent to get the action
                action = Agent.predict(theta, theta_dot, theta_double_dot, x_pivot)
                # digital_twin.perform_action(action[0], action[1])


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
                        running = False # Quit the simulation

        with open('recording.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(history_buffer)
            history_buffer = []

        pygame.quit()


def reward_calculation(postion, angle):
    # Reward function
    reward = -abs(angle - np.pi)
    return reward
    pass

# Plot the data
# Load the data from the CSV file
#%%
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('recording.csv')

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(10, 5))

# First y-axis for x_pivot
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('x_pivot', color='blue')
ax1.plot(data['time'], data['x_pivot'], label='x_pivot', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Second y-axis for theta_dot
ax2 = ax1.twinx()
ax2.set_ylabel('theta_dot', color='red')
ax2.plot(data['time'], data['theta_dot'], label='theta_dot', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Title and Grid
plt.title('x_pivot and theta_dot over Time')
ax1.grid(True)

# Legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Show plot
plt.show()

# %%
