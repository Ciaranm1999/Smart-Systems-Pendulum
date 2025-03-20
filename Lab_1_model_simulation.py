import time
import pygame
from Digital_twin import DigitalTwin
import csv
from reinforcement import PendulumRlAgent
import numpy as np
from hyperparameters import hyperparameters
import os
# Before starting run pip install -r requirements.txt
hp = hyperparameters()
# Clear the contents of the recording.csv file
os.makedirs('./training_sessions/session_1', exist_ok=True)
with open('./recordings/recording.csv', mode='w', newline='') as file:
    file.truncate()
    names=['time', 'theta', 'theta_dot', 'theta_double_dot', 'x_pivot']
    writer = csv.writer(file)
    writer.writerow(names)

with open('./replays/replay.csv', mode='w', newline='') as file:
    file.truncate()
    names=['time', 'theta', 'theta_dot', 'theta_double_dot', 'x_pivot', 'action', 'reward', 'next_theta', 'next_theta_dot', 'next_theta_double_dot', 'next_x_pivot']
    writer = csv.writer(file)
    writer.writerow(names)

digital_twin = DigitalTwin()
sample_time = hp.SAMPLE_TIME

history_buffer = []
replay_buffer = []

render = hp.RENDER
print_output = hp.PRINT_OUTPUT

max_run_time = hp.MAX_RUN_TIME
translated_action = ['left', 0]
action_index = 0 

load_model_number = -1
if load_model_number > 0:

    agent = PendulumRlAgent(load_model_number)
else:
    agent = PendulumRlAgent()

epoch = 0
epochs_max = hp.EPOCHS_MAX
mean_reward = 0

start_time = time.time()



if __name__=='__main__':
    print("press 'l' to toggle rendering on/off")
    print("press 'p' to toggle print output on/off")
    print("press 'esc' to quit")
    while epoch < epochs_max:
        epoch += 1
        digital_twin = DigitalTwin()
        running = True
        time_in_current_state = 0.
        run_time = 0.
        state_rewards = 0.

        previous_state = []
        while running and run_time < max_run_time:
            run_time += digital_twin.delta_t
            time_in_current_state += digital_twin.delta_t
            theta, theta_dot, theta_double_dot, x_pivot = digital_twin.step()
            theta_corrected = theta%(2*np.pi)
            state_rewards += agent.reward_calculation(x_pivot, theta_corrected)
            
            # the history buffer will never get really big, unless we run for hours.
            # history_buffer.append([run_time, theta_corrected, theta_dot, theta_double_dot, x_pivot])
            
            if render:
                digital_twin.render(theta_corrected, x_pivot)
                # time.sleep(digital_twin.delta_t)

            if time_in_current_state >= sample_time:
                time_in_current_state = 0.
                current_state = [theta_corrected, theta_dot, theta_double_dot]
                if len(previous_state) == 0:
                     previous_state = current_state
                buffer_entry = previous_state + [action_index, state_rewards] + current_state
                agent.replay_buffer.push(*buffer_entry)
                # replay_buffer.append([run_time] +buffer_entry)
                # code to call the RL agent to get the action
                translated_action, action_index = agent.predict(*current_state)
                digital_twin.perform_action(translated_action[0], translated_action[1])
                sample_time = translated_action[1]/1000
                previous_state = current_state
                state_rewards = 0.

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    # if event.key in digital_twin.actions:
                    #     direction, duration = digital_twin.actions[event.key]
                    #     digital_twin.perform_action(direction, duration)
                    # elif event.key == pygame.K_r:
                    #         digital_twin = DigitalTwin()  # Restart the system
                    #         print("System restarted")
                    if event.key == pygame.K_ESCAPE:
                        running = False # Quit the simulation
                    elif event.key == pygame.K_l:
                        render = not render
                    elif event.key == pygame.K_p:
                        print_output = not print_output
        # with open(f'./recordings/recording_{epoch}.csv', mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(history_buffer)
        #     history_buffer = []

        # with open(f'./replays/replay_{epoch}.csv', mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(replay_buffer)
        #     replay_buffer = []
        temp_print = print_output
        if temp_print ==  False:
            if epoch % 100 == 0:
                temp_print = True

        mean_reward = agent.train_faster(temp_print)
        if temp_print:
            print(f"Epoch {epoch} completed")

    pygame.quit()
    
print("Training completed")
# Save the training session details to a file
# Get the number of files in the ./training_sessions directory
training_sessions_dir = './training_sessions'
num_files = len([name for name in os.listdir(training_sessions_dir) if os.path.isfile(os.path.join(training_sessions_dir, name))])
training_time = time.time() - start_time
with open(f'./training_sessions/session_{num_files}/hyperparameters.txt', mode='a') as session_file:
    session_file.write(f"Training Session Summary:\n")
    session_file.write(f'Number of epochs: {epoch}\n')
    session_file.write(f"Hyper parameters: {hp.to_string()}\n")
    session_file.write(f"Mean reward: {mean_reward}\n")
    session_file.write(f"Training time: {training_time}\n")

agent.save_model()
print("Model saved")
# Plot the data
# Load the data from the CSV file
#%%
# import pandas as pd
# import matplotlib.pyplot as plt

# # Load data
# data = pd.read_csv('recording.csv')

# # Create figure and axis
# fig, ax1 = plt.subplots(figsize=(10, 5))

# # First y-axis for x_pivot
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('x_pivot', color='blue')
# ax1.plot(data['time'], data['x_pivot'], label='x_pivot', color='blue')
# ax1.tick_params(axis='y', labelcolor='blue')

# # Second y-axis for theta_dot
# ax2 = ax1.twinx()
# ax2.set_ylabel('theta_dot', color='red')
# ax2.plot(data['time'], data['theta_dot'], label='theta_dot', color='red')
# ax2.tick_params(axis='y', labelcolor='red')

# # Title and Grid
# plt.title('x_pivot and theta_dot over Time')
# ax1.grid(True)

# # Legends
# ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')

# # Show plot
# plt.show()

# # %%
# buffer = []
# previous_state = []
# val_1 = 1
# val_2 = 2
# val_3 = 3
# val_4 = 4
# previous_state = [val_1, val_2, val_3, val_4]
# print(previous_state)
# extra_value = 5
# buffer_entry = previous_state + [extra_value]
# buffer.append(buffer_entry)
# print(buffer)


# %%
