import time
import pygame
from Digital_twin import DigitalTwin
import numpy as np
import os

# Before starting run pip install -r requirements.txt

digital_twin = DigitalTwin()

if __name__=='__main__':
        running = True
        # You can test a sequence of actions (find the action map in the digitalTwin).
        # Each action is performed after 200ms so that the actions do not overlap in time.
        # Can also use your keyboard to manually control the system.

        # actions =  [ 4, 5, 8, 2, 5, 6, 4, 8, 6, 7, 2, 6, 1, 7, 3, 7, 3, 7, 4, 7, 3, 8, 4, 7, 7, 3, 5, 4, 7, 4, 8, 7, 2, 6, 4, 8, 4, 6, 6, 3 ]
        folder_path = "./best_actions"
        file_count = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
        print(f"Number of files in '{folder_path}': {file_count}")
        actions = list(np.load(f"{folder_path}/best_actions_{file_count-1}.npy"))


        test = 0.5
        steps_time = test / digital_twin.delta_t
        # actions =  [7, 2, 0, 4, 7, 5, 3, 8, 4, 0]
        # actions=[3, 7, 2, 6, 1, 6, 2, 4, 2, 6, 2, 8, 4, 8, 4, 8, 4, 8, 4, 7]
        # actions = [8, 4, 5, 5, 3, 2, 6, 6, 7, 5, 5, 3, 7, 4, 8, 4, 8, 4, 8, 4, 8, 4, 1, 8, 1, 7, 4, 8, 4, 2, 3, 2, 8, 4, 2, 8, 1, 6, 6, 6] #-1.0815428753827656.
        # actions =  [ 4, 5, 8, 2, 5, 6, 4, 8, 6, 7, 2, 6, 1, 7, 3, 7, 3, 7, 4, 7, 3, 8, 4, 7, 7, 3, 5, 4, 7, 4, 8, 7, 2, 6, 4, 8, 4, 6, 6, 3 ]
        # 4, 7, 1, 8, 3, 7, 4, 8, 4, 8, 4, 8, 4, 8, 5, 3, 8, 4, 8, 4
        #digital_twin.connect_device()
        #digital_twin.start_recording("test_data_3")
        while running:
            # Perform action after 200ms > one step in the dt simulation is delta_t,  
            # in this case 0.005 seconds or 5ms.
            if digital_twin.steps%steps_time == 0 and len(actions) > 0:
                action = actions.pop(0)
                direction, duration = digital_twin.action_map[action]
                digital_twin.perform_action(direction, duration)
                
            #digital_twin.read_data()
            theta, theta_dot, x_pivot = digital_twin.step()
            digital_twin.render(theta, x_pivot)
            time.sleep(digital_twin.delta_t)
            
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        pygame.quit()
