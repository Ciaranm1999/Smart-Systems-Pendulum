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

        actions = list(np.load("./best_action-20k.npy"))

        test = 0.15
        steps_time = test / digital_twin.delta_t

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
        
