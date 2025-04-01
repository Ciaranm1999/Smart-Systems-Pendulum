import time
import pygame
from Digital_twin import DigitalTwin
import matplotlib.pyplot as plt
# Before starting run pip install -r requirements.txt

digital_twin = DigitalTwin()
theta_history = []
if __name__=='__main__':
    running = True
    while running:
        
        theta, theta_dot, x_pivot = digital_twin.step()
        theta_history.append(theta)
        # print(theta_dot)
        digital_twin.render(theta, x_pivot)
        # time.sleep(digital_twin.delta_t)
        
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
    pygame.quit()
    
    plt.figure(figsize=(10, 6))
    plt.plot(theta_history)
    plt.xlabel('Time step')
    plt.ylabel('Theta (rad)')
    plt.title('Pendulum Angle Over Time')
    plt.grid()
    plt.show()