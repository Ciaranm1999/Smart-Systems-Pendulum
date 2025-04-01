#%%
import numpy as np
import matplotlib.pyplot as plt
class test_theta_double_dot:
    def __init__(self, scale=1.0):
        self.g = 9.81 * scale  # Acceleration due to gravity (m/s^2)
        self.l = 0.4   # Length of the pendulum (m)
        self.c_air = 0.001  # Air friction coefficient
        self.c_c = 0.01 * scale   # Coulomb friction coefficient
        self.a_m = 0.0001 # Motor acceleration force tranfer coefficient
        self.m = 0.3 # Mass of the pendulum
        self.currentmotor_acceleration = 0.0

    def get_theta_double_dot(self, theta, theta_dot):
            """
            Computes the angular acceleration (theta_double_dot) for the pendulum
            considering:
            - Motor acceleration
            - Gravity
            - Coulomb friction
            - Air friction
            """

            # Motor-driven acceleration term
            motor_torque = -(1 / self.l) * self.currentmotor_acceleration * np.cos(theta)
            
            # Gravity contribution
            gravity_torque = -(self.g * np.sin(theta)) / self.l
            
            # Coulomb friction
            coulomb_friction = -((self.c_c * np.sign(theta_dot)) / (self.m * self.l**2))

            # **New: Air friction term**
            air_friction = - (self.c_air * theta_dot) / (self.m * self.l**2)
            
            # Total angular acceleration
            theta_double_dot = motor_torque + gravity_torque + coulomb_friction + air_friction
            
            return theta_double_dot
    
delta_t = 0.005  # Time step for simulation
time_t = 3  # Total time for simulation
time_array = np.arange(0, time_t, delta_t)  # Time array for simulation

theta = np.pi / 2  # Initial angle (radians)
theta_dot = 0.0  # Initial angular velocity (radians/s)
tdd_history = []  # History of theta_double_dot values
theta_history = []  # History of theta values
for t in time_array:
    # Update the motor acceleration (this should be replaced with actual motor control logic)
    
    # Create an instance of the test class
    pendulum_test = test_theta_double_dot()
    
    # Call the method to get theta_double_dot
    theta_double_dot = pendulum_test.get_theta_double_dot(theta, theta_dot)
    tdd_history.append(theta_double_dot)
    
    
    theta += theta_dot * delta_t  # Update angle
    theta_history.append(theta)
    theta_dot += theta_double_dot * delta_t  # Update angular velocity
# Plot theta history over time
plt.figure(figsize=(10, 6))
plt.plot(time_array, theta_history, label='Theta (rad)')
plt.xlabel('Time (s)')
plt.ylabel('Theta (radians)')
plt.title('Pendulum Angle Over Time')
plt.legend()
plt.grid()
plt.show()


