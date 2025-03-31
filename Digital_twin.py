import pygame
import scipy.integrate
import serial
import numpy as np
import csv
import math
import scipy.integrate as it
import time
import pandas as pd
from hyperparameters import hyperparameters
hp = hyperparameters()

class DigitalTwin:    
    def __init__(self, scale=1.0):
        # Initialize Pygame parameters
        self.screen = None
        # Initialize serial communication parameters
        self.ser = None
        self.device_connected = False
        # State configuration parameters
        self.steps = 0
        self.theta = 0.5 * np.pi-0.01
        self.theta_dot = 0.
        self.theta_double_dot = 0.
        self.x_pivot = 0
        self.delta_t = 0.005  # Example value, adjust as needed in seconds 19.42879347
        
        # Model parameters
        self.g = 9.81 * scale  # Acceleration due to gravity (m/s^2)
        self.l = 0.4   # Length of the pendulum (m)
        self.c_air = 0.001  # Air friction coefficient
        self.c_c = 0.02 * scale   # Coulomb friction coefficient
        self.a_m = 0.0001 # Motor acceleration force tranfer coefficient
        self.m = 0.3 # Mass of the pendulum
        self.future_motor_accelerations = []
        self.future_motor_positions = []
        self.currentmotor_acceleration = 0.
        self.time = 0.
        # Sensor data
        self.sensor_theta = 0
        self.current_sensor_motor_position = 0.
        self.current_action = 0

        # render precalculations
        self.pendulum_length = (hp.PENDULUM_LENGHT * hp.TRACK_DRAW_LENGTH) / hp.TRACK_LENGTH
        self.track_x_start = 500 - int(hp.TRACK_DRAW_LENGTH / 2)
        self.track_x_end = 500 + int(hp.TRACK_DRAW_LENGTH / 2)
        self.x_pivot_multiplier = hp.TRACK_DRAW_LENGTH / hp.TRACK_LENGTH
        # Keyboard action mappings
        _action_durations = [200, 150, 100, 50]  # Durations in milliseconds
        _keys_left = [pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_f]
        _keys_right = [pygame.K_SEMICOLON, pygame.K_l, pygame.K_k , pygame.K_j]
        self.actions = {key: ('left', duration) for key, duration in zip(_keys_left, _action_durations)}
        self.actions.update({key: ('right', duration) for key, duration in zip(_keys_right, _action_durations)})
        self.action_map = [
            ('left', 0),  # No action
            ('left', 50), ('left', 100), ('left', 150), ('left', 200),
            ('right', 50), ('right', 100), ('right', 150), ('right', 200)
        ]
        self.recording = False
        self.writer = None
        self.start_time = 0.
        self.df = None

        # Parameters that have to do with the new motor model
        self.R = hp.MOTOR_R  # Armature Resistance (Ohms)
        self.K_t = hp.MOTOR_K_t  # Torque Constant (Nm/A)
        self.K_e = hp.MOTOR_K_e  # Back EMF Constant (V/(rad/s)) - Often numerically equal to K_t in SI
        self.J = hp.MOTOR_J  # Rotor Inertia (kg.m^2)
        self.b = hp.MOTOR_B  # Viscous Friction Coefficient (Nm.s/rad)
        self.pully_circumference = hp.PULLY_RADIUS * 2 * np.pi  # Circumference of the pulley (m)
        self.pully_radius = hp.PULLY_RADIUS  # Circumference of the pulley (m)
        self.sample_time = hp.SAMPLE_TIME # seconds
        self.current_omega = 0.0  # Initial angular velocity (rad/s)
        self.motor_voltage = 0.0  # Initial motor voltage (V)
        # Initialize a pygame window
        self.initialize_pygame_window()

    def initialize_pygame_window(self):
        # Set up the drawing window
        pygame.init()
        self.screen = pygame.display.set_mode([1000, 800])
        self.font = pygame.font.Font(None, 36)  # Initialize font (36 is the font size)
        self.clock = pygame.time.Clock()  # Create a clock object to track FPS

    def connect_device(self, port='COM3', baudrate=115200):
        # Establish a serial connection for sensor data
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=0, writeTimeout=0)
        self.device_connected = True
        print("Connected to: " + self.ser.portstr)

    def read_data(self):
        line = self.ser.readline()
        line = line.decode("utf-8")
        try:
            if len(line) > 2 and line != '-':
                sensor_data = line.split(",")
                if len(sensor_data[0]) > 0 and len(sensor_data[3]) > 0:
                    self.sensor_theta = int(sensor_data[0])
                    self.current_sensor_motor_position = -int(sensor_data[3])
        except Exception as e:
            print(e)
        if self.recording:
            self.writer.writerow([round(time.time() * 1000)-self.start_time, self.sensor_theta, self.current_sensor_motor_position])


    def process_data(self):
        """
        Lab 2: Use the sensor data retured by the function read_data. 
        The sensor data needs to be represented in the virtual model.
        First the data should be scaled and calibrated,
        Secondly noise should be reduced trough a filtering method.
        Return the processed data sush that it can be used in visualization and recording.
        Also, transform the current_sensor_motor_position to be acurate. 
        This means that the encoder value should be scaled to match the displacement in the virtual model.
        """
        self.sensor_theta = 0
        self.current_sensor_motor_position = 0
        
    def start_recording(self, name):
        # If you are working on the bonus assignments then you should also add a columb for actions (and safe those).
        self.recording = True
        self.file = open('{}.csv'.format(name), 'w', newline='')  
        self.writer = csv.writer(self.file)
        self.start_time = round(time.time() * 1000)
        self.writer.writerow(["time", "theta", "x_pivot"])

    def stop_recording(self):
        self.recording = False
        self.file.close()
    
    def load_recording(self, name):
        self.df = pd.read_csv('{}.csv'.format(name))
        print("recording is loaded")
    
    def recorded_step(self,i):
        a = self.df["time"].pop(i)
        b = self.df["theta"].pop(i)
        c = self.df["x_pivot"].pop(i)  
        return a, b, c

    def perform_action(self, direction, duration):
        # Send the command to the device.
        # if self.device_connected:
        if hp.OLD_METHOD:
            if duration > 0:
                self.update_motor_accelerations(direction, duration/1000)
        else:
            if direction == 'left':
                d = -duration
            else:
                d = duration
                # self.ser.write(str(d).encode())
            #if abs(duration) > 0:
            self.motor_voltage = d * 0.005 * 12 # Scale to 0-12V. *0.005 is a scaling factor to convert the duration to a voltage level.


    # --- Corrected Motor Model Function ---
    # Based on τ_m = (K_t/R)*V - (K_t*K_e/R)*ω
    # and dω/dt = (1/J) * (τ_m - b*ω)
    def motor_model_dynamics(self, voltage, omega, R, K_t, K_e, J, b):
        """
        Calculates the angular acceleration (dω/dt) of the DC motor.

        Args:
            voltage (float): Current input voltage (V).
            omega (float): Current angular velocity (rad/s).
            R (float): Armature Resistance (Ohms).
            K_t (float): Torque Constant (Nm/A).
            K_e (float): Back EMF Constant (V/(rad/s)).
            J (float): Rotor Inertia (kg.m^2).
            b (float): Viscous Friction Coefficient (Nm.s/rad).

        Returns:
            float: Angular acceleration (dω/dt) in rad/s^2.
        """
        # Calculate current based on V, omega, K_e, R (can skip if not needed elsewhere)
        # current_I = (voltage - K_e * omega) / R
        # Calculate motor torque (τ_m)
        # tau_m = K_t * current_I
        # Combined torque calculation:
        tau_m = (K_t / R) * voltage - (K_t * K_e / R) * omega

        # Calculate angular acceleration (dω/dt)
        domega_dt = (1.0 / J) * (tau_m - b * omega)

        return domega_dt 
    
    def update_motor_accelerations_2(self, voltage):
        # Calculate the angular acceleration using the motor model dynamics
        total_samples = int(self.sample_time / self.delta_t)
        self.future_motor_accelerations = np.zeros(total_samples)
        self.omegas = np.zeros(total_samples)
        for i in range(total_samples):
            acceleration = self.motor_model_dynamics(voltage, self.omega, self.R, self.K_t, self.K_e, self.J, self.b)
            self.future_motor_accelerations[i] = acceleration
            self.current_omega += acceleration * self.delta_t
            self.omegas[i] = self.current_omega
        
    def update_motor_accelerations(self, direction, duration):
        if direction == 'left':
            direction = -1
        else:
            direction = 1

        """
        Lab 1 & 3 bonus: Model the expected acceleration response of the motor.
        """
        a_m_1 = 0.05
        a_m_2 = 0.05
        t1 = duration / 4
        t2_d = duration / 4
        t2 = duration - t2_d

        t_values = np.arange(0.0, duration + self.delta_t, self.delta_t)
        self.future_motor_accelerations = np.zeros_like(t_values)

        # Vectorized conditions
        condition1 = t_values <= t1
        condition2 = (t_values > t1) & (t_values < t2)
        condition3 = t_values >= t2

        # Vectorized calculations
        self.future_motor_accelerations[condition1] = -4 * direction * a_m_1 / (t1 * t1) * t_values[condition1] * (t_values[condition1] - t1)
        self.future_motor_accelerations[condition2] = 0
        self.future_motor_accelerations[condition3] = 4 * direction * a_m_2 / (t2_d * t2_d) * (t_values[condition3] - t2) * (t_values[condition3] - duration)

        _velocity = it.cumulative_trapezoid(self.future_motor_accelerations, initial=0)
        self.future_motor_positions = list(it.cumulative_trapezoid(_velocity, initial=0))
        # Apply position constraints and adjust acceleration
        # for i, relative_position in enumerate(self.future_motor_positions):
        #     absolute_position = relative_position + self.x_pivot
        #     if absolute_position > 100:
        #         self.future_motor_positions[i] = 100 - self.x_pivot
        #         if i > 0:
        #             self.future_motor_accelerations[i] = 0
        #             _velocity[i] = 0
        #     elif absolute_position < -100:
        #         self.future_motor_positions[i] = -100 - self.x_pivot
        #         if i > 0:
        #             self.future_motor_accelerations[i] = 0
        #             _velocity[i] = 0
        # self.future_motor_positions = list(it.cumulative_trapezoid(_velocity, initial=0)) #recalculate positions after velocity correction
        
    
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
    
    def step(self):
        # Get the predicted motor acceleration for the next step and the shift in x_pivot
        self.check_prediction_lists()
        #print(self.future_motor_accelerations)
        self.currentmotor_acceleration = self.future_motor_accelerations[0]
        self.future_motor_accelerations = np.delete(self.future_motor_accelerations, 0)
        self.x_pivot = self.x_pivot + self.future_motor_positions[0] / 3
        self.future_motor_positions = np.delete(self.future_motor_positions, 0)
        # Update the system state based on the action and model dynamics
        self.theta_double_dot = self.get_theta_double_dot(self.theta, self.theta_dot)
        self.theta += self.theta_dot * self.delta_t
        if self.theta > 2 * np.pi:
            self.theta -= 2 *np.pi
        elif self.theta < 0:
            self.theta += 2 * np.pi
        self.theta_dot += self.theta_double_dot * self.delta_t
        self.time += self.delta_t
        self.steps += 1

        return self.theta, self.theta_dot, self.theta_double_dot, self.x_pivot
        
    def step_2(self):
        # Get the predicted motor acceleration for the next step and the shift in x_pivot
        # only self.currentmotor_acceleration is used in the function get_theta_double_dot
        angular_acceleration = self.motor_model_dynamics(self.motor_voltage, self.current_omega, self.R, self.K_t, self.K_e, self.J, self.b)
        self.currentmotor_acceleration = angular_acceleration * self.pully_radius
        self.current_omega += angular_acceleration * self.delta_t
        self.x_pivot = self.x_pivot + self.current_omega * self.delta_t * self.pully_radius

        # Update the system state based on the action and model dynamics
        self.theta_double_dot = self.get_theta_double_dot(self.theta, self.theta_dot)
        self.theta_dot += self.theta_double_dot * self.delta_t
        self.theta += self.theta_dot * self.delta_t
        if self.theta > 2 * np.pi:
            self.theta -= 2 *np.pi
        elif self.theta < 0:
            self.theta += 2 * np.pi
        
        self.time += self.delta_t
        self.steps += 1

        return self.theta, self.theta_dot, self.theta_double_dot, self.x_pivot
    
    def draw_line_and_circles(self, colour, start_pos, end_pos, line_width=5, circle_radius=9):
        pygame.draw.line(self.screen, colour, start_pos, end_pos, line_width)
        pygame.draw.circle(self.screen, colour, start_pos, circle_radius)
        pygame.draw.circle(self.screen, colour, end_pos, circle_radius)

    def draw_pendulum(self, colour ,x, y, x_pivot):
        self.draw_line_and_circles(colour, [x_pivot+500, 400], [y+x_pivot+500, x+400])
        
    def render(self, theta, x_pivot):
        self.screen.fill((255, 255, 255))
        # Drawing length of the pendulum
        
        corrected_pivot = x_pivot * self.x_pivot_multiplier
        self.draw_pendulum((0,0,0),math.cos(theta)* self.pendulum_length,math.sin(theta)*self.pendulum_length,corrected_pivot)
        # Draw black line and circles for horizontal axis

        self.draw_line_and_circles((0, 0, 0), [self.track_x_start, 400], [self.track_x_end, 400])

        fps = self.clock.get_fps()  # Get the current FPS
        fps_text = self.font.render(f"FPS: {int(fps)}", True, (0, 0, 0))  # Render the FPS text in white
        velocity_text = self.font.render(f"Velocity: {(self.current_omega * self.pully_radius) } m/s", True, (0, 0, 0))
        screen = pygame.display.get_surface()  # Get the current display surface
        screen.blit(fps_text, (screen.get_width() - 100, 10))
        screen.blit(velocity_text, (screen.get_width() - 300, 10))
        pygame.display.flip()
        self.clock.tick(200) 

    def check_prediction_lists(self):
        if len(self.future_motor_accelerations) == 0:
            self.future_motor_accelerations = [0]
        if len(self.future_motor_positions) == 0:
            self.future_motor_positions = [0]

    def set_gravity_scale(self, scale):
        self.g = 9.81 * scale
        self.c_c = 0.01 * scale