import pygame
import scipy.integrate
import serial
import numpy as np
import csv
import math
import scipy.integrate as it
import time
import pandas as pd

import hyperparameters
hp = hyperparameters.hyperparameters()

class DigitalTwin:
    def __init__(self):
        # Initialize Pygame parameters
        self.screen = None
        # Initialize serial communication parameters
        self.ser = None
        self.device_connected = False
        # State configuration parameters
        self.steps = 0
        self.theta = 0.#np.pi-0.01
        self.theta_dot = 0.
        self.theta_double_dot = 0.
        self.x_pivot = 0
        self.delta_t = hp.DELTA_T  # Example value, adjust as needed in seconds 19.42879347
        # Model parameters
        self.g = hp.GRAVITY  # Acceleration due to gravity (m/s^2)
        self.l = hp.PENDULUM_LENGHT   # Length of the pendulum (m)
        self.c_air = hp.AIR_FRICTION  # Air friction coefficient
        self.c_c = hp.COLOUMB_FRICTION   # Coulomb friction coefficient
        self.a_m = 1 # Motor acceleration force tranfer coefficient
        self.m = hp.PENDULUM_MASS # Mass of the pendulum
        self.future_motor_accelerations = []
        self.future_motor_positions = []
        self.currentmotor_acceleration = 0.
        self.time = 0.
        # Sensor data
        self.sensor_theta = 0
        self.current_sensor_motor_position = 0.
        self.current_action = 0

        # Parameters that have to do with the new motor model
        self.R = hp.MOTOR_R  # Armature Resistance (Ohms)
        self.K_t = hp.MOTOR_K_t  # Torque Constant (Nm/A)
        self.K_e = hp.MOTOR_K_e  # Back EMF Constant (V/(rad/s)) - Often numerically equal to K_t in SI
        self.J = hp.MOTOR_J  # Rotor Inertia (kg.m^2)
        self.b = hp.MOTOR_B  # Viscous Friction Coefficient (Nm.s/rad)
        self.pully_circumference = hp.PULLY_RADIUS * 2 * np.pi  # Circumference of the pulley (m)
        self.pully_radius = hp.PULLY_RADIUS  # Circumference of the pulley (m)
        self.current_omega = 0.0  # Initial angular velocity (rad/s)
        self.motor_voltage = hp.MOTOR_V_MAX  # Initial motor voltage (V)

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
        # Initialize a pygame window
        self.initialize_pygame_window()

    def initialize_pygame_window(self):
        # Set up the drawing window
        pygame.init()
        self.screen = pygame.display.set_mode([1000, 800])

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
        if self.device_connected:
            if direction == 'left':
                d = -duration
            else:
                d = duration
            self.ser.write(str(d).encode())
        if duration > 0:
            self.update_motor_accelerations_2(direction, duration/1000)

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
    def update_motor_accelerations_2(self, direction, duration):

        """
        Simulates the motor's acceleration and CALCULATES RELATIVE POSITION change
        over a duration using the physical model and a specific voltage profile:
        - Phase 1 (e.g., 1st 3/4 duration): Full Voltage
        - Phase 2 (e.g., Last 1/4 duration): Zero Voltage (Coast/Dynamic Braking)

        The output position list contains angular positions relative to the
        starting position at the time this function is called.

        Args:
            direction (str): 'left' or 'right'.
            duration (float): Total simulation duration (s).
        """
        if direction == 'left':
            direction_multiplier = -1
        else: # 'right'
            direction_multiplier = 1

        # Reset instantaneous velocity state for the new simulation run
        # Keep any absolute position state separate if needed elsewhere in the class
        self.current_omega = 0.0

        # Initialize lists for output
        self.future_motor_accelerations = []
        self.future_motor_positions = []

        # --- Track position RELATIVE to the start of this simulation ---
        relative_position = 0.0
        # Add the initial relative position (0)
        self.future_motor_positions.append(relative_position)

        # --- Define Phase Timing ---
        # Point where voltage switches from full to zero
        t_voltage_off = duration * 0.75 # Adjust fraction as needed

        # --- Simulation Loop ---
        num_steps = int(round(duration / self.delta_t))
        if num_steps <= 0:
            print("Warning: Duration too short for the given delta_t.")
            # Still return the initial relative position if no steps run
            return

        for i in range(num_steps):
            # Calculate time at the *start* of the interval for voltage decision
            current_interval_start_time = i * self.delta_t

            # --- Voltage Control Strategy (Full Voltage then Zero Voltage) ---
            if current_interval_start_time < t_voltage_off:
                # Phase 1: Apply full voltage
                applied_voltage = direction_multiplier * self.motor_voltage
            else:
                # Phase 2: Apply zero voltage
                applied_voltage = 0.0
            # --- End Voltage Control ---

            # 1. Calculate current acceleration using the model
            angular_acceleration = self.motor_model_dynamics(
                applied_voltage,
                self.current_omega, # Use omega from the beginning of the timestep
                self.R, self.K_t, self.K_e, self.J, self.b
            )
            linear_acceleration = angular_acceleration * self.pully_radius
            # Store the calculated acceleration for this step
            self.future_motor_accelerations.append(linear_acceleration)

            # 2. Update velocity for the *end* of this step / start of next (Forward Euler)
            # NOTE: self.current_omega still needs to be updated for the dynamics calculation
            next_omega = self.current_omega + angular_acceleration * self.delta_t

            # 3. Calculate CHANGE in position during this step and update relative position
            #    Using next_omega (velocity at end of interval) * delta_t
            delta_position = next_omega *self.pully_radius * self.delta_t
            relative_position = relative_position + delta_position

            # Store the calculated CUMULATIVE RELATIVE position
            self.future_motor_positions.append(relative_position)

            # 4. Update omega state for the next iteration's dynamics calculation
            self.current_omega = next_omega
    def update_motor_accelerations(self, direction, duration):
        if direction == 'left':
            direction = -1
        else:
            direction = 1

        """
        Lab 1 & 3 bonus: Model the expected acceleration response of the motor.  
        """

        # total_samples = int(duration/self.delta_t) + 1
        # self.future_motor_accelerations = []
        # self.future_motor_positions = []
        # self.current_omega = 0.0  # Reset current omega for each action
        # for i in range(total_samples):
        #     angular_acceleration = self.motor_model_dynamics(self.motor_voltage, self.current_omega, self.R, self.K_t, self.K_e, self.J, self.b)
        #     self.current_omega += acceleration * self.delta_t

        #     linear_acceleration = acceleration * self.pully_radius
        #     self.future_motor_accelerations.append(linear_acceleration)

        a_m_1 = 0.05
        a_m_2 = 0.05
        t1 = duration/4
        t2_d = duration/4
        t2 = duration - t2_d
        for t in np.arange(0.0, duration+self.delta_t, self.delta_t):
            if t <= t1:
                c = -4*direction*a_m_1/(t1*t1) * t * (t-t1)
            elif t < t2 and t > t1:
                c = 0 
            elif t >= t2:
                c = 4*direction*a_m_2/(t2_d*t2_d) * (t-t2) * (t-duration)
            
            self.future_motor_accelerations.append(c)
        
        _velocity = it.cumulative_trapezoid(self.future_motor_accelerations,initial=0)
        self.future_motor_positions = list(it.cumulative_trapezoid(_velocity,initial=0))
    
    
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
        motor_torque = -(1 / self.l) * self.currentmotor_acceleration * np.cos(theta) * self.a_m
        
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
        self.currentmotor_acceleration = self.future_motor_accelerations.pop(0)
        self.x_pivot = self.x_pivot + self.future_motor_positions.pop(0)/3
        # Update the system state based on the action and model dynamics
        self.theta_double_dot = self.get_theta_double_dot(self.theta, self.theta_dot)
        self.theta_dot += self.theta_double_dot * self.delta_t
        self.theta += self.theta_dot * self.delta_t

        self.time += self.delta_t
        self.steps += 1

        return self.theta, self.theta_dot, self.x_pivot
        

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
        pygame.display.flip()

    def check_prediction_lists(self):
        if len(self.future_motor_accelerations) == 0:
            self.future_motor_accelerations = [0]
        if len(self.future_motor_positions) == 0:
            self.future_motor_positions = [0]
