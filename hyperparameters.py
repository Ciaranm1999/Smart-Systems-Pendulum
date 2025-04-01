class hyperparameters:
    # Digital twin
    DELTA_T  = 0.005  # Time step for the simulation in seconds
    GRAVITY = 9.81  # m/s^2, acceleration due to gravity
    PENDULUM_LENGHT = 0.35  # Length of the pendulum in meters
    PENDULUM_MASS = 0.3  # kg, mass of the pendulum bob
    PENDULUM_MOTOR_FORCE_MULTIPLIER = 2.5  # Multiplier for the motor force
    COLOUMB_FRICTION = 0.02  # Coefficient of Coulomb friction
    AIR_FRICTION = 0.001  # Coefficient of air friction

    TRACK_LENGTH = 0.5 # Length of the track in meters
    TRACK_DRAW_LENGTH = 300 # Length of the track in pixels

    # motor class
    MOTOR_R =1.5  # Resistance in ohms
    MOTOR_K_t = 0.00375 * MOTOR_R  /2# Torque constant in Nm/A
    MOTOR_K_e = 0.00375 * MOTOR_R  /2# Back EMF constant in V/(rad/s)
    MOTOR_J = 9e-5  # Rotor inertia in kg.m^2
    MOTOR_B = 1e-6  # Viscous friction coefficient in Nm.s/rad
    MOTOR_V_MAX = 6.0  # Maximum voltage in volt
    PULLY_RADIUS = 0.01 # 1 cm

    