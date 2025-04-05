class hyperparameters:
    # Digital twin
    DELTA_T  = 0.01  # Time step for the simulation in seconds
    GRAVITY = 9.81  # m/s^2, acceleration due to gravity
    PENDULUM_LENGHT = 0.2358244897959183  # Length of the pendulum in meters
    PENDULUM_MASS = 0.3  # kg, mass of the pendulum bob
    PENDULUM_MOTOR_FORCE_MULTIPLIER = 2.5  # Multiplier for the motor force

    AIR_FRICTION = 0.0004564646464646465  # Coefficient of air friction
    COLOUMB_FRICTION = 0.0071535353535353  # Coefficient of Coulomb friction
    # 0.0007374749498997996, 0.005070140280561123, 0.23240816326530614
    TRACK_LENGTH = 0.27 # Length of the track in meters
    TRACK_DRAW_LENGTH = 250 # Length of the track in pixels

    # motor class
    MOTOR_R =7  # Resistance in ohms
    MOTOR_K_t = 0.0325 # Torque constant in Nm/A
    MOTOR_K_e = 0.0325 # Back EMF constant in V/(rad/s)
    MOTOR_J = 4e-4  # Rotor inertia in kg.m^2
    MOTOR_B = 1e-6  # Viscous friction coefficient in Nm.s/rad
    MOTOR_V_MAX = 12.0  # Maximum voltage in volt
    PULLY_RADIUS = 0.005 # 1 cm

    range_encoder = 163

    RENDER = True