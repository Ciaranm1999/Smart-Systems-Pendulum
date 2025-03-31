class hyperparameters:
    # reinforcement neural network class
    HIDDEN_LAYER_NODES = 32

    # reinformcement replay buffer class
    LEARNING_RATE = 1e-4
    EPSILON_START = 0.9
    EPSILON_DECAY = 0.9999
    EPSILON_MIN = 0.05

    # reinforcement pendulum class
    BATCH_SIZE = 512
    GAMMA = 0.9
    TAU = 0.001
    Q_MAX_VALUE = 40
    Q_MIN_VALUE = -40
    REWARDS_SCALING = 1/50
    # Lab 1 
    MAX_RUN_TIME = 60.0 # seconds
    SAMPLE_TIME = 0.5 # seconds
    EPOCHS_MAX = 50000
    MAX_POSITION = 0.6 # meters

    PRINT_OUTPUT = False
    RENDER = False
    SCALE = 1
    MANUAL_MODE = False

    # motor class
    MOTOR_R = 1.5  # Resistance in ohms
    MOTOR_K_t = 0.00375 * MOTOR_R  # Torque constant in Nm/A
    MOTOR_K_e = 0.00375 * MOTOR_R  # Back EMF constant in V/(rad/s)
    MOTOR_J = 9e-5  # Rotor inertia in kg.m^2
    MOTOR_B = 1e-6  # Viscous friction coefficient in Nm.s/rad
    MOTOR_V_MAX = 4.0  # Maximum voltage in volt
    PULLY_RADIUS = 0.01 # 1 cm

    PENDULUM_LENGHT = 0.35  # Length of the pendulum in meters
    TRACK_LENGTH = 0.5 # Length of the track in meters
    TRACK_DRAW_LENGTH = 300 # Length of the track in pixels
    OLD_METHOD = False  # Use the old method for calculating the motor model
    @staticmethod
    def to_string():
        attributes = {attr: getattr(hyperparameters, attr) for attr in dir(hyperparameters) 
                      if not callable(getattr(hyperparameters, attr)) and not attr.startswith("__")}
        return '\n'.join(f"{key}: {value}" for key, value in attributes.items())
    