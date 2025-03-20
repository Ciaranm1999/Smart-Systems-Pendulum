class hyperparameters:
    # reinforcement neural network class
    HIDDEN_LAYER_NODES = 64

    # reinformcement replay buffer class
    LEARNING_RATE = 1e-4
    EPSILON_START = 0.9
    EPSILON_DECAY = 0.9999
    EPSILON_MIN = 0.05

    # reinforcement pendulum class
    BATCH_SIZE = 512
    GAMMA = 0.65
    TAU = 0.001
    Q_MAX_VALUE = 40
    Q_MIN_VALUE = -40
    REWARDS_SCALING = 1/15
    # Lab 1 
    MAX_RUN_TIME = 15.0 # seconds
    SAMPLE_TIME = 0.2 # seconds
    EPOCHS_MAX = 50000

    PRINT_OUTPUT = False
    RENDER = False

    @staticmethod
    def to_string():
        attributes = {attr: getattr(hyperparameters, attr) for attr in dir(hyperparameters) 
                      if not callable(getattr(hyperparameters, attr)) and not attr.startswith("__")}
        return '\n'.join(f"{key}: {value}" for key, value in attributes.items())
    