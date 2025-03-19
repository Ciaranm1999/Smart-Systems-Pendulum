class hyperparameters:
    # reinforcement neural network class
    HIDDEN_LAYER_NODES = 128

    # reinformcement replay buffer class
    LEARNING_RATE = 1e-4
    EPSILON_START = 0.5
    EPSILON_DECAY = 0.995
    EPSILON_MIN = 0.01

    # reinforcement pendulum class
    BATCH_SIZE = 512
    GAMMA = 0.6
    TAU = 0.001
    Q_MAX_VALUE = 40
    Q_MIN_VALUE = -40
    REWARDS_SCALING = -10
    # Lab 1 
    MAX_RUN_TIME = 15.0 # seconds
    SAMPLE_TIME = 0.2 # seconds
    EPOCHS_MAX = 50000

    PRINT_OUTPUT = False
    RENDER = False