from hyperparameters import hyperparameters
hp = hyperparameters()

class NewDigitalTwin:
    def __init__(self, scale): 
        self.scale = scale
        self.sample_time = hp.SAMPLE_TIME
        self.time = 0.0
        