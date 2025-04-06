#%%
from Digital_twin_simplified import DigitalTwin
import numpy as np
digital_twin = DigitalTwin()
actions = [0.2, 0.15, 0.1, 0.05]

for action in actions:
    digital_twin.update_motor_accelerations_2('right', action)
    print(f"Motor acceleration for action {action}: {np.sum(digital_twin.future_motor_positions)}")
    digital_twin.future_motor_positions = []
