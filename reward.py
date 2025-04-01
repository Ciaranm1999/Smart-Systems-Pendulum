#%%
import torch
import numpy as np
from reinforcement import PendulumRlAgent
import matplotlib.pyplot as plt

pa = PendulumRlAgent()

rewards = []
for i in np.linspace(2.0 * torch.pi, 0, 100):
    rewards.append(pa.reward_calculation(torch.tensor([0.0]), torch.tensor([i])))

plt.plot(np.linspace(2.0 * torch.pi, 0.1, len(rewards)), rewards)
plt.xlabel('Angle (radians)')
plt.ylabel('Reward')
plt.title('Reward vs Angle')
plt.grid(True)
plt.show()