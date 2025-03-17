import torch
from torch import nn
import numpy as np
import random

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 9),
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class PendulumRlAgent:
    def __init__(self, capacity=13000):
        self.device = torch.device('cuda')
        self.model = NeuralNetwork().to(self.device)
        self.target_model = NeuralNetwork().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.actions = [('left', 0), 
                        ('left', 50), 
                        ('left', 100), 
                        ('left', 150), 
                        ('left', 200), 
                        ('right', 50), 
                        ('right', 100), 
                        ('right', 150), 
                        ('right', 200)]
        self.replay_buffer = ReplayBuffer(capacity)
        # self.data = [self.position, self.velocity, self.angle, self.angle_velocity, self.angle_acceleration]
        # self.x_data = torch.tensor(self.data).float().to(self.device)
        # self.logits = self.model(self.x_data)
        # self.pred_probab = nn.Softmax()(self.logits)
        # self.y_pred = self.pred_probab.argmax()
        # self.loss = self.loss_fn(self.logits, self.x_data)
        # self.optimizer.zero_grad()
        # self.loss.backward()
        # self.optimizer.step()
        # print(f"Predicted class: {self.y_pred}")
        # print(self.loss)
        # print(self.model)
    
    def predict(self, angle, angle_velocity, angle_acceleration, position):
        normalized_data = self._normalise_input(angle, angle_velocity, angle_acceleration, position)
        data = [normalized_data[key] for key in ['position', 'angle_velocity', 'angle_acceleration', 'angle']]
        x_data = torch.tensor(data).float().to(self.device)
        logits = self.model(x_data)
        pred_probab = nn.Softmax(dim=0)(logits)
        y_pred = torch.multinomial(pred_probab, 1).item()

        return self.actions[y_pred], y_pred
        self.loss = self.loss_fn(self.logits, self.x_data)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        print(f"Predicted class: {self.y_pred}")
        print(self.loss)
        print(self.model)
    
    def _normalise_input(self, angle=None, angle_velocity=None, angle_acceleration=None, position=None):
        """
        Normalizes input values and returns a dictionary of normalized values.

        Args:
            angle: The angle value to normalize.
            angle_velocity: The angle velocity value to normalize.
            angle_acceleration: The angle acceleration value to normalize.
            position: The position value to normalize.

        Returns:
            A dictionary of normalized values.
        """

        def normalize(value, min_val, max_val):
            """Helper function to normalize a single value."""
            if value is None:
                return None
            return (value - min_val) / (max_val - min_val)

        normalization_ranges = {
            "angle": (0, 2 * np.pi),
            "angle_velocity": (-15.0, 15.0),
            "angle_acceleration": (-20.0, 20.0),
            "position": (-100.0, 100.0),
        }

        input_values = {
            "angle": angle,
            "angle_velocity": angle_velocity,
            "angle_acceleration": angle_acceleration,
            "position": position,
        }

        normalized_data = {}
        for key, value in input_values.items():
            if value is not None:
                min_val, max_val = normalization_ranges[key]
                normalized_value = normalize(value, min_val, max_val)
                normalized_data[key] = normalized_value

        return normalized_data

    def reward_calculation(self, position, angle):
        # Reward function
        angle_scale = 1.0
        position_scale = 0.3

        # Normalize the input values
        normalized_data = self._normalise_input(angle=angle, position=position)
        angle_normalized = normalized_data['angle']
        position_normalized = normalized_data['position']

        # we want around 0.5 and 0.0 for angle and position respectively        
        angle_desired = 0.5
        angle_reward = 0.5-abs(angle_normalized - angle_desired) * angle_scale

        position_desired = 0.5
        position_reward = 0.5-abs(position_normalized - position_desired) * position_scale

        reward = angle_reward + position_reward
    
        return reward
    
    def train(self):
        BATCH_SIZE = 64
        GAMMA = 0.99
        STATE_DIM = 4
        REWARD_MULTIPLIER = 0.1
        TAU = 0.005 

        if len(self.replay_buffer) < BATCH_SIZE:
            print(f"Replay buffer has {len(self.replay_buffer)} samples, waiting for {BATCH_SIZE}")
            return
        
        samples = self.replay_buffer.sample(BATCH_SIZE)
        q_values = []
        target_q_values = []
        rewards = []
        for sample in samples:
            state = sample[:STATE_DIM]
            action = sample[STATE_DIM]
            reward = sample[STATE_DIM+1] * REWARD_MULTIPLIER
            next_state = sample[STATE_DIM+2:]
            
            state = torch.tensor(state).float().to(self.device)
            next_state = torch.tensor(next_state).float().to(self.device)
            action = torch.tensor(action).long().to(self.device).unsqueeze(0)
            reward = torch.tensor(reward).float().to(self.device).unsqueeze(0)
            rewards.append(reward)
            q_value = self.model(state).gather(0, action)
            temp_var = self.target_model(next_state)

            next_q_value = temp_var.max(0)[0]#self.target_model(next_state).max(1)[0].unsqeeze(0)
            target_q_value = reward + GAMMA * next_q_value
            q_values.append(q_value)
            target_q_values.append(target_q_value)
        
        loss = nn.MSELoss()(torch.stack(q_values), torch.stack(target_q_values))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        mean_reward = torch.mean(torch.stack(rewards)).item()
        print(f"Loss: {loss.item()}\t Reward: {mean_reward}")
        
        # soft update of target model
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
            # state, action, reward, next_state = sample
            # state = torch.tensor(state).float().to(self.device)
            # next_state = torch.tensor(next_state).float().to(self.device)
            # action = torch.tensor(action).float().to(self.device)
            # reward = torch.tensor(reward).float().to(self.device)

            # self.optimizer.zero_grad()
            # q_values = self.model(state)
            # next_q_values = self.model(next_state)
            # next_q_value = torch.max(next_q_values).item()
            # target_q = reward + GAMMA * next_q_value
            # loss = self.loss_fn(q_values, target_q)
            # loss.backward()
            # self.optimizer.step()
        pass
# device = torch.device('cuda')
# model = NeuralNetwork().to(device)
# print(model)
#%%
position = 1.0
velocity = 0.5
angle = 0.1
angle_velocity = 0.05
angle_acceleration = 0.01

# data = [position, velocity, angle, angle_velocity, angle_acceleration]
# x_data = torch.tensor(data).float().to(device)
# # print(torch.flatten(x_data))
# logits = model(x_data)
# pred_probab = nn.Softmax()(logits)
# y_pred = pred_probab.argmax()
# nn.functional.mse_loss(logits, x_data)
# print(f"Predicted class: {y_pred}")
# %%


