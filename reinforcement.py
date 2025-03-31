import torch
from torch import nn
import numpy as np
import random
from hyperparameters import hyperparameters
import pickle
import os

hp = hyperparameters()

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        mid_dim = hp.HIDDEN_LAYER_NODES
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
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
    def __init__(self, capacity=10_500_000, load_session=None):
        self.device = torch.device('cuda')
        self.model = NeuralNetwork().to(self.device)
        self.replay_buffer = ReplayBuffer(capacity)
        if load_session is not None:
            path = f'./training_sessions/session_{load_session}/'
            self.load_model(path)
                
        self.target_model = NeuralNetwork().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hp.LEARNING_RATE)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3
        # self.loss_fn = nn.MSELoss()
        self.actions = [('right', 1),	
                        ('left', 1)]

        self.epsilon = hp.EPSILON_START
        self.epsilon_decay = hp.EPSILON_DECAY
        self.epsilon_min = hp.EPSILON_MIN

        self.hard_update_counter = 0
        self.hard_update_frequency = 1000

    def predict(self, angle, angle_velocity, angle_acceleration, velocity):
        # normalized_data = self.normalise_input(angle, angle_velocity, angle_acceleration, velocity)
        # data = [normalized_data[key] for key in ['angle_velocity', 'angle_acceleration', 'angle', 'velocity']]
        data = [angle_velocity, angle_acceleration, angle, velocity]
        x_data = torch.tensor(data).float().to(self.device)
        with torch.no_grad():
            logits = self.model(x_data)
            pred_probab = nn.Softmax(dim=0)(logits)
            top_action = pred_probab.argmax().item()  # Action with highest probability

        if random.random() < self.epsilon:  # Explore
            y_pred = random.randint(0, len(self.actions) - 1)
        else:  # Exploit
            y_pred = top_action
        
        return self.actions[y_pred], y_pred, data
    
    def normalise_input(self, angle=None, angle_velocity=None, angle_acceleration=None, position=None, reward=None):
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
            "angle_acceleration": (-30.0, 30.0),
            "position": (-100.0, 100.0),
            "reward": (-80.0, 40.0),
        }

        input_values = {
            "angle": angle,
            "angle_velocity": angle_velocity,
            "angle_acceleration": angle_acceleration,
            "position": position,
            "reward": reward,
        }

        normalized_data = {}
        for key, value in input_values.items():
            if value is not None:
                min_val, max_val = normalization_ranges[key]
                normalized_value = normalize(value, min_val, max_val)
                normalized_data[key] = normalized_value

        return normalized_data

    def reward_calculation(self, position, angle):
        angle_desired = torch.tensor(1.0 * torch.pi) # Desired angle (upright)
        position_desired = torch.tensor(0.0)  # Desired position (center)
        angle_sigma = torch.tensor(0.3 * torch.pi)  # Width of angle "tolerance"
        position_sigma = torch.tensor(0.1)  # Width of position "tolerance"
        hanging_threshold = torch.tensor(1.7 * torch.pi) # Angle to consider "hanging"
        hanging_penalty = 0.5  # Penalty for hanging
        position_penalty = 0.1 # Penalty for being away from center
        # --- Angle Reward ---
        # Wrap angle to be within -pi to pi of desired angle
        angle_error = torch.abs(angle - angle_desired)
        angle_error = torch.min(angle_error, 2 * torch.pi - angle_error)  # Wrap-around distance

        angle_reward = torch.exp(-0.5 * (angle_error / angle_sigma) ** 2)  # Gaussian-like reward

        # --- Position Reward ---
        position_reward = torch.exp(-0.5 * (position / position_sigma) ** 2)

        # --- Hanging Penalty ---
        hanging_penalty_term = torch.where(
            (angle > hanging_threshold) | (angle < (2 * torch.pi - hanging_threshold)),
            -hanging_penalty,
            0.0,
        )

        # --- Position Penalty
        position_penalty_term = -position_penalty * torch.abs(position)

        # --- Total Reward ---
        total_reward = angle_reward + position_reward + hanging_penalty_term + position_penalty_term

        return angle_reward + hanging_penalty_term
    
        return reward
    
    def train(self):
        BATCH_SIZE = hp.BATCH_SIZE
        GAMMA = hp.GAMMA
        STATE_DIM = 4
        TAU = hp.TAU 

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
            reward = sample[STATE_DIM+1]
            next_state = sample[STATE_DIM+2:]
            
            state = torch.tensor(state).float().to(self.device)
            next_state = torch.tensor(next_state).float().to(self.device)
            action = torch.tensor(action).long().to(self.device).unsqueeze(0)
            reward = self.normalise_input(reward = torch.tensor(reward).float().to(self.device).unsqueeze(0))['reward']

            rewards.append(reward)
            q_value = self.model(state).gather(0, action)

            next_q_value = self.target_model(next_state).max(0)[0]#self.target_model(next_state).max(1)[0].unsqeeze(0)
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

    def train_faster(self, print_output=False):
        BATCH_SIZE = hp.BATCH_SIZE
        GAMMA = hp.GAMMA
        STATE_DIM = 4
        TAU = hp.TAU
        REWARDS_SCALING = hp.REWARDS_SCALING
        if len(self.replay_buffer) < BATCH_SIZE:
            if print_output:
                print(f"Replay buffer has {len(self.replay_buffer)} samples, waiting for {BATCH_SIZE}")
            return 0

        samples = self.replay_buffer.sample(BATCH_SIZE)

        # Extract data from samples
        states = torch.tensor([sample[:STATE_DIM] for sample in samples]).float().to(self.device)
        actions = torch.tensor([sample[STATE_DIM] for sample in samples]).long().to(self.device).unsqueeze(1)
        rewards = torch.tensor([sample[STATE_DIM + 1] for sample in samples]).float().to(self.device)
        next_states = torch.tensor([sample[STATE_DIM + 2:] for sample in samples]).float().to(self.device)

        # Normalize rewards
        # rewards = self.normalise_input(reward=rewards)['reward']
        rewards *= REWARDS_SCALING
        # Calculate Q-values
        q_values = self.model(states).gather(1, actions)
        
        # Calculate target Q-values
        next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards.unsqueeze(1) + GAMMA * next_q_values
        target_q_values = torch.clamp(target_q_values, hp.Q_MIN_VALUE, hp.Q_MAX_VALUE)
        # Calculate loss
        # loss = nn.MSELoss()(q_values, target_q_values)
        loss = nn.SmoothL1Loss()(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Calculate mean reward
        mean_reward = torch.mean(rewards).item()
        if print_output:
            print(f"Loss: {loss.item()}\t Reward: {mean_reward}\tepsilon: {self.epsilon}")

        # Soft update of target model
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
        
        # Hard update of target model
        # self.hard_update_counter += 1
        # if self.hard_update_counter % self.hard_update_frequency == 0:
        #     self.target_model.load_state_dict(self.model.state_dict())  # Hard update
        #     self.hard_update_counter = 0  # Reset the counter

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return mean_reward

    def save_model(self, path):
        if not path.endswith('/'):
            path += '/'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'replay_buffer': self.replay_buffer
        }, path + 'checkpoint.pth') #save all the data into one file.
        
        

    def load_model(self, path):
        if not path.endswith('/'):
            path += '/'
        # checkpoint = torch.load(path + 'checkpoint.pth', map_location=torch.device('cpu'))
        checkpoint = torch.load(path + 'checkpoint.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.replay_buffer = checkpoint['replay_buffer']
        self.target_model.load_state_dict(self.model.state_dict()) #ensure target model is in sync.


