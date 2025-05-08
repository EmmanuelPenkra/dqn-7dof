import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device", device)

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state (7 for joint angles)
            action_size (int): Dimension of each action (3^7 = 2187)
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Paper: Input Layer (7) -> FC (128, ReLU) -> FC (256, ReLU) -> FC (2187)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        np.random.seed(seed)


    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, hyperparams):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            hyperparams (dict): dictionary of hyperparameters
        """
        self.state_size = state_size
        self.action_size = action_size
        random.seed(seed) # Seed for python's random
        torch.manual_seed(seed) # Seed for PyTorch
        np.random.seed(seed) # Seed for NumPy


        self.buffer_size = hyperparams.get('BUFFER_SIZE', 100000)
        self.batch_size = hyperparams.get('BATCH_SIZE', 64)
        self.gamma = hyperparams.get('GAMMA', 0.9)
        self.tau = hyperparams.get('TAU', 1e-3)
        self.lr = hyperparams.get('LEARNING_RATE', 0.01)
        self.update_every = hyperparams.get('UPDATE_EVERY', 1) # Learn on every step
        self.gradient_clip = hyperparams.get('GRADIENT_CLIP', 1.0)
        # Start learning when buffer has enough samples, paper implies after 1st batch is ready
        self.learn_starts = hyperparams.get('LEARN_STARTS', self.batch_size) 

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Initialize target network with local network's weights
        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1.0) # Hard copy initially

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.learn_step_counter = 0 # To count learning steps for target network update if hard update was used

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) >= self.learn_starts:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
                self.learn_step_counter +=1


    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state (joint angles)
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Q_targets = r + γ * max_a' Q_target(s', a')
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets) # Paper Eq. (7) (R + gamma max Q' - Q)^2
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.gradient_clip)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        # Soft update is used based on "Target Smooth Factor" in Table 1
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, filename="dqn_agent.pth"):
        """Saves the local Q-network weights."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.qnetwork_local.state_dict(), filename)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename, state_size, action_size, seed, hyperparams):
        """Loads a DQNAgent from a saved Q-network state_dict."""
        agent = cls(state_size, action_size, seed, hyperparams)
        # Ensure the model is loaded to the correct device (e.g. CPU if trained on GPU but testing on CPU)
        agent.qnetwork_local.load_state_dict(torch.load(filename, map_location=device))
        agent.qnetwork_target.load_state_dict(torch.load(filename, map_location=device)) # also sync target
        agent.qnetwork_local.eval() # Set to eval mode after loading
        agent.qnetwork_target.eval()
        print(f"Model loaded from {filename}")
        return agent