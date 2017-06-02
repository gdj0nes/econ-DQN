import numpy as np
import tensorflow as tf

class ExperienceBuffer(object):
    """Used for creating a buffer of experiences to train the agent"""

    def __init__(self, buffer_size, batch_size, state_dims=1, action_dims=1, history_length=1):

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_dims = state_dims

        self.actions = np.empty([buffer_size, action_dims], dtype=np.float32)
        self.states = np.empty([buffer_size, state_dims])  # T
        self.rewards = np.empty([buffer_size], dtype=np.float32)
        self.history_length = history_length
        # The next state for a given state will be the next index
        self.terminals = np.zeros(buffer_size)  # Stores whether the observation has ended

        self.current = 0
        self.count = 0

    def add(self, action, reward, state, terminal):

        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.states[self.current] = state
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.buffer_size

    def add_batch(self, action, reward, state, terminal):

        size = len(action)
        self.actions[self.current:self.current + size] = action
        self.rewards[self.current:self.current + size] = reward.squeeze()
        self.states[self.current:self.current + size] = state
        self.terminals[self.current:self.current + size] = terminal
        self.count = max(self.count, self.current + size)
        self.current = (self.current + size) % self.buffer_size

    def sample(self):

        # Generate a random index
        prestates = np.empty([self.batch_size, self.state_dims])
        poststates = np.empty([self.batch_size, self.state_dims])

        indexes = []
        while len(indexes) < self.batch_size:
            # Keep drawing candidate indices
            while True:
                index = np.random.randint(self.history_length, self.count - 1)  # This is going to be an exausting
                # Determine if enough prior observations
                if index >= self.current > index - self.history_length:
                    continue
                # Check if terminal component in sequence
                if self.terminals[(index - self.history_length):index].any():
                    continue
                break

            # Fill pre and post states
            prestates[len(indexes), ...] = self.retrieve(index - 1)
            poststates[len(indexes), ...] = self.retrieve(index)
            indexes.append(index)

        # Get data to return in batch
        rv_rewards = self.rewards[indexes]
        rv_actions = self.actions[indexes]
        rv_terminals = self.terminals[indexes]

        rv_rewards = (rv_rewards - rv_rewards.mean()) / rv_rewards.std()
        return rv_actions, rv_rewards, prestates, poststates, rv_terminals

    def retrieve(self, index):

        index = index % self.count
        if index >= self.history_length - 1:
            return self.states[(index - (self.history_length - 1)):(index + 1), ...]
        else:
            indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
            return self.states[indexes, ...]