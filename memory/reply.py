from abc import ABC, abstractmethod
from collections import deque
import random
import numpy as np


DEFAULT_BUFFER_CAPACITY = 10000
DEFAULT_EMBEDDER_DIM = 384

##################################
# Reply Buffer Interfaces
##################################

# --- simple Replay Buffer interface ------------------------------------------------

class IReplayBuffer(ABC):
    """
    Interface for a replay buffer.
    """

    @abstractmethod
    def push(self, s, a, r, s2, done, info):
        """
        Add a transition to the buffer.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """
        Number of transitions in the buffer.
        """
        raise NotImplementedError


##################################
# Reply Buffer Classes
##################################

# --- random Replay Buffer ------------------------------------------------

class RandomReplayBuffer(IReplayBuffer):
    """
    Replay buffer implementation using random sampling.
    """

    def __init__(self, capacity=DEFAULT_BUFFER_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done, info):
        self.buffer.append((s, a, r, s2, done, info))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d, i = map(np.array, zip(*batch))
        return s, a, r, s2, d, i

    def __len__(self):
        return len(self.buffer)
    
# --- Embed-based Replay Buffer interface ------------------------------------------------

class EmbedReplyBuffer(IReplayBuffer):
    def __init__(self
                 , embedder = lambda s, a, r, s2, done, info: np.zeros((DEFAULT_EMBEDDER_DIM,))
                ):
        self.embedder = embedder
        self.embeddings = []

    def push(self, s, a, r, s2, done, info):
        embeddings = self.embedder(s, a, r, s2, done, info)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)

# # --- vecDB Replay Buffer interface ------------------------------------------------

# from retriever import *

# class EmbedReplyBuffer(IReplayBuffer):
#     def __init__(self, index, key_mapper: lambda s, a, r, s2, done, info: r):
#         self.index = index
#         self.lookup = dict()
#         self.key_mapper

#     def push(self, s, a, r, s2, done, info):
#         add_embeddings(self.index, s2)
#         self.lookup = dict(self.key_mapper(s, a, r, s2, done, info)=)

#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         return batch

#     def __len__(self):
#         return len(self.buffer)