import numpy as np
import random
import torch

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, device='cpu'):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.device = device
        
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros(capacity, dtype=np.int64)
        self.rews_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
        
    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        batch = dict(
            obs=torch.as_tensor(self.obs_buf[idxs], device=self.device),
            next_obs=torch.as_tensor(self.next_obs_buf[idxs], device=self.device),
            acts=torch.as_tensor(self.acts_buf[idxs], device=self.device),
            rews=torch.as_tensor(self.rews_buf[idxs], device=self.device),
            dones=torch.as_tensor(self.done_buf[idxs], device=self.device)
        )
        return batch

    def __len__(self):
        return self.size


