import random
import numpy as np
import torch
from collections import deque


class RandomReplayBuffer:
    
    def __init__(self, buffer_size=10000, batch_size=32, use_conv=True, use_minimax=True):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.use_conv = use_conv
        self.use_minimax = use_minimax
        self.buffer = deque(maxlen=buffer_size)
        self.start_size = 50000
    
    def add(self, *exp):        

        s,*a,r,s_prime,mask,d = exp

        if self.use_conv:
            s_ = s.reshape(6,7)
            s_prime_ = s_prime.reshape(6,7)
        else:
            s_ = s.flatten()
            s_prime_ = s_prime.flatten()

        self.buffer.append((s_,*a,r,s_prime_,mask,d))
    
    def get_length(self):
        return len(self.buffer)
    def get_maxlen(self):
        return self.buffer.maxlen
    
    def shuffle(self):
        random.shuffle(self.buffer)

    def sample(self):
        minibatch = random.sample(self.buffer, self.batch_size)

        if self.use_conv:
            # state_batch.shape: (batch_size, 1, 6, 7)
            s_batch = torch.stack([s1 for (s1,*a,r,s2,m,d) in minibatch]).unsqueeze(1).to(self.device)
            s_prime_batch = torch.stack([s2 for (s1,*a,r,s2,m,d) in minibatch]).unsqueeze(1).to(self.device)
            
        else:
            # state_batch.shape: (batch_size, 42)
            state1_batch = torch.stack([s1 for (s1,*a,r,s2,m,d) in minibatch]).to(self.device)
            state2_batch = torch.stack([s2 for (s1,*a,r,s2,m,d) in minibatch]).to(self.device)

        # action_batch.shape: (batch_size, )
        a_batch = torch.Tensor([a[0] for (s1,*a,r,s2,m,d) in minibatch]).to(self.device)
        if self.use_minimax:
            b_batch = torch.Tensor([a[1] for (s1,*a,r,s2,m,d) in minibatch]).to(self.device)
            a_batch = 7*a_batch + b_batch
        r_batch = torch.Tensor([r for (s1,*a,r,s2,m,d) in minibatch]).to(self.device)
        m_batch = torch.stack([m for (s1,*a,r,s2,m,d) in minibatch]).to(self.device)
        d_batch = torch.Tensor([d for (s1,*a,r,s2,m,d) in minibatch]).to(self.device)
        
        return s_batch, a_batch, r_batch, s_prime_batch, m_batch, d_batch
    

    def clear(self):
        self.buffer.clear()