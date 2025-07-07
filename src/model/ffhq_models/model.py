import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from model_util import Encoder

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Linear(256,256)
    
    def forward(self, t, x):
        # input Bx3x256x256, output B 
        return self.W(x).reshape(len(x), -1).mean(1)


class Discriminator2(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Encoder()
        self.linear = nn.Linear(512,1)
    
    def forward(self, t, x):
        # input (B, Bx18x512), output B 
        temb = get_timestep_embedding(t, 512)[:,None,:]
        emb = torch.cat([temb, x], dim=1)

        out = self.transformer(emb)
        out = out.mean(1)     
        return self.linear(out).squeeze(1)
    
# class Discriminator3(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.transformer = Encoder()
#         self.all_module = nn.Sequential(
#                                         nn.Linear(512, 512),
#                                         nn.ReLU(),
#                                         nn.Linear(512, 512),
#                                         nn.ReLU(),
#                                         nn.Linear(512, 512),
#                                         nn.ReLU(),
#                                         nn.Linear(512, 512),
#                                         nn.ReLU(),
#                                         nn.Linear(512, 1)
#                                         )
    
#     def forward(self, t, x):
#         # input (B, Bx18x512), output B 
#         temb = get_timestep_embedding(t, 512)[:,None,:]
#         emb = torch.cat([temb, x], dim=1)

#         out = self.transformer(emb)
#         out = out.mean(1)     
#         return self.linear(out).squeeze(1)


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.transformer = Encoder(n_layers=4)
        self.linear = nn.Sequential(nn.Linear(512,512))

    def forward(self, t, x):
        # input (B, Bx18x512), output Bx1x512
        temb = get_timestep_embedding(t, 512)[:,None,:]
        # xemb = self.space_embedding(x)
        emb = torch.cat([temb, x], dim=1)

        out = self.transformer(emb)[:, 1:]
        # out = torch.sum(out, dim=1, keepdim=True)
        # out = torch.sum(out, dim=1)
        # return self.linear(out[:,1:])
        return self.linear(out)
    

