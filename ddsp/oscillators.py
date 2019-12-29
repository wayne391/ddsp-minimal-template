# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np


class HarmonicOscillators(nn.Module):
    def __init__(self, n_partial, sample_rate, block_size):
        super(HarmonicOscillators, self).__init__()
        print(' >', self.__class__.__name__)
        self.n_partial = n_partial
        self.upsample = nn.Upsample(scale_factor = block_size, mode="linear", align_corners=False)
        self.k = nn.Parameter(torch.arange(1, n_partial + 1).reshape(1,1,-1).float(), requires_grad=False)
        self.sample_rate = sample_rate
    
    def forward(self, amp, alpha, f0):   
        # print('[harmonic osc]')  

        # print('f0:', f0.shape)
        # print('amp:', amp.shape)
        # print('alpha:', alpha.shape)

        # Upsample parameters
        f0          = self.upsample(f0.transpose(1,2)).squeeze(1) / self.sample_rate
        amp         = self.upsample(amp.transpose(1,2)).squeeze(1)
        alpha       = self.upsample(alpha.transpose(1,2)).transpose(1,2)
        
        # print('---------')
        # print('f0:', f0.shape)
        # print('amp:', amp.shape)
        # print('alpha:', alpha.shape)
        # Generate phase
        phi = torch.zeros(f0.shape).to(f0.device)
        for i in np.arange(1,phi.shape[-1]):
            phi[:,i] = 2 * np.pi * f0[:,i] + phi[:,i-1]
        phi = phi.unsqueeze(-1).expand(alpha.shape)
        # Filtering above Nyquist
        anti_alias = (self.k * f0.unsqueeze(-1) < .5).float()
        # print('anti_alias:', anti_alias.shape)
        # Generate the output signal
        # print('sin:', torch.sin(self.k * phi).shape)
        # print('ori:', (anti_alias * alpha * torch.sin(self.k * phi)).shape)
        y = amp * torch.sum(anti_alias * alpha * torch.sin(self.k * phi), -1)
        return y

    def n_parameters(self):
        """ Return number of parameters in the module """
        return self.n_partial + 1