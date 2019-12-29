# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class FilteredNoise(nn.Module):
    """
    Filtered noise generator implemented through frequency sampling as described in
    Engel et al. "DDSP: Differentiable Digital Signal Processing"
    https://openreview.net/pdf?id=B1x1ma4tDr
    
    Arguments:
            filter_size (int)   : size of the filter (number of coefficients)
            block_size (int)    : size of the block
    """
    
    def __init__(self, filter_size, block_size):
        super(FilteredNoise, self).__init__()
        print(' >', self.__class__.__name__)
        self.block_size = block_size
        self.filter_size = filter_size
        self.noise_att = 1e-4
        self.filter_window = nn.Parameter(torch.hann_window(filter_size).roll(filter_size//2,-1),requires_grad=False)
    
    def n_parameters(self):
        return self.filter_size // 2 + 1
    
    def forward(self, signal, filter_coef):
        # print('sig:', sig.shape)
        # print('filter_coef:', filter_coef.shape)
        # Create noise source
        noise = torch.randn(signal.shape).detach().to(signal.device).reshape(-1, self.block_size) * self.noise_att
        # print('noise:', noise.shape)
        
        S_noise = torch.rfft(noise, 1).reshape(signal.shape[0], -1, self.block_size // 2 + 1, 2)
        # Reshape filter coefficients to complex form
        # print('filter_coef ori:', filter_coef.shape)
        filter_coef = filter_coef.reshape([-1, self.filter_size // 2 + 1, 1]).expand([-1, self.filter_size // 2 + 1, 2]).contiguous()
        # print('filter_coef re:', filter_coef.shape)
        filter_coef[:,:,1] = 0
        # Compute filter windowed impulse response
        h = torch.irfft(filter_coef, 1, signal_sizes=(self.filter_size,))
        h_w = self.filter_window.unsqueeze(0) * h
        h_w = nn.functional.pad(h_w, (0, self.block_size - self.filter_size), "constant", 0)
        # Compute the spectral mask
        H = torch.rfft(h_w, 1).reshape(signal.shape[0], -1, self.block_size // 2 + 1, 2)
        # Filter the original noise
        S_filtered          = torch.zeros_like(H)
        # print(H[:,:,:,0].shape, S_noise[:,:,:,0].shape, H[:,:,:,1].shape, S_noise[:,:,:,1].shape)
        S_filtered[:,:,:,0] = H[:,:,:,0] * S_noise[:,:,:,0] - H[:,:,:,1] * S_noise[:,:,:,1]
        S_filtered[:,:,:,1] = H[:,:,:,0] * S_noise[:,:,:,1] + H[:,:,:,1] * S_noise[:,:,:,0]
        S_filtered          = S_filtered.reshape(-1, self.block_size // 2 + 1, 2)
        # Inverse the spectral noise back to signal
        filtered_noise = torch.irfft(S_filtered, 1)[:,:self.block_size].reshape(signal.shape[0], -1)
        return filtered_noise