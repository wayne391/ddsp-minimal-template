# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from modules import ResConv1d
    

class FIRFilter(nn.Module):
    """
    FIR filter block implemented through frequency sampling as described in
    Engel et al. "DDSP: Differentiable Digital Signal Processing"
    https://openreview.net/pdf?id=B1x1ma4tDr
    
    Arguments:
            conv (nn.Conv1d)            : convolution module to wrap
            window_name (str or None)   : name of the window used to smooth the convolutions
            squared (bool)              : if `True`, square the smoothing window
    """
    
    def __init__(self, transform, filter_size, block_size):
        super(Filter, self).__init__()
        print(' >', self.__class__.__name__)
        self.apply(self.init_parameters)
        self.block_size = block_size
        self.filter_size = filter_size
        self.noise_att = 1e-4
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        """ Initialize internal parameters (sub-modules) """
        m.data.uniform_(-0.01, 0.01)

    def n_parameters(self):
        """ Return number of parameters in the module """
        return self.filter_size
    
    def set_paramters(self, z):
        # Obtain filter coefficients through network (amortization)
        self.fiter_coef = z
    
    def forward(self, z):
        z, cond = z
        # Reshape filter coefficients to complex form
        filter_coef = self.filter_coef.reshape([-1, self.filter_size // 2 + 1, 1]).expand([-1, self.filter_size // 2 + 1, 2]).contiguous()
        filter_coef[:,:,1] = 0
        # Compute filter windowed impulse response
        h = torch.irfft(filter_coef, 1, signal_sizes=(self.filter_size,))
        h_w = self.filter_window.unsqueeze(0) * h
        h_w = nn.functional.pad(h_w, (0, self.block_size - self.filter_size), "constant", 0)
        # Compute the spectral transform 
        S_sig = torch.rfft(z, 1).reshape(z.shape[0], -1, self.block_size // 2 + 1, 2)
        # Compute the spectral mask
        H = torch.rfft(h_w, 1).reshape(z.shape[0], -1, self.block_size // 2 + 1, 2)
        # Filter the original noise
        S_filtered          = torch.zeros_like(H)
        S_filtered[:,:,:,0] = H[:,:,:,0] * S_sig[:,:,:,0] - H[:,:,:,1] * S_sig[:,:,:,1]
        S_filtered[:,:,:,1] = H[:,:,:,0] * S_sig[:,:,:,1] + H[:,:,:,1] * S_sig[:,:,:,0]
        S_filtered          = S_filtered.reshape(-1, self.block_size // 2 + 1, 2)
        # Inverse the spectral noise back to signal
        filtered_noise = torch.irfft(S_filtered, 1)[:,:self.block_size].reshape(z.shape[0], -1)
        return filtered_noise

