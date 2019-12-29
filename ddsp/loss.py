import torch
import torch.nn as nn

import numpy as np


amp = lambda x: x[:,:,:,0]**2 + x[:,:,:,1]**2

class MSSTFTLoss(nn.Module):
    def __init__(self, scales, overlap=0.75):
        super(MSSTFTLoss, self).__init__()
        self.scales = scales
        self.overlap = overlap
        self.windows = nn.ParameterList(nn.Parameter(torch.from_numpy(np.hanning(scale)).float(), requires_grad=False) for scale in self.scales)

    def forward(self, x, x_orig):
        # print('[Loss] x:', x.shape)
        # print('[Loss] x_orig:', x_orig.shape)

        stfts = []
        # First compute multiple STFT for x
        # print('loss x', x.shape)
        x = x[:, :x_orig.shape[-1]]
        # print('[Loss] x (crop):', x.shape)
        for i, scale in enumerate(self.scales):
            cur_fft = torch.stft(x, n_fft=scale, window=self.windows[i], hop_length=int((1-self.overlap)*scale), center=False)
            # print(cur_fft.shape)
            stfts.append(amp(cur_fft))

        # print('loss x orig', x_orig.shape)
        x_orig = x_orig.squeeze()
        # print('loss x orig', x_orig.shape)
        stfts_orig = []
        # First compute multiple STFT for x_orig
        for i, scale in enumerate(self.scales):
            cur_fft = torch.stft(x_orig, n_fft=scale, window=self.windows[i], hop_length=int((1-self.overlap)*scale), center=False)
            # print(cur_fft.shape)
            # print('amp:', amp(cur_fft).shape)
            stfts_orig.append(amp(cur_fft))

        # Compute loss scale x batch
        lin_loss = sum([torch.mean(abs(stfts_orig[i][j] - stfts[i][j])) for j in range(len(stfts[i])) for i in range(len(stfts))])
        log_loss = sum([torch.mean(abs(torch.log(stfts_orig[i][j] + 1e-4) - torch.log(stfts[i][j] + 1e-4)))  for j in range(len(stfts[i])) for i in range(len(stfts))])
        return lin_loss + log_loss
