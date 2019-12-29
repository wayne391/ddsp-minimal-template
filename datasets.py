'''
audio effect
'''

import os
import sys
import glob
import json
import collections
import numpy as np

import librosa
import soundfile as sf
# import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

from config import SEQUENCE_LEN, BLOCK_SIZE


class AudioEffectDataset(Dataset):
    def __init__(self, path_feats, device):
        print('\n - {building dataset} - ')
        self.device = device
        if path_feats.endswith('.npz'):
            print('[o] load compiled npz from', path_feats)
            loaded = np.load(path_feats)
            self.list_y = loaded['list_y']
            self.list_f0 = loaded['list_f0']
            self.list_lo = loaded['list_lo']
            self.num_file = len(self.list_y)
            print('num of samples:', self.num_file)
            return 

        list_npz = glob.glob(os.path.join(path_feats, '*.npz'))

        list_y = []
        list_f0 = []
        list_lo = []

        print('[*] loading feats from', path_feats)
        for idx in range(len(list_npz)):
            print('\n---', idx)
            file = list_npz[idx]
            loaded = np.load(file)
            
            # laod
            audio = loaded['audio']
            loudness = loaded['loudness']
            f0 = loaded['f0']

            # reshape audio (batch x samples)
            samples = BLOCK_SIZE * SEQUENCE_LEN
            audio_re = audio.reshape(-1, samples)

            # reshape features (batch x frames)
            num_batch = audio_re.shape[0]
            frames = num_batch * SEQUENCE_LEN
            loudness_re = loudness[:frames].reshape(-1, SEQUENCE_LEN)
            f0_re = f0[:frames].reshape(-1, SEQUENCE_LEN)

            # print
            print('    audio:   ', audio_re.shape)
            print('    loudness:', loudness_re.shape)
            print('    f0:      ', f0_re.shape)
            
            # append
            list_y.append(audio_re)
            list_f0.append(f0_re)
            list_lo.append(loudness_re)
        
        # compile 
        self.list_y = np.concatenate(list_y, axis=0)
        self.list_f0 = np.concatenate(list_f0, axis=0)
        self.list_lo = np.concatenate(list_lo, axis=0)

        print('audio:   ', self.list_y.shape)
        print('f0:      ', self.list_f0.shape)
        print('loudness:', self.list_lo.shape)

        # save
        path_compiled = os.path.join(path_feats, 'compile.npz')
        self.num_file = len(self.list_y)
        np.savez(
                path_compiled, 
                list_y=self.list_y,
                list_f0=self.list_f0,
                list_lo=self.list_lo)
        print('num of samples:', self.num_file)


    def __getitem__(self, index):
        result = {
            'audio': torch.from_numpy(self.list_y[index]).float().unsqueeze(0),
            'f0': torch.from_numpy(self.list_f0[index]).float().unsqueeze(0),
            'lo': torch.from_numpy(self.list_lo[index]).float().unsqueeze(0)
        }
        return result

    def __len__(self):
        return self.num_file


if __name__ == '__main__':
    # config - input
    path_feats = 'violin/feats'
    dataset = AudioEffectDataset(path_feats)
    # path_npz = 'violin/feats/compile.npz'
    # dataset = AudioEffectDataset(path_npz)

    train_loader = DataLoader(
        dataset, 
        batch_size=20, 
        shuffle=True)  


    # get one batch
    batch = next(iter(train_loader))
    print(batch['audio'].shape)
    print(batch['f0'].shape)
    print(batch['lo'].shape)