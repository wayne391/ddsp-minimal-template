import torch
import torch.nn as nn

from ddsp.oscillators import HarmonicOscillators
from ddsp.generators import FilteredNoise
from ddsp.effects import Reverb


class HarmonicSynth(nn.Module):
    def __init__(self, args):
        super(HarmonicSynth, self).__init__()
        print(' >', self.__class__.__name__)
        self.harmonic = HarmonicOscillators(args.n_partial, args.sr, args.block_size)
        self.noise = FilteredNoise(args.filter_size, args.block_size)
        self.reverb = Reverb(args)
    
    def forward(self, params, conditions):
        amp, alpha, filter_coeff, reverb = params
        f0, loud = conditions
    
        x_harmonic = self.harmonic(amp, alpha, f0)
        x_noise = self.noise(x_harmonic, filter_coeff)
        x_dry = x_harmonic + x_noise
        # x_final = self.reverb(x_dry)
        return x_dry, (x_harmonic, x_noise, x_dry)
