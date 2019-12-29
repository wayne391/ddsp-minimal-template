import torch
import torch.nn as nn
import numpy as np


class AESynth(nn.Module):
    def __init__(self, encoder, decoder, encoder_dims, latent_dims):
        super(AESynth, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dims = latent_dims
        self.encoder_dims = encoder_dims

        # Latent gaussians
        self.mu = nn.Linear(encoder_dims, latent_dims)
        self.log_var = nn.Sequential(
            nn.Linear(encoder_dims, latent_dims))

        # initialize
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
            

    def forward(self, x, conditions):
        # run enc
        z = self.encoder(x)
        # run dec
        x_rec, params, tensors = self.decoder(z, conditions)
        return x_rec, params, tensors
    

def mod_sigmoid(x):
    """
    Implementation of the modified sigmoid described in the original article.
    Arguments :
        x (Tensor)      : input tensor, of any shape
    Returns:
        Tensor          : output tensor, same shape of x
    """
    return 2*torch.sigmoid(x)**np.log(10) + 1e-7

class MLP(nn.Module):
    """
    Implementation of the MLP, as described in the original paper
    Parameters :
        in_size (int)   : input size of the MLP
        out_size (int)  : output size of the MLP
        loop (int)      : number of repetition of Linear-Norm-ReLU
    """
    def __init__(self, in_size=512, out_size=512, loop=3):
        super().__init__()
        self.linear = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_size, out_size),
                nn.modules.normalization.LayerNorm(out_size),
                nn.ReLU()
            )] + [nn.Sequential(nn.Linear(out_size, out_size),
                nn.modules.normalization.LayerNorm(out_size),
                nn.ReLU()
            ) for i in range(loop - 1)])

    def forward(self, x):
        # print('[MLP] in', x.shape)
        for lin in self.linear:
            x = lin(x)
            # print('[MLP] x', x.shape)
        return x

class Encoder(nn.Module):
    """
    Raw waveform encoder, based on VQVAE
    """
    def __init__(self, args):
        super().__init__()
        self.out_size = args.encoder_dims
        self.convs = nn.ModuleList(
            [nn.Conv1d(1, args.channels, args.kernel_size,
                        padding=args.kernel_size // 2,
                        stride=args.strides[0])]
            + [nn.Conv1d(args.channels, args.channels, args.kernel_size,
                         padding=args.kernel_size // 2,
                         stride=args.strides[i]) for i in range(1, len(args.strides) - 1)]
            + [nn.Conv1d(args.channels, args.encoder_dims, args.kernel_size,
                         padding=args.kernel_size // 2,
                         stride=args.strides[-1])])

    def forward(self, x):
        # print('in x:', x.shape)
        for i,conv in enumerate(self.convs):
            x = conv(x)
            # print(i, 'x:', x.shape)
            if i != len(self.convs)-1:
                x = torch.relu(x)
        return x


class Decoder(nn.Module):
    """
    Decoder with the architecture originally described in the DDSP paper

    Parameters:
        hidden_size (int)       : Size of vectors inside every MLP + GRU + Dense
        n_partial (int)         : Number of partial involved in the harmonic generation. (>1)
        filter_size (int)       : Size of the filter used to shape noise.
    """
    def __init__(self, args, synth):
        super().__init__()
        # Map the different conditions
        self.f0_MLP = MLP(1,args.n_hidden)
        self.lo_MLP = MLP(1,args.n_hidden)
        # Map the latent vector
        self.z_MLP  = MLP(args.latent_dims, args.n_hidden)
        # Recurrent model to handle temporality
        self.gru    = nn.GRU(3 * args.n_hidden, args.n_hidden, batch_first=True)
        # Mixing MLP after the GRU
        self.fi_MLP = MLP(args.n_hidden, args.n_hidden)
        # Outputs to different parameters of the synth
        self.dense_amp    = nn.Linear(args.n_hidden, 1)
        self.dense_alpha  = nn.Linear(args.n_hidden, args.n_partial)
        self.dense_filter = nn.Linear(args.n_hidden, args.filter_size // 2 + 1)
        self.dense_reverb = nn.Linear(args.n_hidden, 2)
        self.n_partial = args.n_partial
        self.synth = synth

    def forward(self, z, conditions, hx=None):
        f0, lo = conditions

        # Forward pass for the encoding
        z = z.transpose(1, 2)
        f0 = self.f0_MLP(f0)
        lo = self.lo_MLP(lo)
        z  = self.z_MLP(z)
        # Recurrent model
        x, h = self.gru(torch.cat([z, f0, lo], -1), hx)
        # Mixing parameters
        x = self.fi_MLP(x)
        # Retrieve various parameters
        amp          = mod_sigmoid(self.dense_amp(x))
        alpha        = mod_sigmoid(self.dense_alpha(x))
        filter_coeff = mod_sigmoid(self.dense_filter(x))
        reverb       = self.dense_reverb(x)
        alpha        = alpha / torch.sum(alpha,-1).unsqueeze(-1)
        # Return the set of parameters

        # --- synth --- #
        params = (amp, alpha, filter_coeff, reverb)
        x_final, tensor = self.synth(params, conditions)
        return x_final, params, tensor
