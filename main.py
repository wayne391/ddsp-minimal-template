import matplotlib
#matplotlib.use('agg')
import os

import solver
import argparse
import torch


GPU_ID = 0
exp_memo = "Hello world"

# Define arguments
print('\n\n')
parser = argparse.ArgumentParser()

# exp
parser.add_argument('--exp_dir', default='exp', help='datasave directory')
parser.add_argument('--exp_memo', default=exp_memo, help='datasave directory')

# Preprocessing arguments
parser.add_argument('--sr',             type=int,   default=16000,          help='Sample rate of the signal')
parser.add_argument('--fft_scales',     type=list,  default=[64, 6],        help='Minimum and number of scales')
parser.add_argument('--smooth_kernel',  type=int,   default=8,              help='Size of the smoothing kernel')

# DDSP parameters
parser.add_argument('--n_partial',      type=int,   default=50,             help='Number of partials')
parser.add_argument('--filter_size',    type=int,   default=64,             help='Size of the filter')
parser.add_argument('--block_size',     type=int,   default=160,            help='Number of samples in blocks')
parser.add_argument('--kernel_size',    type=int,   default=15,             help='Size of the kernel')
parser.add_argument('--sequence_size',  type=int,   default=200,            help='Size of the sequence')

# Model arguments
parser.add_argument('--layers',         type=str,   default='gru',          help='Type of layers in the model')
parser.add_argument('--strides',        type=list,  default=[2,4,4,5],      help='Set of processing strides')
parser.add_argument('--n_hidden',       type=int,   default=512,            help='Number of hidden units')
parser.add_argument('--n_layers',       type=int,   default=4,              help='Number of computing layers')
parser.add_argument('--channels',       type=int,   default=128,            help='Number of channels in convolution')
parser.add_argument('--kernel',         type=int,   default=15,             help='Size of convolution kernel')
parser.add_argument('--encoder_dims',   type=int,   default=16,             help='Number of encoder output dimensions')
parser.add_argument('--latent_dims',    type=int,   default=16,             help='Number of latent dimensions')

# Optimization arguments
parser.add_argument('--early_stop',     type=int,   default=60,             help='Early stopping')
parser.add_argument('--train_type',     type=str,   default='random',       help='Fixed or random data split')
parser.add_argument('--batch_size',     type=int,   default=8,             help='Size of the batch')
parser.add_argument('--epochs',         type=int,   default=200,            help='Number of epochs to train on')
parser.add_argument('--lr',             type=float, default=2e-4,           help='Learning rate')

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_device(GPU_ID)
print('[device]: {}, id: {}'.format(device, GPU_ID))
parser.add_argument('--device',         type=str,   default=device,          help='Device')

# monitoring
parser.add_argument('--interval_print_loss',  type=int,   default=5,            help='Interval of plotting frequency')
parser.add_argument('--interval_plot',        type=int,   default=100,            help='Interval of plotting frequency')
parser.add_argument('--interval_save_model',  type=int,   default=500,            help='Interval of plotting frequency')

args = parser.parse_args()

# set scales
args.scales = []
for s in range(args.fft_scales[1]):
    args.scales.append(args.fft_scales[0] * (2 ** s))


def main():

    from model import AESynth, Encoder, Decoder
    from ddsp.synth import HarmonicSynth
    from ddsp.loss import MSSTFTLoss
    from datasets import AudioEffectDataset

    # model
    print('\n - {building model} - ')
    synth = HarmonicSynth(args)
    encoder = Encoder(args)
    decoder = Decoder(args, synth)
    vae_model = AESynth(encoder, decoder, args.encoder_dims, args.latent_dims)
    mlti_sclae_loss = MSSTFTLoss(args.scales)

    # data
    path_npz = '../datasets/violen/feats/compile.npz'
    dataset = AudioEffectDataset(path_npz, args.device)

    # train
    solver.train(args, vae_model, mlti_sclae_loss, dataset)


if __name__ == '__main__':
    main()
