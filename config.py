import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SEQUENCE_LEN = 200
BLOCK_SIZE = 160
SR = 16000
KERNEL_SIZE = 15

