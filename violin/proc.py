import os
import glob
import crepe
import librosa
import numpy as np

import datetime

from ddsp.analysis import Loudness
from config import SEQUENCE_LEN, BLOCK_SIZE, SR, KERNEL_SIZE
import soundfile as sf

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def extract_feature(dir_audio, dir_feat, ext='wav', thres=0.2):
    extractor_lo = Loudness(BLOCK_SIZE, KERNEL_SIZE).float()
    list_wav = glob.glob(os.path.join(dir_audio, '*.' + ext))
    
    total_time = 0.0
    for idx, wav_file in enumerate(list_wav):
        print(' = [{}] ==='.format(idx))
        print('file:', wav_file)
        # y, sr = sf.read(wav_file)
        #assert sr == SR
        y, sr = librosa.load(wav_file, sr=SR)
        fname, _ = os.path.splitext(os.path.basename(wav_file))
        
        mod = BLOCK_SIZE * SEQUENCE_LEN
        y = y[:mod*(len(y)//mod)]
        total_time += len(y) / sr

        print('[*] extracting f0 ...')
        hop = int(1000 * BLOCK_SIZE / SR)
        _, freq, conf, _ = crepe.predict(y, SR, step_size=hop, verbose=False)  # time, freq, conf, act
        freq_thres = np.copy(freq)
        freq_thres[conf<thres] = 0
        print(' > f0:', freq_thres.shape)
        # np.save(os.path.join(dir_feat, fname + '_f0.npy'), freq_thres)

        print('[*] extracting loudness ...')
        lo = extractor_lo(y)
        print(' > lo:', lo.shape)
        # np.save(os.path.join(dir_feat, fname + '_lo.npy'), lo)

        np.savez(os.path.join(dir_feat, fname + '.npz'),
            audio=y,
            loudness=lo,
            f0=freq_thres)

    print('===')
    print('total_time:', str(datetime.timedelta(seconds=total_time))+'\n')


if __name__ == '__main__':
    path_wavdir = 'instr_datasets/guitar_synth'
    path_featsdir = 'instr_datasets/guitar_synth/feats'
    if not os.path.exists(path_featsdir):
        os.makedirs(path_featsdir)
    print('input dir:', path_wavdir)
    tf.config.set_soft_device_placement(True)
    with tf.device('/gpu:0'):
        extract_feature(path_wavdir, path_featsdir)
