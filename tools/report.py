import os
import re
import time
import numpy as np

import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

import librosa
import librosa.display

from tools import utils


def plot_curve(curve, filename=None, title=''):
#     plt.figure()
    plt.title(title)
    plt.plot(curve)
    if filename:
        plt.savefig(filename)
        plt.close()
    
    
def plot_waveform(wav, sr, filename=None, title=''):
#     plt.figure()
    librosa.display.waveplot(wav, sr=sr)
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("amplitude")
    if filename:
        plt.savefig(filename)
        plt.close()
        
      
def plot_spec(wav, filename=None, title=''):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(wav, n_fft=512)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear', x_axis='frames')
    # plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    if filename:
        plt.savefig(filename)
        plt.close()
        

def plot_runtime_info(x, x_rec, sr, conditions, params, tensors, path_fig, counter=None): 
    # unpack
    f0, loud = conditions
    amp, alpha = params
    x_harmonic, x_noise, x_dry = tensors

    # convert to np
    x, x_rec, f0, loud, x_harmonic, x_noise, x_dry = [utils.convert_tensor_to_numpy(tensor) \
                for tensor in [x, x_rec, f0, loud, x_harmonic, x_noise, x_dry]]
    amp = utils.convert_tensor_to_numpy(amp)
    alpha = utils.convert_tensor_to_numpy(alpha)

    if counter is not None:
        path_fig = os.path.join(path_fig, str(counter))
        if not os.path.exists(path_fig):
            os.makedirs(path_fig)

    # --- plot conditions --- #
    path_codition = os.path.join(path_fig, 'conditions')
    plt.figure(dpi=300)
    plt.title('conditions')
    
    # f0
    plt.subplot(2,1,1)
    plot_curve(f0, title='f0')
    
    # loudness
    plt.subplot(2,1,2)
    plot_curve(loud, title='loudness')

    # save
    plt.tight_layout()
    plt.savefig(path_codition)
    plt.close()

    # --- plot param --- #
    path_param = os.path.join(path_fig, 'harmonic_osc')
    plt.figure(dpi=300)
    plt.title('param')

    # plot f0
    plt.subplot(4,1,1)
    plot_curve(f0, title='f0')

    # plot amp
    plt.subplot(4,1,2)
    plot_curve(amp, title='amp')
    
    # plot alpha
    plt.subplot(4,1,3)
    plt.title('alpha')
    plt.imshow(alpha.T, aspect="auto", cmap='magma')

    # plot alpha
    plt.subplot(4,1,4)
    plot_waveform(x_harmonic, sr, title='harmonic osc')

    # save
    plt.tight_layout()
    plt.savefig(path_param)
    plt.close()

    # --- plot waves --- #
    # ndarry, title, is_save_wav
    toplot_wav_list = [
        (x, 'original', True),
        (x_rec, 'est_resynth', True), 
        (x_harmonic, 'est_harmonic', True),
        (x_noise , 'est_noise', True),
        (x_dry, 'est_dry', True)
    ]
    
    # plot
    for wav, title, is_save_wav in toplot_wav_list:
        # new figure
        path_wav = os.path.join(path_fig, title)
        plt.figure(dpi=300)
        plt.title(title)

        # wavform
        plt.subplot(2,1,1)
        plot_waveform(wav, sr, title='waveform')
        
        # spec
        plt.subplot(2,1,2)
        plot_spec(wav, title='spectorgram')

        # save
        plt.tight_layout()
        plt.savefig(path_wav)
        plt.close()

        # save wav
        if is_save_wav:
            path_wav = os.path.join(path_fig, title+'.wav')
            librosa.output.write_wav(path_wav, wav, sr, norm=False)


def make_loss_report(
        path_log,
        path_figure='loss.png',
        dpi=300, 
        x_range=None,
        y_range=None):
    
    # init
    list_loss = []
    list_loss_step = []
    training_time = ''
    
    # collect info
    with open(path_log) as f:
        for line in f:
            line = line.strip()
            if 'epoch' in line:
                loss = float(re.findall("loss: \d+\.\d+", line)[0][len('loss: '):])
                counter = int(re.findall("iter: \d+", line)[0][len('iter: '):])
                training_time = re.findall("time: \d+:\d+\:\d+.\d+", line)[0][len('time: '):]
                list_loss.append(loss)
                list_loss_step.append(counter)

    # plot
    fig = plt.figure(dpi=dpi)
    plt.title('training process')
    plt.plot(list_loss_step, list_loss, label='train')
    plt.legend(loc='upper right')
    if x_range:  
        plt.xlim(x_range[0], x_range[1])
    if y_range:  
        plt.xlim(y_range[0], y_range[1])
    txt = 'time: {}\ncounter: {}'.format(
            training_time,
            list_loss_step[-1]
        )
    fig.text(.5, -.05, txt, ha='center')
    plt.tight_layout()
    plt.savefig(path_figure)