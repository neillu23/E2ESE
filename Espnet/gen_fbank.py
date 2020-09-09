import librosa
import os
import sys
import numpy as np
import scipy
import torch
from tqdm import tqdm
from fbank import spec2fbank

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
clean_list= sys.argv[1] #"/home/neillu/End2End/TIMIT_filelist.txt"
out_file= sys.argv[2] #"/mnt/Data/user_vol_2/user_neillu/TIMIT_fbank/timit_fbank.txt"


def check_path(path):
    if not os.path.isdir(path): 
        os.makedirs(path)
        
def check_folder(path):
    path_n = '/'.join(path.split('/')[:-1])
    check_path(path_n)

def make_spectrum(filename=None, y=None, is_slice=False, feature_type='logmag', mode=None, FRAMELENGTH=None, SHIFT=None, _max=None, _min=None):
    if y is not None:
        y = y
    else:
        y, sr = librosa.load(filename, sr=16000)
        if sr != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        if y.dtype == 'int16':
            y = np.float32(y/32767.)
        elif y.dtype !='float32':
            y = np.float32(y)

    ### Normalize waveform
    y = y / np.max(abs(y)) #/ 2.

    #[Neil] modify the frame size
    #D = librosa.stft(y,center=False, n_fft=512, hop_length=160,win_length=512,window=scipy.signal.hamming)
    D = librosa.stft(y,center=False, n_fft=400, hop_length=160,win_length=400,window=scipy.signal.hamming)
    utt_len = D.shape[-1]
    phase = np.exp(1j * np.angle(D))
    D = np.abs(D)

    ### Feature type
    if feature_type == 'logmag':
        Sxx = np.log1p(D)
    elif feature_type == 'lps':
        Sxx = np.log10(D**2)
    else:
        Sxx = D

    if mode == 'mean_std':
        mean = np.mean(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))
        std = np.std(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))+1e-12
        Sxx = (Sxx-mean)/std  
    elif mode == 'minmax':
        Sxx = 2 * (Sxx - _min)/(_max - _min) - 1

    return Sxx, phase, len(y)

def write_txt(out_file,array):
    for i in range(array.shape[0]-1):
        for j in range(array.shape[1]-1):
            f.write(str(array[i][j]))
            f.write(" ")
        f.write(str(array[i][-1]))
        f.write("\n")
    for j in range(array.shape[1]-1):
        f.write(str(array[-1][j]))
        f.write(" ")
    f.write(str(array[-1][-1]))
    f.write(" ]\n")


if __name__ == '__main__':
    c_files = np.array([x[:-1] for x in open(clean_list).readlines()])
    check_folder(out_file)
    os.system("rm {}".format(out_file))
    for i,c_ in enumerate(tqdm(c_files)):
        name = c_.split('/')[-2] + '_' + c_.split('/')[-1].split('.')[0]
        c_wav,c_sr = librosa.load(c_,sr=16000)
        c_data,_,_ = make_spectrum(y= c_wav)
        c_data = torch.transpose(torch.from_numpy(c_data),0,1)
        filter_banks = spec2fbank(c_data).detach().numpy() 
        with open(out_file, 'a') as f:
            f.write(name+"  [\n")
            write_txt(f,filter_banks)
            # break
    
