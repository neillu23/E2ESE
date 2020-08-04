import librosa
import os
import numpy as np
import scipy
import torch
from tqdm import tqdm
# import pdb, mkl
from util import check_folder


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

clean_list="/home/neillu/Desktop/Workspace/yo/End2End/PytorchSE/secode/TIMIT_filelist.txt"
noisy_list="/home/neillu/Desktop/Workspace/yo/End2End/PytorchSE/secode/TIMIT_noisy_40hr_wav_filelist.txt"
out_path="/home/neillu/Desktop/Workspace/TIMIT_spec/"
#dev_list = "/mnt/Data/user_vol_2/user_neillu/DNS_Challenge_timit/dv_timit"
#test_list = "/mnt/Data/user_vol_2/user_neillu/DNS_Challenge_timit/ts_timit_new_all"

epsilon = np.finfo(float).eps
def get_filepaths(directory,ftype='.wav'):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ftype):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.

    return sorted(file_paths)

def normalize(metrix):
    # pdb.set_trace()
    m_mean = np.mean(metrix, axis=-1).reshape(metrix.shape[0],1)
    m_std = np.std(metrix, axis=-1).reshape(metrix.shape[0],1)+1e-12
    metrix_n = (metrix-m_mean)/m_std

    return metrix_n

def wav_to_spec(x,Norm=False,log=False, padding=False):

    length_wav = len(x)
    F = librosa.stft(x,n_fft=512,hop_length=160,win_length=512,window=scipy.signal.hamming)
    Lp_spec=np.abs(F)
    phase_spec=np.angle(F)
    if Norm:
        if log:
            Lp_spec = normalize(np.log10((Lp_spec+epsilon)**2))
        else:
            Lp_spec = normalize(Lp_spec**2)
    else:
        if log:
            Lp_spec = np.log10((Lp_spec+epsilon)**2)
        else:
            Lp_spec = Lp_spec**2
    if padding:
#         print(Lp_spec.dtype)
        Lp_pad = np.zeros((Lp_spec.shape[0]*3,Lp_spec.shape[1]),dtype=np.float32)
#         Lp_pad[:Lp_spec.shape[0],2:] = Lp_spec[:,:-2]
        Lp_pad[:Lp_spec.shape[0],1:] = Lp_spec[:,:-1]
        Lp_pad[Lp_spec.shape[0]:Lp_spec.shape[0]*2] = Lp_spec
        Lp_pad[Lp_spec.shape[0]*2:,:-1] = Lp_spec[:,1:]
#         Lp_pad[Lp_spec.shape[0]*4:Lp_spec.shape[0]*5,:-2] = Lp_spec[:,2:]
        Lp_spec = Lp_pad
        
    return Lp_spec, phase_spec, length_wav

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
    # y = y / np.max(abs(y)) / 2.
    
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


def check_dir(path):
    if not os.path.isdir('/'.join(list(path.split('/')[:-1]))):
        os.makedirs('/'.join(list(path.split('/')[:-1])))

if __name__ == '__main__':
    n_frame = 64
    n_files = np.array([x[:-1] for x in open(noisy_list).readlines()])
    c_files = np.array([x[:-1] for x in open(clean_list).readlines()])
    n_ptfiles =[]
    c_dict={}
    for i,c_ in enumerate(tqdm(c_files)):
        c_tmp=c_.replace('.WAV','').split('/')
        c_dict[c_tmp[-2]+'_'+c_tmp[-1]]=c_.replace(c_tmp[-2]+'/'+c_tmp[-1]+'.WAV','')
        
    
    
    for i,n_ in enumerate(tqdm(n_files)):
        ### use noisy filename to find clean file
        name = n_.split('/')[-1].split('_')[0] + '_' + n_.split('/')[-1].split('_')[1]
        n_folder = n_.split('/')[-3]+ '/' + n_.split('/')[-2]
        name=name.replace('.wav','')
        c_fn=name.split('_')[0]+'/'+name.split('_')[1]+'.WAV'
        out_name_c = os.path.join(out_path+'clean/', name+'.pt')
        out_name_n = out_name_c.replace('clean','noisy/'+n_folder)
        
        if name in c_dict:
            c_folder=c_dict.pop(name)
            c_ = os.path.join(c_folder, c_fn)
            c_wav,c_sr = librosa.load(c_,sr=16000)
            c_data,_,_ = make_spectrum(y= c_wav)
            
            check_folder(out_name_c)
            torch.save(torch.from_numpy(c_data.transpose()),out_name_c)

        n_wav,sr = librosa.load(n_,sr=16000) 
        n_data,_,_ = make_spectrum(y= n_wav)

        check_folder(out_name_n)
        torch.save(torch.from_numpy(n_data.transpose()),out_name_n)
        n_ptfiles.append(out_name_n)

    with open('./TIMIT_noisy_40hr_spec_filelist.txt', 'w') as f:
        for item in n_ptfiles:
            f.write('%s\n' % item)
    
