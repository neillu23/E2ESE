import librosa,os
import numpy as np
from pypesq import pesq
from pystoi.stoi import stoi
import scipy
import tqdm
import random
import pdb

epsilon = np.finfo(float).eps



def getfilename(folder, mode=None):
    fnlist=[]
    for dirpath, _, files in os.walk(folder):
        for file_name in files:
            if file_name.endswith(".wav") or file_name.endswith(".WAV") or file_name.endswith(".pt"):
                fnlist.append(os.path.join(dirpath, file_name))
    print('folder:',folder,', len:',len(fnlist))
    fnlist=sorted(fnlist)
    random.shuffle(fnlist)
    if mode != None:
        list_name="./"+str(mode)+"_file_list.txt"
        with open(list_name, 'w') as filehandle:
            for listitem in fnlist:
                filehandle.write('%s\n' % listitem)
    return fnlist


def asr2clean(asr_name, corpus="TIMIT"):
    if corpus == "TIMIT":
        c_name = asr_name
    elif corpus == "TMHINT":
        speaker=["M1","M2","M3","F1","F2","F3"]
        tmp=asr_name.split("_")
        if tmp[0] in speaker:
            if int(tmp[2])>=13:
                c_name=str(int(speaker.index(tmp[0]))*200+(int(tmp[2])-13)*10+int(tmp[3]))
            else:
                c_name = None
        else:
            c_name = "_".join(tmp[1:])
    elif corpus == "TMHINT_DYS":
        num=int(asr_name.split("_")[-1])
        c_name=asr_name.split("_")[-2]+"_%02d%02d"%(((num-1)/10)+1,(num-1)%10+1)

    return c_name

def noisy2clean_train(n_file, corpus="TIMIT"):
    # for spectrum .pt file
    if corpus == "TIMIT":
        n_folder = '/'.join(n_file.split('/')[-4:-1])
    elif corpus == "TMHINT":
        n_folder = '/'.join(n_file.split('/')[-3:-1])
    c_file = n_file.replace(n_folder,"clean")
    c_name = n_file.split('/')[-1].replace('.pt','')
    return c_file, c_name

def get_cleanwav_dic(clean_wav_path, corpus="TIMIT"):
    print('get clean wav path:', clean_wav_path)
    clean_wav=getfilename(clean_wav_path)
    c_files = np.array(clean_wav)
    c_dict={}
    ### use clean test filename to find clean filepath
    if corpus == "TIMIT" or corpus == "TMHINT":
        for c_ in c_files:
            c_name='/'.join(c_.split('/')[-2:])
            c_path=c_.replace(c_name,'')
            c_dict[c_name]=c_path
    return c_dict
    
def noisy2clean_test(test_file, c_dict, corpus="TIMIT"):
    # for waveform .wav file
    if corpus == "TIMIT":
        n_folder = '/'.join(test_file.split('/')[-4:-1])
        c_name = '/'.join([test_file.split('/')[-1].split('_')[0], test_file.split('/')[-1].split('_')[1]])
        c_name = c_name.replace('.wav','.WAV')
    elif corpus == "TMHINT":
        n_folder = '/'.join(test_file.split('/')[-3:-1])
        c_name = '/'.join([test_file.split('/')[-3], test_file.split('/')[-1]])
    # elif corpus == "TMHINT_DYS":
    #     name = test_file.split('/')[-1]+'/'+test_file.split('/')[-1].replace('.wav','')
    #     if test_file.split('/')[-2] == "PairedOne":
    #     n_folder = '/'.join(test_file.split('/')[-3:-1])
    #     c_name = name + '.wav'
    c_folder = c_dict[c_name]
    c_file = os.path.join(c_folder, c_name)
    return c_file, n_folder




def check_path(path):
    if not os.path.isdir(path): 
        os.makedirs(path)
        
def check_folder(path):
    path_n = '/'.join(path.split('/')[:-1])
    check_path(path_n)

def cal_score(clean,enhanced):
    clean = clean/abs(clean).max()
    enhanced = enhanced/abs(enhanced).max()

    s_stoi = stoi(clean, enhanced, 16000)
    s_pesq = pesq(clean, enhanced, 16000)
    
    return round(s_pesq,5), round(s_stoi,5)


def make_spectrum(filename=None, y=None, is_slice=False, feature_type='logmag', mode=None, FRAMELENGTH=400, SHIFT=160, _max=None, _min=None):
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
    y = y / np.max(abs(y)) # / 2.

    D = librosa.stft(y,center=False, n_fft=FRAMELENGTH, hop_length=SHIFT,win_length=FRAMELENGTH,window=scipy.signal.hamming)
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

def recons_spec_phase(Sxx_r, phase, length_wav, feature_type='logmag'):
    if feature_type == 'logmag':
        Sxx_r = np.expm1(Sxx_r)
        if np.min(Sxx_r) < 0:
            print("Expm1 < 0 !!")
        Sxx_r = np.clip(Sxx_r, a_min=0., a_max=None)
    elif feature_type == 'lps':
        Sxx_r = np.sqrt(10**Sxx_r)

    R = np.multiply(Sxx_r , phase)
    result = librosa.istft(R,
                     center=False,
                     hop_length=160,
                     win_length=400,
                     window=scipy.signal.hamming,
                     length=length_wav)
    return result
