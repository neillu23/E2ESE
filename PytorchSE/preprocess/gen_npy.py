import librosa
import os, argparse
import numpy as np
import scipy
import torch
from tqdm import tqdm
# import pdb, mkl

import sys

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from secode.utils.util import check_folder, make_spectrum, getfilename, get_cleanwav_dic

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


epsilon = np.finfo(float).eps

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--TMHINT' , action='store_true')
    parser.add_argument('--data', type=str, default="trdata")
    
    # Input wave file list
    parser.add_argument('--clean_wav_path', type=str, default='')
    parser.add_argument('--noisy_wav_path', type=str, default='')
    
    # Output spectrum path
    parser.add_argument('--out_path', type=str, default='')
    args = parser.parse_args()
    return args

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


def check_dir(path):
    if not os.path.isdir('/'.join(list(path.split('/')[:-1]))):
        os.makedirs('/'.join(list(path.split('/')[:-1])))

if __name__ == '__main__':

    # get parameter
    args = get_args()
    spec_path = os.path.join(args.out_path,"spec/")
    
    if args.data=='trdata':
        spec_path = os.path.join(spec_path,"train/")
        #args.spec_path+='train/'
        #check_folder(os.path.join(os.getcwd(),'data/train/'))
        #noisy_spec_list='data/train/noisy_spec_filelist.txt'
        #c_wavfolder_dic='data/train/c_wavfolder_dic.npy'
        #noisy_wav_list='data/train/noisy_wav_filelist.txt'
    else: 
        spec_path = os.path.join(spec_path,"test/")
        #args.spec_path+='test/'
        #check_folder(os.path.join(os.getcwd(),'data/test/'))
        #noisy_spec_list='data/test/noisy_spec_filelist.txt'
        #c_wavfolder_dic='data/test/c_wavfolder_dic.npy'
        #noisy_wav_list='data/test/noisy_wav_filelist.txt'


    noisy_wav=getfilename(args.noisy_wav_path)
    clean_wav=getfilename(args.clean_wav_path)

    n_frame = 64
    n_files = np.array(noisy_wav)
    c_files = np.array(clean_wav)

    '''
    if os.path.exists(noisy_wav_list):
            os.remove(noisy_wav_list)

    with open(noisy_wav_list, 'w') as f:
        for item in noisy_wav:
            f.write('%s\n' % item)
    '''



    #n_ptfiles =[]
    

    c_dict={}
    if args.TMHINT:
        for i,c_ in enumerate(tqdm(c_files)):
            k=c_.replace('.wav','').split('/')[-1]
            c_path=c_.replace(k+'.wav','')
            c_dict[k]=c_path
    else:
        for i,c_ in enumerate(tqdm(c_files)):
            c_tmp=c_.replace('.WAV','').split('/')
            k=c_tmp[-2]+'_'+c_tmp[-1]
            c_path=c_.replace(c_tmp[-2]+'/'+c_tmp[-1]+'.WAV','')
            c_dict[k]=c_path
        
        
    '''
    np.save(c_wavfolder_dic, c_dict) 
    
    if os.path.exists(args.spec_path):
            os.rmdir(args.spec_path)
    '''
    for i,n_ in enumerate(tqdm(n_files)):
        if args.TMHINT:
            ### use noisy filename to find clean file
            name = n_.split('/')[-1].replace('.wav','')
            n_folder = n_.split('/')[-2]
            c_fn=name + '.wav'
        else:
            ### use noisy filename to find clean file
            name = n_.split('/')[-1].split('_')[0] + '_' + n_.split('/')[-1].split('_')[1]
            n_folder = n_.split('/')[-3]+ '/' + n_.split('/')[-2]
            name=name.replace('.wav','')
            c_fn=name.split('_')[0]+'/'+name.split('_')[1]+'.WAV'
        out_name_c = os.path.join(spec_path+'clean/', name+'.pt')
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
        #n_ptfiles.append(out_name_n)

    '''
    if os.path.exists(noisy_spec_list):
        os.remove(noisy_spec_list)
    with open(noisy_spec_list, 'w') as f:
        for item in n_ptfiles:
            f.write('%s\n' % item)
    '''
    
