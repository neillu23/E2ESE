import os
import numpy as np

def getfilename(folder,fnlist):
    for dirpath, _, files in os.walk(folder):
        for file_name in files:
            if file_name.endswith(".wav") or file_name.endswith(".WAV"):
                fnlist.append(os.path.join(dirpath, file_name))
                
    
    print('folder:',folder,', len:',len(fnlist))
        

if __name__ == '__main__':

    #corpusfolder='/home/neillu/Downloads/TIMIT_Manner_E2E/TEST/'
    corpusfolder='/home/neillu/Downloads/NewTsNoisy/'
    filelist=[]
    getfilename(corpusfolder,filelist)

    #outpath='../data/TIMIT_Manner_E2E_wav_filelist.txt'
    outpath='../data/TIMIT_test_noisy_wav_filelist.txt'

    if os.path.exists(outpath):
            os.remove(outpath)

    with open(outpath, 'w') as f:
        for listitem in filelist:
            f.write('%s\n' % listitem)
    



    