import os
import numpy as np

def getfilename(folder,fnlist):
    for dirpath, _, files in os.walk(folder):
        for file_name in files:
            if file_name.endswith(".wav"):
                fnlist.append(os.path.join(dirpath, file_name))
                
    
    print('folder:',folder,', len:',len(fnlist))
        

if __name__ == '__main__':

    corpusfolder='/home/neillu/Desktop/Workspace/TIMIT_noisy_40hr_wav'
    filelist=[]
    getfilename(corpusfolder,filelist)

    with open('TIMIT_noisy_40hr_wav_filelist.txt', 'w') as f:
        for listitem in filelist:
            f.write('%s\n' % listitem)
    



    