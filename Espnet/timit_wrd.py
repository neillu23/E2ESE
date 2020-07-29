import os
import numpy as np
from tqdm import tqdm
import sys

bpc_dict = {"h#":"H","Stops":"T","Vowels":"V","Fricatives":"F","Vowels":"V","Nasals":"N","sil":"S"}

def write_txt(phn_file,txt_file,out_file):
    bpc_list = []
    with open(phn_file,"r") as pf:
        for line in pf.readlines():
            bpc = bpc_dict[line.split()[-1]]
            bpc_list.append(bpc)

    with open(txt_file,"r") as tf:
        line = tf.readlines()[0]
        start,end = line.split()[0:2]

    with open(out_file,"w") as of:
        out_string = start + " " + end + " "
        for bpc in bpc_list:
            out_string += bpc
        of.write(out_string)
    


def write_wrd(phn_file,out_file):
    wrd_list = []
    with open(phn_file,"r") as pf:
        for line in pf.readlines():
            start,end = line.split()[0:2]
            bpc = bpc_dict[line.split()[-1]]
            wrd_list.append([start,end,bpc])
    
    with open(out_file,"w") as of:
        for wrd in wrd_list:
            out_string = wrd[0] + " " + wrd[1] + " " + wrd[2] + "\n"
            of.write(out_string)

    

# phn_file = "/mnt/Data/user_vol_2/user_neillu/TIMIT_Manner_Clean/TRAIN/DR1/FCJF0/SI1027.PHN"
# txt_file = "/mnt/Data/user_vol_2/user_neillu/TIMIT_Manner_Clean/TRAIN/DR1/FCJF0/SI1027.TXT"
# o1 = "txt"
# o2 = "wrd"
# write_txt(phn_file,txt_file,o1)
# write_wrd(phn_file,o2)
for dirPath, dirNames, fileNames in os.walk("/mnt/Data/user_vol_2/user_neillu/TIMIT_Manner_E2E/"):
    for fn in fileNames:
        if ".PHN" in fn:
            phn_file = os.path.join(dirPath, fn)
            txt_file = os.path.join(dirPath, fn.replace("PHN","TXT"))
            wrd_file = os.path.join(dirPath, fn.replace("PHN","WRD"))
            write_txt(phn_file,txt_file,txt_file)
            write_wrd(phn_file,wrd_file)
