import os
import sys
import csv
import numpy as np

result_file = sys.argv[1]
score_path = sys.argv[2]


pesq_score_list = {}
stoi_score_list = {}
wer_score_list = {}
clean_wer_score_list = {}
with open(result_file, newline='') as rf:
    r_lines = csv.reader(rf)
    for res in r_lines:
        if res[0] == "Filename" or res[0] == "Average":
            continue
        name = res[0]
        pesq = res[1]
        stoi = res[2]
        wer = res[3]
        clean_wer = res[4]
        key = name.split("/")[-2].replace("n","-").replace("dB","")
        if key not in pesq_score_list:
            pesq_score_list[key] = []
            stoi_score_list[key] = []
            wer_score_list[key] = []
            clean_wer_score_list[key] = []
        pesq_score_list[key].append(float(pesq))
        stoi_score_list[key].append(float(stoi))
        wer_score_list[key].append(float(wer))
        clean_wer_score_list[key].append(float(clean_wer))

key_list=sorted(pesq_score_list.keys())
if not os.path.isdir(score_path): 
    os.makedirs(score_path)
fp = open(os.path.join(score_path,result_file.split("/")[-1]),"w", newline='')
writer = csv.writer(fp)
writer.writerow(["SNR", "pesq_avg", "stoi_avg","wer_avg","clean_wer_avg"])
print("SNR, pesq_avg, stoi_avg, wer_avg, clean_wer_avg")
for snr in key_list:
    pesq_score = np.average(pesq_score_list[snr])
    stoi_score = np.average(stoi_score_list[snr])
    wer_score = np.average(wer_score_list[snr])
    clean_wer_score = np.average(clean_wer_score_list[snr])
    print(snr,pesq_score,stoi_score,wer_score,clean_wer_score)
  
    writer.writerow([snr,pesq_score,stoi_score,wer_score,clean_wer_score])
    
