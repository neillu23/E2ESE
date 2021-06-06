import os
import sys
import csv
import numpy as np

result_file = sys.argv[1]
score_path = sys.argv[2]


pesq_score_list = {}
stoi_score_list = {}
with open(result_file, newline='') as rf:
    r_lines = csv.reader(rf)
    for res in r_lines:
        if res[0] == "Filename" or res[0] == "Average":
            continue
        name = res[0]
        pesq = res[1]
        stoi = res[2]
        key = int(name.split("/")[-2].replace("n","-").replace("dB",""))
        noise_type = name.split("/")[-3]
        if noise_type not in pesq_score_list:
            pesq_score_list[noise_type] = {}
            stoi_score_list[noise_type] = {}
        if key not in pesq_score_list[noise_type]:
            pesq_score_list[noise_type][key] = []
            stoi_score_list[noise_type][key] = []
        pesq_score_list[noise_type][key].append(float(pesq))
        stoi_score_list[noise_type][key].append(float(stoi))

key_list=sorted(pesq_score_list[noise_type].keys())

if not os.path.isdir(score_path): 
    os.makedirs(score_path)
fp = open(os.path.join(score_path,result_file.split("/")[-1]),"w", newline='')
writer = csv.writer(fp)
for noise_type in pesq_score_list:
    writer.writerow("\n")
    print("")
    writer.writerow([noise_type])
    print(noise_type)
    writer.writerow(["SNR", "pesq_avg", "stoi_avg"])
    print("SNR, pesq_avg, stoi_avg")
    for snr in key_list:
        pesq_score = np.average(pesq_score_list[noise_type][snr])
        stoi_score = np.average(stoi_score_list[noise_type][snr])
        print(snr,pesq_score,stoi_score)
    
        writer.writerow([snr,pesq_score,stoi_score])
        
