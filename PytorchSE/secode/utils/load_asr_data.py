import json
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.dataset import TransformDataset
from espnet.asr.pytorch_backend.asr import CustomConverter
from espnet.utils.io_utils import LoadInputsAndTargets
import numpy as np
import itertools 
from tqdm import tqdm

def load_asr_data(json_path, dic, TMHINT=None):
    with open(json_path, "rb") as f:
        train_feature = json.load(f)["utts"]

    converter = CustomConverter(subsampling_factor=1)
    train = make_batchset(train_feature,batch_size=len(train_feature))
    name=[train[0][i][0] for i in range(len(train_feature))]
    load_tr = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        preprocess_conf=None,
        preprocess_args={"train": True},  # Switch the mode of preprocessing
        )
    dataset = TransformDataset(train, lambda data: converter([load_tr(data)]))
    data1 = dataset[0][1]
    data2 = dataset[0][2]
    
    for i in tqdm(range(len(name))):
        ilen = data1[i]
        y = data2[i]

        if TMHINT:
            speaker=["M1","M2","M3","F1","F2","F3"]
            tmp=name[i].split("_")
            if tmp[0] in speaker:
                if int(tmp[2])>=13:
                    name_key=int(speaker.index(tmp[0]))*200+(int(tmp[2])-13)*10+int(tmp[3])
                    dic[str(name_key)]=[ilen,y]
            else:
                dic["_".join(tmp[1:])]=[ilen,y]

        else:
            name_key=name[i]
            dic[name_key]=[ilen,y]
    return dic

'''
def savetodic(jsonpath,dic):
    
    dataset, name=data_prepare(train_feature)

    for i in tqdm(range(len(name))):
        ilen = dataset[0][1][i]
        y = dataset[0][2][i]
        dic[name[i]]=[ilen,y]
    
    return dic

if __name__ == "__main__":
    dic={}
    jsonpath1='./data_test.json'
    jsonpath2='./data_train_dev.json'
    jsonpath3='./data_train_nodev.json'

    dic=savetodic(jsonpath1,dic)
    dic=savetodic(jsonpath2,dic)
    dic=savetodic(jsonpath3,dic)

    #np.save('file_manner_y.npy', dic) 
   
'''
