import json
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.dataset import TransformDataset
from espnet.asr.pytorch_backend.asr import CustomConverter
from espnet.utils.io_utils import LoadInputsAndTargets
import numpy as np
import itertools 
from tqdm import tqdm
from utils.util import asr2clean


def load_y_dict(args):
    print('Reading json files...')
    asr_y_path = [item for item in args.asr_y_path.split(',')]
    asr_dict = {}
    for json_path in asr_y_path:
        asr_dict = load_asr_data(json_path, asr_dict, args.corpus)
    return asr_dict


def load_asr_data(json_path, asr_dict, corpus="TIMIT"):
    with open(json_path, "rb") as f:
        train_feature = json.load(f)["utts"]

    converter = CustomConverter(subsampling_factor=1)
    train = make_batchset(train_feature,batch_size=len(train_feature))
    asr_names = [train[0][i][0] for i in range(len(train_feature))]
    load_tr = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        preprocess_conf=None,
        preprocess_args={"train": True},  # Switch the mode of preprocessing
        )
    dataset = TransformDataset(train, lambda data: converter([load_tr(data)]))
    data1 = dataset[0][1]
    data2 = dataset[0][2]
    
    for i in tqdm(range(len(asr_names))):
        ilen = data1[i]
        y = data2[i]
        clean_name = asr2clean(asr_names[i], corpus)
        if clean_name:
            asr_dict[clean_name]=[ilen,y]
    return asr_dict

