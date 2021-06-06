import json
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.dataset import TransformDataset
from espnet.asr.pytorch_backend.asr import CustomConverter
from espnet.utils.io_utils import LoadInputsAndTargets
import numpy as np
import itertools 
from tqdm import tqdm
from utils.util import asr2clean
from jiwer import wer
import  speech_recognition as sr
from scipy.io import wavfile
# import pdb


# Convert `data` to 32 bit integers:
def convert(data,wav_path,fs=16000):
    y = (np.iinfo(np.int32).max * (data/np.abs(data).max())).astype(np.int32)
    wavfile.write(wav_path, fs, y)
    
def cal_asr(wav):
    wav = wav/abs(wav).max()
    r = sr.Recognizer()
    convert(wav,'./wavfile.wav')
#     convert(enhanced,'./enhanced.wav')
#     librosa.output.write_wav('./clean.wav',clean,sr=16000)
#     librosa.output.write_wav('./enhanced.wav',enhanced,sr=16000)
    try:
        with sr.AudioFile('./wavfile.wav') as source:
            audio = r.record(source)
            result = r.recognize_google(audio,language='en-US')
    except:
        result = ''
#         result = result.replace(" ","")
    return result

def get_clean_txt(c_text):
    answer = []
    with open(c_text) as f:
        lines = f.readlines()
    lines = lines[0].replace('.\n','')
    answer.append(' '.join(lines.split(' ')[2:]).lower())
    return answer


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

