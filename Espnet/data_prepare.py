import torch
import json
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.dataset import TransformDataset
from espnet.asr.pytorch_backend.asr import CustomConverter
from espnet.utils.io_utils import LoadInputsAndTargets

converter = CustomConverter(subsampling_factor=1, dtype=torch.float32)

with open("dump/train_nodev/deltafalse/data.json", "rb") as f:
    train_json = json.load(f)["utts"]
    train = make_batchset(train_json,batch_size=30)

load_tr = LoadInputsAndTargets(
    mode="asr",
    load_output=True,
    preprocess_conf=None,
    preprocess_args={"train": True},  # Switch the mode of preprocessing
    )
dataset = TransformDataset(train, lambda data: converter([load_tr(data)]))

model = torch.load("model.loss.best.entire")
model.train()
for i in range(1):
    for key in train[0]:
        print(key[0])
    x = dataset[i][0]
    ilen = dataset[i][1]
    y = dataset[i][2]
    print("x:",x)
    print("x size:",x.size())
    print("ilen:",ilen)
    print("ilen size:",ilen.size())
    print("y:",y)
    print("y size:",y.size())
    # print("forward :",model.forward(x.cuda(),ilen.cuda(),y.cuda()))


