import torch
import json
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.dataset import TransformDataset
from espnet.asr.pytorch_backend.asr import CustomConverter
from espnet.utils.io_utils import LoadInputsAndTargets
from CombinedModel import CombinedModel

'''
with open(jsonpath, "rb") as f:
       train_feature = json.load(f)["utts"]

'''

def data_prepare(train_feature):

    converter = CustomConverter(subsampling_factor=1, dtype=torch.float32)
    train = make_batchset(train_feature,batch_size=30)
    load_tr = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        preprocess_conf=None,
        preprocess_args={"train": True},  # Switch the mode of preprocessing
        )
    dataset = TransformDataset(train, lambda data: converter([load_tr(data)]))

    return dataset
'''
ASRmodel = torch.load("/home/eyelab/yo/se/espnet/egs/timit/asr1/exp/train_nodev_pytorch_train/results/model.acc.best_entire.pth")
model = CombinedModel(None,ASRmodel,None,alpha=0.5)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print('dataset.shape',len(dataset))
for i in range(50):
    optimizer.zero_grad()
    x = dataset[i][0]
    ilen = dataset[i][1]
    y = dataset[i][2]
    
    
    loss=model(x.cuda(),ilen.cuda(),y.cuda())
    print("loss :",loss)
    #loss = self.model(*x).mean() / self.accum_grad

    loss.backward()
    optimizer.step()
'''


