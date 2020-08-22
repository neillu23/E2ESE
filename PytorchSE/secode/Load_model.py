import torch, os
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
import pdb
from tqdm import tqdm 
from joblib  import parallel_backend, Parallel, delayed
from utils.load_asr import load_asr

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.3))
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
        
        
def load_checkoutpoint(model,optimizer,checkpoint_path):

    if os.path.isfile(checkpoint_path):
        model.eval()
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {epoch})")
        
        return model,epoch,best_loss,optimizer
    else:
        raise NameError(f"=> no checkpoint found at '{checkpoint_path}'")



def Load_model(args,model,checkpoint_path,model_path):
    
    criterion = {
        'mse'     : nn.MSELoss(),
        'l1'      : nn.L1Loss(),
        'l1smooth': nn.SmoothL1Loss()}

    device    = torch.device(f'cuda:{args.gpu}')
#     pdb.set_trace()
#     from models.DDAE import DDAE_01 as model

    criterion = criterion[args.loss_fn].to(device)
    
    optimizers = {
        'adam'    : Adam(model.parameters(),lr=args.lr,weight_decay=0)}
    optimizer = optimizers[args.optim]
    
    if args.resume:
        model,epoch,best_loss,optimizer = load_checkoutpoint(model,optimizer,checkpoint_path)
    elif args.retrain:
        model,epoch,best_loss,optimizer = load_checkoutpoint(model,optimizer,model_path)
        
        
    else:
        epoch = 0
        best_loss = 10
        model.apply(weights_init)
        
    para = count_parameters(model)
    print(f'Num of model parameter : {para}')
        
    return model,epoch,best_loss,optimizer,criterion,device

# [Yo]
def pad_collate(batch):
    (xx, yy, asr_l, asr_y) = zip(*batch)

    ind=sorted(range(len(list(asr_l))), key=lambda k: list(asr_l)[k], reverse=True)
    
    xx=[xx[i] for i in ind]
    yy=[yy[i] for i in ind]
    asr_l=[asr_l[i] for i in ind]
    asr_y=[asr_y[i] for i in ind]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    asr_y_pad = pad_sequence(asr_y, batch_first=True, padding_value=0)
    
    return xx_pad, yy_pad, torch.tensor(asr_l), asr_y_pad


def Load_data(args):

    train_paths = []
    val_paths = []
    #[Neil] Modify fea_path
    #[Yo] Modify n_files, test/train split(test_size set to  0.1)
    n_files = np.array([x[:-1] for x in open(args.train_path).readlines() if str(x.split('/')[-3])[0]=='n'])
    train_paths,val_paths = train_test_split(n_files[:args.train_num],test_size=args.val_ratio,random_state=999)
    
    print('Reading json files...')
    asr_y_path = [item for item in args.asr_y_path.split(',')]
    asr_dict = {}
    for json_path in asr_y_path:
        asr_dict = load_asr(json_path,asr_dict)

    train_dataset, val_dataset = CustomDataset(train_paths, asr_dict), CustomDataset(val_paths, asr_dict)
    # [Yo] Add padding collate_fn
    loader = { 
        'train':DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=8, pin_memory=True, collate_fn=pad_collate),
        'val'  :DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=8, pin_memory=True, collate_fn=pad_collate)
    }

    return loader

class CustomDataset(Dataset):

    def __init__(self, paths, asr_dict):   # initial logic happens like transform
        #[Neil] Modify CustomDataset
        #[Yo] Modify CustomDataset
        self.n_paths = paths
        self.asr_dict = asr_dict
        self.noisy = []
        self.clean = []
        self.asr_ilen = []
        self.ars_y = []
        print('Reading data...')
 
        for _,p in enumerate(tqdm(self.n_paths)):
            self.noisy += [torch.load(p)]
            n_folder = '/'.join(p.split('/')[-4:-1])
            self.clean += [torch.load(p.replace(n_folder,"clean"))]

            name = p.split('/')[-1].replace('.pt','')
            self.asr_ilen += [self.asr_dict[name][0]]
            self.ars_y += [self.asr_dict[name][1]]

    def __getitem__(self, index):
        return self.noisy[index], self.clean[index], self.asr_ilen[index], self.ars_y[index]

    def __len__(self):
        return len(self.n_paths)


'''class CustomDataset(Dataset):

    def __init__(self, paths,clean_path):   # initial logic happens like transform

        self.n_paths = paths
        self.c_paths = [os.path.join(clean_path,'noise_'+'_'.join((noisy_path.split('/')[-1].split('_')[-2:])) ) for noisy_path in paths]


    def __getitem__(self, index):

        noisy,clean = torch.load(self.n_paths[index]),torch.load(self.c_paths[index]) 


        return noisy,clean

    def __len__(self):  # return count of sample we have
        
        return len(self.n_paths)'''


        
        
    
