import torch, os
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from util import get_filepaths
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
import pdb
from tqdm import tqdm 
from joblib  import parallel_backend, Parallel, delayed

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
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, yy_pad, x_lens, y_lens


def Load_data(args, Train_path):

    train_paths = []
    val_paths = []
    #[Neil] Modify fea_path
    #[Yo] Modify n_files, test/train split(test_size set to  0.1)
    n_files = np.array([x[:-1] for x in open(Train_path).readlines() if str(x.split('/')[-3])[0]=='n'])
    train_paths,val_paths = train_test_split(n_files[:500],test_size=0.1,random_state=999)
    
    train_dataset, val_dataset = CustomDataset(train_paths), CustomDataset(val_paths)
    # [Yo] Add padding collate fn
    loader = { 
        'train':DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=8, pin_memory=True, collate_fn=pad_collate),
        'val'  :DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=8, pin_memory=True, collate_fn=pad_collate)
    }

    return loader

class CustomDataset(Dataset):

    def __init__(self, paths):   # initial logic happens like transform
        #[Neil] Modify CustomDataset
        #[Yo] Modify CustomDataset
        self.n_paths = paths
        self.noisy = []
        self.clean = []
        print('Reading data...')
        for _,p in enumerate(tqdm(self.n_paths)):
            self.noisy += [torch.load(p)]
            name = p.split('/')[-1]
            n_folder = '/'.join(p.split('/')[-4:-1])
            self.clean += [torch.load(p.replace(n_folder,"clean"))]
        

    def __getitem__(self, index):

#         noisy,clean,target = torch.load(self.n_paths[index])
#         clean = torch.load(self.c_paths[index])  

#         return noisy,clean,target
        
        return self.noisy[index],self.clean[index]

    def __len__(self):  # return count of sample we have
        
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


        
        
    