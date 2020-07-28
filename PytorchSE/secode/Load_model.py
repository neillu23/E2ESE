import torch, os
import torch.nn as nn
from torch.optim import Adam
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


def Load_data(args, Train_path):
    
#     torch.set_num_threads(torch.get_num_threads()//2)
    train_paths = []
    val_paths = []
    #[Neil] Modify fea_path
    fea_path = '/mnt/Data/user_vol_2/user_neillu/E2E_Spec/training/'
    folders = os.listdir(fea_path)
    for folder in folders:
        file_paths = get_filepaths(os.path.join(fea_path,folder),ftype='.pt')
        train_path,val_path = train_test_split(file_paths,test_size=10,random_state=999)
        train_paths = train_paths+train_path
        # print(train_paths)
        val_paths = val_paths+val_path
    train_dataset, val_dataset = CustomDataset(train_paths), CustomDataset(val_paths)
    loader = { 
        'train':DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=8, pin_memory=True),
        'val'  :DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=8, pin_memory=True)
    }

    return loader

def load_torch(path):
    return torch.load(path)

class CustomDataset(Dataset):

    def __init__(self, paths):   # initial logic happens like transform

#         self.n_paths = paths
        #[Neil] Modify for CustomDataset
        self.n_paths = paths
        self.noisy = []
        self.clean = []
        for p in self.n_paths:
            self.noisy += [load_torch(p)]
            name = p.split('/')[-1].split('_')[0] + '_' + p.split('/')[-1].split('_')[1] + '_' + p.split('/')[-1].split('_')[-1] 
            self.clean += [load_torch(p.replace("train_noisy","train_clean").replace(p.split('/')[-1],name))]
        
        # with parallel_backend('multiprocessing', n_jobs=64):
        #     self.n_paths = Parallel()(delayed(load_torch)(path) for path in tqdm(paths))
        #     print(self.n_paths)
        # self.noisy,self.clean,_ = zip(*self.n_paths)


#         self.c_paths = [os.path.join(clean_path,'clean_'+'_'.join((noisy_path.split('/')[-1].split('_')[-2:])) ) for noisy_path in paths]


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


        
        
    