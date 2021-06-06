import torch, os
import torch.nn as nn
import numpy as np
import json
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
import pdb
from tqdm import tqdm 
from joblib  import parallel_backend, Parallel, delayed
from utils.load_asr_data import load_y_dict
import models.transformerencoder
import models.e2e_vc_transformer
import models.BLSTM
from utils.util import getfilename, noisy2clean_train
from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt

from espnet.utils.training.batchfy import make_batchset
from espnet.utils.dataset import TransformDataset
from espnet.asr.asr_utils import torch_load
from espnet.vc.pytorch_backend.vc import CustomConverter
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.dynamic_import import dynamic_import

import random


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
        
        
def load_checkoutpoint(semodel,optimizer,checkpoint_path):

    if os.path.isfile(checkpoint_path):
        semodel.eval()
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        semodel.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        try:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        except:
            print("AttributeError: 'NoamOpt' object has no attribute 'state'")
        print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {epoch})")
        
        return semodel, epoch, best_loss, optimizer
    else:
        raise NameError(f"=> no checkpoint found at '{checkpoint_path}'")



def Load_SE_model(args,idim=None, odim=None, train_args=None):
    if args.corpus=="TMHINT_DYS":
        if args.retrain:
            originSEmodel = torch.load(args.VCmodel_path)
            optimizer = get_std_opt(originSEmodel.parameters(), 384, originSEmodel.args.transformer_warmup_steps, originSEmodel.args.transformer_lr)
            
            semodel = models.e2e_vc_transformer.Transformer(idim, odim, train_args)
            semodel, epoch, best_loss, optimizer = load_checkoutpoint(semodel,optimizer,args.model_path)
            criterions = {
            'mse'     : nn.MSELoss(),
            'l1'      : nn.L1Loss(),
            'l1smooth': nn.SmoothL1Loss()}
            device = torch.device(f'cuda:{args.gpu}')
            criterion = criterions[args.loss_fn].to(device)
            return semodel, epoch, best_loss, optimizer, criterion, device
        else:
            semodel = models.e2e_vc_transformer.Transformer(idim, odim, train_args)
            return semodel
    else:
        semodel = eval("models."+args.SEmodel.split('_')[0]+"."+args.SEmodel+"()")

        device = torch.device(f'cuda:{args.gpu}')
                
        criterions = {
            'mse'     : nn.MSELoss(),
            'l1'      : nn.L1Loss(),
            'l1smooth': nn.SmoothL1Loss()}
        criterion = criterions[args.loss_fn].to(device)
        
        optimizers = {
            'adam'    : Adam(semodel.parameters(),lr=args.lr,weight_decay=0)}
        optimizer = optimizers[args.optim]
        
        
        if args.resume:
            semodel, epoch, best_loss, optimizer = load_checkoutpoint(semodel,optimizer,args.checkpoint_path)
        elif args.retrain or args.mode == "test":
            semodel, epoch, best_loss, optimizer = load_checkoutpoint(semodel,optimizer,args.model_path)
        else:
            epoch = 0
            best_loss = 10
            semodel.apply(weights_init)
            
        para = count_parameters(semodel)
        print(f'Num of SE model parameter : {para}')
            
        return semodel, epoch, best_loss, optimizer, criterion, device

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

def sort_len_vc(batch):#[B,3,6,840]
    (xx, asr_l, asr_y) = zip(*batch)
    ind=sorted(range(len(list(asr_l))), key=lambda k: list(asr_l)[k], reverse=True)
    
    xx=tuple([list(xx[i]) for i in ind])    
    xx_new=[]
    for i in range(len(xx[0])):
        lis=[]
        for j in range(len(xx)):
            lis.append(xx[j][i])
        xx_new.append(lis)
            
    
    asr_l=[asr_l[i] for i in ind]
    asr_y=[asr_y[i] for i in ind]
    asr_y_pad = pad_sequence(asr_y, batch_first=True, padding_value=0)
    

    #xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    #asr_y_pad = pad_sequence(asr_y, batch_first=True, padding_value=0)
    
    return xx_new, torch.tensor(asr_l), asr_y_pad


def Load_data(args):
    train_paths = []
    val_paths = []
    
    train_spec_noisy_list=getfilename(os.path.join(args.out_path,'spec','train/noisy'),"train")

    n_files = np.array(train_spec_noisy_list)
    
    if args.train_num is None:
        train_paths,val_paths = train_test_split(n_files,test_size=args.val_ratio,random_state=999)
    else:
        train_paths,val_paths = train_test_split(n_files[:args.train_num],test_size=args.val_ratio,random_state=999)
    
    asr_dict = load_y_dict(args)

    train_dataset, val_dataset = CustomDataset(train_paths, asr_dict, args.corpus), CustomDataset(val_paths, asr_dict, args.corpus)
    
    # [Yo] Add padding collate_fn
    loader = { 
        'train':DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=8, pin_memory=True, collate_fn=pad_collate),
        'val'  :DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=8, pin_memory=True, collate_fn=pad_collate)
    }

    return loader

def read_vcjson(Espnet_path, json_file):
    json_file_path=os.path.join(Espnet_path, json_file)
    print("reading json_file_path:", json_file_path)
    with open(json_file_path, "rb") as f:
        json_data = json.load(f)["utts"]
        for name in json_data:
            for input_feats in json_data[name]["input"]:
                if input_feats["feat"][:3]=="exp":
                    input_feats["feat"] = os.path.join(Espnet_path, input_feats["feat"])
    return json_data


def Load_data_VC(args, VCargs):

    train_json=read_vcjson(args.Espnet_path, VCargs.train_json)
    valid_json=read_vcjson(args.Espnet_path, VCargs.valid_json)
    use_sortagrad = VCargs.sortagrad == -1 or VCargs.sortagrad > 0
    if use_sortagrad:
        VCargs.batch_sort_key = "input"

    train_batchset = make_batchset(
        train_json,
        len(train_json), #VCargs.batch_size, #or len(train_json) ?
        len(train_json), #VCargs.maxlen_in,
        len(train_json), #VCargs.maxlen_out,
        num_batches=1,
        batch_sort_key=VCargs.batch_sort_key,
        min_batch_size=VCargs.ngpu if VCargs.ngpu > 1 else 1,
        shortest_first=use_sortagrad,
        count=VCargs.batch_count,
        batch_bins=VCargs.batch_bins,
        batch_frames_in=VCargs.batch_frames_in,
        batch_frames_out=VCargs.batch_frames_out,
        batch_frames_inout=VCargs.batch_frames_inout,
        swap_io=False,
        iaxis=0,
        oaxis=0,
    )
    

    valid_batchset = make_batchset(
        valid_json,
        len(valid_json),#VCargs.batch_size,
        100000,#VCargs.maxlen_in,
        100000,#VCargs.maxlen_out,
        num_batches=1,
        batch_sort_key=VCargs.batch_sort_key,
        min_batch_size=VCargs.ngpu if VCargs.ngpu > 1 else 1,
        count=VCargs.batch_count,
        batch_bins=VCargs.batch_bins,
        batch_frames_in=VCargs.batch_frames_in,
        batch_frames_out=VCargs.batch_frames_out,
        batch_frames_inout=VCargs.batch_frames_inout,
        swap_io=False,
        iaxis=0,
        oaxis=0,
    )

    load_tr = LoadInputsAndTargets(
        mode="vc",
        use_speaker_embedding=VCargs.use_speaker_embedding,
        use_second_target=VCargs.use_second_target,
        preprocess_conf=VCargs.preprocess_conf,
        preprocess_args={"train": True},  # Switch the mode of preprocessing
        keep_all_data_on_mem=VCargs.keep_all_data_on_mem,
    )

    load_cv = LoadInputsAndTargets(
        mode="vc",
        use_speaker_embedding=VCargs.use_speaker_embedding,
        use_second_target=VCargs.use_second_target,
        preprocess_conf=VCargs.preprocess_conf,
        preprocess_args={"train": False},  # Switch the mode of preprocessing
        keep_all_data_on_mem=VCargs.keep_all_data_on_mem,
    )
    converter = CustomConverter()

    asr_dict = load_y_dict(args)
    
    train_names = [train_batchset[0][i][0] for i in range(len(train_json))]
    valid_names = [valid_batchset[0][i][0] for i in range(len(valid_json))]
    
    train_dataset, val_dataset = CustomDataset(train_batchset, asr_dict, corpus=args.corpus, names=train_names, transform =lambda data: converter([load_tr(data)])), CustomDataset(valid_batchset, asr_dict, corpus=args.corpus, names=valid_names, transform =lambda data: converter([load_cv(data)]))
      
    loader = { 
        'train':DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=not use_sortagrad, num_workers=VCargs.num_iter_processes,collate_fn=sort_len_vc),#collate_fn=lambda x: x[0]),
        'val'  :DataLoader(val_dataset, batch_size=args.batch_size,
                              shuffle=not use_sortagrad, num_workers=VCargs.num_iter_processes,collate_fn=sort_len_vc),# collate_fn=lambda x: x[0])
    }

    return loader


class CustomDataset(Dataset):
    def __init__(self, paths, asr_dict, corpus="TIMIT", names=None, transform=None):
        
        self.corpus = corpus
        self.asr_dict = asr_dict
        self.asr_ilen = []
        self.asr_y = []
        self.dataset = paths
        self.len=0
        

        print('Reading data, corpus=',corpus)

        if self.corpus=="TMHINT_DYS":
            self.len=len(self.dataset[0])
            self.transform=transform
            data=self.transform(self.dataset[0])
            self.vc_x = [] 
            
            for i in tqdm(range(self.len)):
                x_tuple=(data["xs"][i],data["ilens"][i],data["ys"][i],data["labels"][i],data["olens"][i],data["spembs"][i])
                self.vc_x += [x_tuple]
                
                self.asr_y += [self.asr_dict[names[i]][1]]
                self.asr_ilen += [self.asr_dict[names[i]][0]]
        else:
            self.len=len(self.dataset)
            self.noisy = []
            self.clean = []

            for _, n_file in enumerate(tqdm(self.dataset)):
                c_file, c_name = noisy2clean_train(n_file, corpus)
                self.noisy.append(torch.load(n_file))
                self.clean.append(torch.load(c_file))
                self.asr_ilen += [self.asr_dict[c_name][0]]
                self.ars_y += [self.asr_dict[c_name][1]]

    def __getitem__(self, index):
        if self.corpus=="TMHINT_DYS":
            return self.vc_x[index], self.asr_ilen[index], self.asr_y[index] 
        else:
            return self.noisy[index], self.clean[index], self.asr_ilen[index], self.ars_y[index]

    def __len__(self):
        return self.len



        
        
    
