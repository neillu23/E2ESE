import torch.nn as nn
import torch
import pandas as pd
import os, sys
from tqdm import tqdm
import librosa, scipy
import pdb
import numpy as np
from scipy.io.wavfile import write as audiowrite
from util import get_filepaths, check_folder, recons_spec_phase, cal_score
from gen_npy import make_spectrum
maxv = np.iinfo(np.int16).max

class Trainer:
    def __init__(self, model, epochs, epoch, best_loss, optimizer, 
                      criterion, device, loader,Test_path, writer, model_path, score_path, args):
        self.epoch = epoch
        self.epochs = epochs
        self.best_loss = best_loss
        self.model = model.to(device)
        self.optimizer = optimizer


        self.device = device
        self.loader = loader
        self.criterion = criterion
        self.Test_path = Test_path

        self.train_loss = 0
        self.SEtrain_loss = 0
        self.val_loss = 0
        self.SEval_loss = 0
        self.writer = writer
        self.model_path = model_path
        self.score_path = score_path
        self.args = args

    def save_checkpoint(self,):
        state_dict = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_loss': self.best_loss
            }
        check_folder(self.model_path)
        torch.save(state_dict, self.model_path)
    
    def slice_data(self,data,slice_size=64):
        # print("A",data,slice_size)
        # print("B",torch.split(data,slice_size,dim=1))
        # # print("C",torch.split(data,slice_size,dim=1)[:-1])
        #[Neil] Modify for CustomDataset
        data = torch.cat(torch.split(data,slice_size,dim=1),dim=0)
        # data = torch.cat(torch.split(data,slice_size,dim=1)[:-1],dim=0)
        # index = torch.randperm(data.shape[0])
        # return data[index]
        return data

    def _train_step(self, noisy, clean, ilen, asr_y):
        device = self.device
        noisy, clean, ilen, asr_y = noisy.to(device), clean.to(device), ilen.to(device), asr_y.to(device)
        
        #[Yo] Change loss
        loss = self.model(noisy, clean, ilen, asr_y)
        pred = self.model.SEmodel(noisy)
        '''
        for name, param in self.model.SEmodel.named_parameters():
            if param.requires_grad:
                print(name, param.data)
        print('tr_noisy',noisy)
        print('tr_pred',pred)
        print(noisy.size(),pred.size())
        exit()
        '''
        SEloss = self.criterion(pred, clean)

        self.train_loss += loss.item()
        self.SEtrain_loss += SEloss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def _train_epoch(self):
        self.train_loss = 0
        progress = tqdm(total=len(self.loader['train']), desc=f'Epoch {self.epoch} / Epoch {self.epochs} | train', unit='step')
        self.model.train()
        
        for noisy, clean, ilen, asr_y in self.loader['train']:
            
            self._train_step(noisy, clean, ilen, asr_y)
            progress.update(1)
       
        progress.close()
        self.train_loss /= len(self.loader['train'])
        self.SEtrain_loss /= len(self.loader['train'])
        print(f'train_loss:{self.train_loss}, SEtrain_loss:{self.SEtrain_loss}')

    
#     @torch.no_grad()
    def _val_step(self, noisy, clean, ilen, asr_y):
        device = self.device
        noisy, clean, ilen, asr_y = noisy.to(device), clean.to(device), ilen.to(device), asr_y.to(device)
        # [Yo] Delete data slicing, change pred
        #noisy, clean = self.slice_data(noisy), self.slice_data(clean)
        
        pred = self.model.SEmodel(noisy)

        '''
        for name, param in self.model.SEmodel.named_parameters():
            if param.requires_grad:
                print(name, param.data)
        print('val_noisy',noisy)
        print('val_pred',pred)
        print(noisy.size(),pred.size())
        
        exit()
        
        '''
        SEloss = self.criterion(pred, clean)
        E2Eloss = self.model(noisy, clean, ilen, asr_y)
        self.SEval_loss += SEloss.item()
        self.val_loss += E2Eloss.item()
        

    def _val_epoch(self):
        self.val_loss = 0
        progress = tqdm(total=len(self.loader['val']), desc=f'Epoch {self.epoch} / Epoch {self.epochs} | valid', unit='step')
        self.model.eval()

        for noisy, clean, ilen, asr_y in self.loader['val']:
            self._val_step(noisy, clean, ilen, asr_y)
            progress.update(1)

            
        progress.close()

        self.val_loss /= len(self.loader['val'])
        self.SEval_loss /= len(self.loader['val'])
        print(f'val_loss:{self.val_loss}, SEval_loss:{self.SEval_loss}')
        
        if self.best_loss > self.val_loss:
            
            print(f"Save model to '{self.model_path}'")
            self.save_checkpoint()
            self.best_loss = self.val_loss
            
    def write_score(self,test_file,c_dict):
        
        self.model.eval()
        n_data,sr = librosa.load(test_file,sr=16000)
        name = test_file.split('/')[-1].split('_')[0] + '_' + test_file.split('/')[-1].split('_')[1]
#         noisy = n_data
        n_folder = '/'.join(test_file.split('/')[-4:-1])
        name=name.replace('.wav','')
        c_name = name.split('_')[0]+'/'+name.split('_')[1]+'.WAV'
        c_folder=c_dict[name]
        clean_file= os.path.join(c_folder, c_name)

        #
        c_data,sr = librosa.load(clean_file,sr=16000)
        n_data,n_phase,n_len = make_spectrum(y=n_data)
        n_data = torch.from_numpy(n_data.transpose()).to(self.device).unsqueeze(0)
        
        #[Yo] Change prediction
        pred = self.model.SEmodel(n_data).cpu().detach().numpy()
        enhanced = recons_spec_phase(pred.squeeze().transpose(),n_phase,n_len)
        out_path = f"./Enhanced/{self.model.SEmodel.__class__.__name__}/{n_folder+'/'+test_file.split('/')[-1]}"
        check_folder(out_path)
        audiowrite(out_path,16000,(enhanced* maxv).astype(np.int16))

#         s_pesq, s_stoi = cal_score(c_data,noisy)
        s_pesq, s_stoi = cal_score(c_data,enhanced)
        
        with open(self.score_path, 'a') as f:
            f.write(f'{test_file},{s_pesq},{s_stoi}\n')

        
    def train(self):
        print('Training...')
       
        while self.epoch < self.epochs:
            self._train_epoch()
            self._val_epoch()
            self.writer.add_scalars(f'{self.args.task}/{self.model.SEmodel.__class__.__name__}_{self.args.optim}_{self.args.loss_fn}', {'train': self.train_loss},self.epoch)
            self.writer.add_scalars(f'{self.args.task}/{self.model.SEmodel.__class__.__name__}_{self.args.optim}_{self.args.loss_fn}', {'val': self.val_loss},self.epoch)
            self.epoch += 1
            
    
            
    def test(self):
        #[Yo] Modify Test_path
        # load model
        self.model.eval()
#         self.score_path = './Result/Test_Noisy.csv'
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['model'])
        #checkpoint_key ['epoch', 'model', 'optimizer', 'best_loss']
        
        test_files = get_filepaths(self.Test_path['noisy'],folders='BabyCry.wav,cafeteria_babble.wav')
        
        c_dict = np.load(self.args.c_dic,allow_pickle='TRUE').item()
        
        #clean_path = self.Test_path['clean']
        check_folder(self.score_path)
        
        if os.path.exists(self.score_path):
            os.remove(self.score_path)
        with open(self.score_path, 'a') as f:
            f.write('Filename,PESQ,STOI\n')
        print('Testing...')
        for test_file in tqdm(test_files):
            self.write_score(test_file,c_dict)
        
        data = pd.read_csv(self.score_path)

        pesq_mean = data['PESQ'].to_numpy().astype('float').mean()
        stoi_mean = data['STOI'].to_numpy().astype('float').mean()

        with open(self.score_path, 'a') as f:
            f.write(','.join(('Average',str(pesq_mean),str(stoi_mean)))+'\n')
#         with parallel_backend('multiprocessing', n_jobs=20):
#             val_pesq = Parallel()(delayed(write_score)
#                                              (16000,val_list[k][0], val_list[k][1], 'wb')
#                                               for k in range(len(val_list)))
        
        
class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.len = len(loader)
        self.preload()

    def preload(self):
        try:
            self.next_noisy, self.next_clean = next(self.loader)
        except StopIteration:
            self.next_noisy = None
            self.next_clean = None
            return
        with torch.cuda.stream(self.stream):
            self.next_noisy = self.next_noisy.cuda(non_blocking=True)
            self.next_clean = self.next_clean.cuda(non_blocking=True)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        noisy = self.next_noisy
        clean = self.next_clean
        self.preload()
        return noisy,clean
    
    def length(self):
        return self.len
    
    
