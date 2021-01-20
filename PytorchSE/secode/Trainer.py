import torch.nn as nn
import torch
import pandas as pd
import os, sys
from tqdm import tqdm
import librosa, scipy
import pdb
import numpy as np
from scipy.io.wavfile import write as audiowrite
from utils.util import  check_folder, recons_spec_phase, cal_score, make_spectrum, get_clean_file, get_cleanwav_dic, getfilename
maxv = np.iinfo(np.int16).max

def save_checkpoint(epoch, model, optimizer, best_loss, model_path):
    state_dict = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_loss': best_loss
        }
    check_folder(model_path)
    torch.save(state_dict, model_path)
    
def train_epoch(model, optimizer, device, loader, epoch, epochs, mode, alpha):
    train_loss = 0
    train_SE_loss = 0
    train_ASR_loss = 0
    progress = tqdm(total=len(loader[mode]), desc=f'Epoch {epoch} / Epoch {epochs} | {mode}', unit='step')
    if mode == 'train':
        model.train()
        model.SEmodel.train()
    else:
        model.eval()
        model.SEmodel.eval()
        torch.no_grad()

    for noisy, clean, ilen, asr_y in loader[mode]:
        noisy, clean, ilen, asr_y = noisy.to(device), clean.to(device), ilen.to(device), asr_y.to(device)
        
        # predict and calculate loss
        SEloss, ASRloss = model(noisy, clean, ilen, asr_y)
        loss = (1 - alpha) * SEloss + alpha * ASRloss

        # train the model
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            torch.no_grad()

        # record loss
        train_loss += loss.detach().item()
        train_SE_loss += SEloss.detach().item()
        train_ASR_loss += ASRloss.detach().item()
        progress.update(1)
   
    progress.close()
    train_loss = train_loss/len(loader[mode])
    train_SE_loss = train_SE_loss/len(loader[mode])
    train_ASR_loss = train_ASR_loss/len(loader[mode])
    print(f'{mode}_loss:{train_loss}, SE{mode}_loss:{train_SE_loss}, ASR{mode}_loss:{train_ASR_loss}')
    return train_SE_loss, train_ASR_loss
           
def train(model, epochs, epoch, best_loss, optimizer, 
         device, loader, writer, model_path, args):
    print('Training...')

    model = model.to(device)
    while epoch < epochs:
        # add for 2 stage training 
        alpha = args.alpha
        if epoch < args.alpha_epoch:
            alpha = 0
        if epoch == args.alpha_epoch:
            best_loss = 100

        train_SE_loss, train_ASR_loss = train_epoch(model, optimizer, device, loader, epoch, epochs, "train",alpha)
        val_SE_loss, val_ASR_loss = train_epoch(model, optimizer, device, loader, epoch, epochs,"val",alpha)

        train_loss=(1 - alpha) * train_SE_loss + alpha * train_ASR_loss
        val_loss=(1 - alpha) * val_SE_loss + alpha * val_ASR_loss
          
        writer.add_scalars(f'{args.task}/{model.SEmodel.__class__.__name__}_{args.optim}_{args.loss_fn}', {'train': train_loss, 'train_SE': train_SE_loss},epoch)
        writer.add_scalars(f'{args.task}/{model.SEmodel.__class__.__name__}_{args.optim}_{args.loss_fn}', {'val': val_loss, 'val_SE': val_SE_loss},epoch)
        
        if best_loss > val_loss:
            if epoch >= args.alpha_epoch and "after_alpha_epoch" not in model_path:
                model_path = model_path.replace("_alpha_epoch","_after_alpha_epoch")
            print(f"Save SE model to '{model_path}'")
            save_checkpoint(epoch,model.SEmodel, optimizer, best_loss, model_path)
            best_loss = val_loss

        epoch += 1

def prepare_test(test_file, c_dict, device, corpus="TIMIT"):
    clean_file, n_folder = get_clean_file(test_file, c_dict, corpus)

    n_wav,sr = librosa.load(test_file,sr=16000)
    c_wav,sr = librosa.load(clean_file,sr=16000)

    n_spec,n_phase,n_len = make_spectrum(y=n_wav)
    c_spec,c_phase,c_len = make_spectrum(y=c_wav)

    n_spec = torch.from_numpy(n_spec.transpose()).to(device).unsqueeze(0)
    c_spec = torch.from_numpy(c_spec.transpose()).to(device).unsqueeze(0)
    return n_spec, n_phase, n_len, c_wav, c_spec, c_phase, n_folder

    
def write_score(model, device, test_file, c_dict, enhance_path, ilen, y, score_path, asr_result,corpus="TIMIT"):
    n_spec, n_phase, n_len, c_wav, c_spec, c_phase, n_folder = prepare_test(test_file, c_dict,device,corpus)
    #[Yo] Change prediction
    
    if asr_result!=None:
        ### Get ASR prediction results
        Fbank=model.Fbank()
        model.ASRmodel.report_cer=True
        model.ASRmodel.report_wer=True
        if asr_result == 'enhanced':
            spec = model.SEmodel(n_spec)
            phase = n_phase
        elif asr_result == 'noisy':
            spec = n_spec
            phase = n_phase
        else:
            spec= c_spec
            phase = c_phase
        
        fbank = Fbank.forward(spec)
        fbank, ilen, y = fbank.to(device), ilen.to(device), y.to(device)

        ASRloss, asr_cer = model.ASRmodel(fbank, ilen.unsqueeze(0), y.unsqueeze(0))
        spec=spec.cpu().detach().numpy()
        recon_wav = recons_spec_phase(spec.squeeze().transpose(),phase,n_len)
        # cal score
        s_pesq, s_stoi = cal_score(c_wav,recon_wav)
        with open(score_path, 'a') as f:
            f.write(f'{test_file},{s_pesq},{s_stoi},{asr_cer}\n')
    else:
        enhanced_spec = model.SEmodel(n_spec).cpu().detach().numpy()
        enhanced = recons_spec_phase(enhanced_spec.squeeze().transpose(),n_phase,n_len)
        # cal score
        s_pesq, s_stoi = cal_score(c_wav,enhanced)
        with open(score_path, 'a') as f:
            f.write(f'{test_file},{s_pesq},{s_stoi}\n')
        # write enhanced waveform
        out_path = f"{enhance_path}/{n_folder+'/'+test_file.split('/')[-1]}"
        check_folder(out_path)
        audiowrite(out_path,16000,(enhanced* maxv).astype(np.int16))

        
            
def test(model, device, noisy_path, clean_path, asr_dict, enhance_path, score_path, args):
    model = model.to(device)
    # load model
    model.eval()
    torch.no_grad()
    
    # load data
    if args.test_num is None:
        test_files = np.array(getfilename(noisy_path,"test"))
    else:
        test_files = np.array(getfilename(noisy_path,"test")[:args.test_num])

    c_dict = get_cleanwav_dic(clean_path, args.corpus)
    
    #open score file
   
    if os.path.exists(score_path):
        os.remove(score_path)
    
    check_folder(score_path)
    print('Save PESQ&STOI results to:', score_path)
    
    with open(score_path, 'a') as f:
        f.write('Filename,PESQ,STOI\n')

    print('Testing...')       
    for test_file in tqdm(test_files):
        name=test_file.split('/')[-1].replace('.wav','')
        ilen, y=asr_dict[name][0],asr_dict[name][1]
        write_score(model, device, test_file, c_dict, enhance_path, ilen, y, score_path, args.asr_result, args.corpus)

    data = pd.read_csv(score_path)
    pesq_mean = data['PESQ'].to_numpy().astype('float').mean()
    stoi_mean = data['STOI'].to_numpy().astype('float').mean()
    with open(score_path, 'a') as f:
        f.write(','.join(('Average',str(pesq_mean),str(stoi_mean)))+'\n')

        
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
    