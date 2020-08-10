import os, argparse, torch, random, sys
from Trainer import Trainer
from Load_model import Load_model, Load_data
from util import check_folder
from tensorboardX import SummaryWriter
from CombinedModel import CombinedModel
import torch.backends.cudnn as cudnn
import pandas as pd
import pdb

#os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# fix random
SEED = 999
random.seed(SEED)
torch.manual_seed(SEED)
cudnn.deterministic = True


# data path

#[Neil] Modify Train_path but not used?
#[Yo] Modify Train_path, Test_path: 

Train_path = './TIMIT_noisy_40hr_spec_filelist.txt'
Test_path = {'noisy':'./TIMIT_noisy_40hr_wav_filelist.txt','clean':'./TIMIT_Manner_E2E_wav_filelist.txt'}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)  
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--loss_fn', type=str, default='l1')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--SEmodel', type=str, default='transformerencoder_03') 
    #####
    parser.add_argument('--ASRmodel_path', type=str, default='./newfbank.model.acc.best.entire.pth')
    parser.add_argument('--alpha', type=float, default=0.0002) 
    #loss = SEloss + self.alpha * ASRloss
    parser.add_argument('--asr_y_path', type=str, default='./data_test.json,./data_train_dev.json,./data_train_nodev.json') 
    parser.add_argument('--c_dic', type=str, default='./c_wavfolder_dic.npy') 
    #####
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--target', type=str, default='MAP') #'MAP' or 'IRM'
    parser.add_argument('--task', type=str, default='DNS_SE') 
    parser.add_argument('--resume' , action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--re_epochs', type=int, default=300)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    return args

def get_path(args):
    
    checkpoint_path = f'./checkpoint/{args.SEmodel}_{args.target}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                    f'lr{args.lr}.pth.tar'
    model_path = f'./save_model/{args.SEmodel}_{args.target}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                    f'lr{args.lr}.pth.tar'
    score_path = f'./Result/{args.SEmodel}_{args.target}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                    f'lr{args.lr}.csv'
    
    return checkpoint_path,model_path,score_path


if __name__ == '__main__':
    # get current path
    cwd = os.path.dirname(os.path.abspath(__file__))
    print(cwd)
    print(SEED)
       
    # get parameter
    args = get_args()
    
    # declair path
    checkpoint_path,model_path,score_path = get_path(args)
    
    # tensorboard
    writer = SummaryWriter('./logs')
#     writer = SummaryWriter(f'{args.logs}/{args.SEmodel}/{args.optim}/{args.optim}/{args.loss_fn}')
#     exec ("from models.{} import {} as model".format(args.SEmodel.split('_')[0], args.SEmodel))
#     pdb.set_trace()

    ASRmodel = torch.load(args.ASRmodel_path)
    
    exec (f"from models.{args.SEmodel.split('_')[0]} import {args.SEmodel} as SEmodel")
    SEmodel     = SEmodel()
    SEmodel, epoch, best_loss, optimizer, criterion, device = Load_model(args,SEmodel,checkpoint_path, model_path)
    loader = Load_data(args, Train_path)

    model = CombinedModel(SEmodel, ASRmodel, criterion, args.alpha)

    for param in model.ASRmodel.parameters():
        param.requires_grad = False
    
    if args.retrain:
        args.epochs = args.re_epochs 
        checkpoint_path,model_path,score_path = get_path(args)
        
    Trainer = Trainer(model, args.epochs, epoch, best_loss, optimizer, 
                      criterion, device, loader, Test_path, writer, model_path, score_path, args)
    try:
        if args.mode == 'train':
            Trainer.train()
        Trainer.test()
        
    except KeyboardInterrupt:
        state_dict = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss
            }
        check_folder(checkpoint_path)
        torch.save(state_dict, checkpoint_path)
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
