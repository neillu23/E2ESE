import os, argparse, torch, random, sys
from Trainer import train, test
from Load_model import Load_SE_model, Load_data
from utils.util import check_folder
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



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    #####
    parser.add_argument('--train_noisy', type=str, default='')
    parser.add_argument('--train_clean', type=str, default='')
    parser.add_argument('--test_noisy', type=str, default='')
    parser.add_argument('--test_clean', type=str, default='')
    parser.add_argument('--spec_path', type=str, default='')

    '''
    parser.add_argument('--train_noisy', type=str, default="data/train/noisy_spec_filelist.txt")
    parser.add_argument('--test_noisy', type=str, default="data/test/noisy_spec_filelist.txt")
    parser.add_argument('--train_noisy_wav', type=str, default="data/train/noisy_wav_filelist.txt")
    parser.add_argument('--test_noisy_wav', type=str, default="data/test/noisy_wav_filelist.txt")
    '''
    #####
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=4)  
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--loss_fn', type=str, default='l1')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--SEmodel', type=str, default='transformerencoder_03') 
    #####
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--train_num', type=int, default=None)
    #####
    parser.add_argument('--ASRmodel_path', type=str, default='data/newctcloss.model.acc.best.entire.pth')
    parser.add_argument('--alpha', type=float, default=0.3) #loss = (1 - self.alpha) * SEloss + self.alpha * ASRloss
    parser.add_argument('--asr_y_path', type=str, default='data/data_test.json,data/data_train_dev.json,data/data_train_nodev.json') 
    #####
    #parser.add_argument('--tr_c_dic', type=str, default='data/train/c_wavfolder_dic.npy') 
    #parser.add_argument('--ts_c_dic', type=str, default='data/test/c_wavfolder_dic.npy') 
    #####
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--target', type=str, default='MAP') #'MAP' or 'IRM'
    parser.add_argument('--task', type=str, default='DNS_SE') 
    parser.add_argument('--resume' , action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--re_epochs', type=int, default=300)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    args = get_path(args)
    return args

def get_path(args):
    args.checkpoint_path = f'out/checkpoint/{args.SEmodel}_{args.target}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                    f'lr{args.lr}.pth.tar'
    args.model_path = f'out/save_model/{args.SEmodel}_{args.target}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                    f'lr{args.lr}.pth.tar'
    args.score_path = f'out/Result/{args.SEmodel}_{args.target}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                    f'lr{args.lr}.csv'
    args.enhance_path = f'out/Enhanced/{args.SEmodel}/'
    return args


if __name__ == '__main__':
    # get current path
    cwd = os.path.dirname(os.path.abspath(__file__))
    print(cwd)
    print(SEED)
       
    # get and process arguments
    args = get_args()
    
    # tensorboard
    writer = SummaryWriter('out/logs')

    # load and construct the model
    semodel, epoch, best_loss, optimizer, secriterion, device = Load_SE_model(args)
    model = CombinedModel(args, semodel, secriterion)

    if args.mode == 'train':
        # load data into the Loader
        loader = Load_data(args)

    # control parameter
    for param in model.SEmodel.parameters():
        param.requires_grad = True
    for param in model.ASRmodel.parameters():
        param.requires_grad = False
    
    if args.retrain:
        args.epochs = args.re_epochs 
    
    # Trainer = Trainer(model, args.epochs, epoch, best_loss, optimizer, 
                    #   criterion, device, loader, args.test_noisy_wav, writer, model_path, score_path, args)
    try:
        if args.mode == 'train':
            train(model, args.epochs, epoch, best_loss, optimizer, 
                    device, loader,  writer, args.model_path, args)
        test(model, device, args.test_noisy, args.test_clean, args.enhance_path, args.score_path, args)
        
    except KeyboardInterrupt:
        state_dict = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss
            }
        check_folder(args.checkpoint_path)
        torch.save(state_dict, args.checkpoint_path)
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
