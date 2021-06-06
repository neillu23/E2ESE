import os, argparse, torch, random, sys
from Trainer import train, test
from Load_model import Load_SE_model, Load_data, Load_data_VC
from utils.util import check_folder
from utils.load_asr_data import load_y_dict
from tensorboardX import SummaryWriter
from CombinedModel import CombinedModel, CombinedModel_VC
from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt
from espnet.asr.asr_utils import get_model_conf
import torch.backends.cudnn as cudnn
import pandas as pd
# import pdb

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
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
    parser.add_argument('--out_path', type=str, default='')
    #####
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=4)  
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--loss_fn', type=str, default='l1')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--SEmodel', type=str, default='transformerencoder_03') 
    #####
    parser.add_argument('--val_ratio', type=float, default = 0.1)
    parser.add_argument('--train_num', type=int, default = None)
    parser.add_argument('--test_num', type=int, default = None)
    #####
    parser.add_argument('--Espnet_path', type=str, default=None)
    parser.add_argument('--ASRmodel_path', type=str, default='data/ASRmodel.acc.best.entire')
    parser.add_argument('--VCmodel_path', type=str, default='data/TTSmodel.pretrained.entire')
    parser.add_argument('--VC_test_json', type=str, default=None)
    #####
    parser.add_argument('--alpha', type=float, default=0.001) #loss = (1 - self.alpha) * SEloss + self.alpha * ASRloss
    parser.add_argument('--alpha_epoch', type=int, default=70) # alpha = 0 when epoch < alpha_epoch
    parser.add_argument('--asr_y_path', type=str, default='data/data_test.json,data/data_train_dev.json,data/data_train_nodev.json') 
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--target', type=str, default='MAP') #'MAP' or 'IRM'
    parser.add_argument('--task', type=str, default='DNS_SE') 
    parser.add_argument('--resume' , action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--corpus', type=str, default="TIMIT") # corpus: TIMIT, TMHINT, TMHINT_DYS
    parser.add_argument('--asr_result', type=str, default=None)
    parser.add_argument('--tag', type=str, default="")
    parser.add_argument('--after_alpha_epoch', action='store_true') # on when test or retrain using after_alpha_epoch model
    parser.add_argument('--re_epochs', type=int, default=150)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    args = get_path(args)
    return args

def get_path(args):
    tag_str = ""
    if args.tag:
        tag_str = "_" + args.tag
    args.checkpoint_path = f'{args.out_path}/checkpoint/{args.SEmodel}{tag_str}_{args.target}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_alpha{args.alpha}_alpha_epoch{args.alpha_epoch}_batch{args.batch_size}_'\
                    f'lr{args.lr}.pth.tar'
    args.model_path = f'{args.out_path}/save_model/{args.SEmodel}{tag_str}_{args.target}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_alpha{args.alpha}_alpha_epoch{args.alpha_epoch}_batch{args.batch_size}_'\
                    f'lr{args.lr}.pth.tar'
    args.score_path = f'{args.out_path}/Result/{args.SEmodel}{tag_str}_{args.target}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_alpha{args.alpha}_alpha_epoch{args.alpha_epoch}_batch{args.batch_size}_'\
                    f'lr{args.lr}.csv'

    if args.after_alpha_epoch:
        args.model_path = args.model_path.replace("_alpha_epoch","_after_alpha_epoch")
        args.checkpoint_path = args.checkpoint_path.replace("_alpha_epoch","_after_alpha_epoch")
        args.score_path = args.score_path.replace("_alpha_epoch","_after_alpha_epoch")

    args.enhance_path = f'{args.out_path}/Enhanced/{args.SEmodel}/'
    return args


if __name__ == '__main__':
    # get current path
    cwd = os.path.dirname(os.path.abspath(__file__))
    print(cwd)
    print(SEED)
       
    # get and process arguments
    args = get_args()
    
    # tensorboard
    writer = SummaryWriter(f'{args.out_path}/logs')

    
    
    if args.corpus=="TMHINT_DYS":
        if args.mode == 'test':
            print("Error: Run test using the VC script!")
            exit()
            exit()
            exit()
        
        # TMHINT Train
        epoch=0
        args.checkpoint_path=args.checkpoint_path.replace("transformerencoder_03","VCmodel")
        args.model_path=args.model_path.replace("transformerencoder_03","VCmodel")
        args.score_path=args.score_path.replace("transformerencoder_03","VCmodel")
        
        if args.retrain or args.resume:
            idim, odim, train_args = get_model_conf(args.model_path, None)
            semodel, epoch, best_loss, optimizer, criterion, device = Load_SE_model(args, idim, odim, train_args)
            model = CombinedModel_VC(args,semodel)
        else:
            best_loss=1000
            device = torch.device(f'cuda:{args.gpu}')
            model = CombinedModel_VC(args)
        
        loader = Load_data_VC(args, model.SEmodel.args)
            
    else:
        # load and construct the model
        semodel, epoch, best_loss, optimizer, secriterion, device = Load_SE_model(args)
        model = CombinedModel(args, semodel, secriterion)
        if args.mode == 'train':
            loader = Load_data(args)
        else:
            asr_dict = load_y_dict(args)



    # control parameter
    if args.mode == 'train':
        for param in model.SEmodel.parameters():
            param.requires_grad = True
    else:
        for param in model.SEmodel.parameters():
            param.requires_grad = False

    for param in model.ASRmodel.parameters():
        param.requires_grad = False
    
    # if args.retrain:
    #     args.epochs = args.re_epochs 
    
    
    try:
        if args.mode == 'train':
            if args.corpus=="TMHINT_DYS":
                # --adim, default=384, type=int, "Number of attention transformation dimensions"
                optimizer = get_std_opt(model.SEmodel.parameters(), 384, model.SEmodel.args.transformer_warmup_steps, model.SEmodel.args.transformer_lr)
            train(model, args.epochs, epoch, best_loss, optimizer, 
                    device, loader,  writer, args.model_path, args)
        
        # mode=="test"
        else: 
            test(model, device, args.test_noisy, args.test_clean, asr_dict, args.enhance_path, args.score_path, args)
            
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
