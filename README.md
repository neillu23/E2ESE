# End2End

#ESPnet 
1. python timit_wrd.py for changing TIMIT's word into broad phone labels.

2. run espnet/egs/timit/asr1/run.sh until stage 2 for preparing data.json.

3. python gen_fbank.py to generate the FBank feature in txt file.

4. copy-feats ark,t:fbank.txt ark,scp:fbank.ark,fbank.scp

5. python change_data.py to modify fbank feature in data.json.

6. run espnet/egs/timit/asr1/run.sh --stage 3 for new ASR model with new fbank.

7. copy the pre-trained model from espnet.
cp /espnet/egs/timit/asr_fbank/exp/train_nodev_pytorch_train/results/model.loss.best.entire PytorchSE/data/newctcloss.model.acc.best.entire.pth


#Pytorch SE
1.modify PATHS in the run.sh

2.sh run.sh 




