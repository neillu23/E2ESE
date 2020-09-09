ESPNET_PATH="/mnt/Data/user_vol_2/user_neillu/End2End/espnet/egs/timit/asr_normwave_fbank/"
CLEAN_TIMIT="/mnt/Data/user_vol_2/user_neillu/TIMIT_Manner_E2E/"
DATA_PATH="/mnt/Data/user_vol_2/user_neillu/TIMIT_fbank/"


# step 1: generate normalized fbank
mkdir out/
python make_list.py  $CLEAN_TIMIT out/TIMIT_filelist.txt
python gen_fbank.py  out/TIMIT_filelist.txt $DATA_PATH/timit_fbank.txt
copy-feats ark,t:$DATA_PATH/timit_fbank.txt ark,scp:$DATA_PATH/timit_fbank.ark,$DATA_PATH/timit_fbank.scp

# step 2: change data.json 
python change_data.py $DATA_PATH/timit_fbank.scp $ESPNET_PATH/dump/test/deltafalse/data.json out/data_test.json
python change_data.py $DATA_PATH/timit_fbank.scp $ESPNET_PATH/dump/train_dev/deltafalse/data.json out/data_train_dev.json
python change_data.py $DATA_PATH/timit_fbank.scp $ESPNET_PATH/dump/train_nodev/deltafalse/data.json out/data_train_nodev.json
cp out/data_test.json $ESPNET_PATH/dump/test/deltafalse/data.json
cp out/data_train_dev.json $ESPNET_PATH/dump/train_dev/deltafalse/data.json 
cp out/data_train_nodev.json $ESPNET_PATH/dump/train_nodev/deltafalse/data.json 

# step 3: train & test ASR model
cd $ESPNET_PATH/
./run.sh --stage 3 

# step 4: copy model from espnet and run SE-ASR
cd -
cd ../PytorchSE
cp $ESPNET_PATH/exp/train_nodev_pytorch_train/results/model.loss.best.entire data/newctcloss.model.acc.best.entire.pth
sh run.sh
