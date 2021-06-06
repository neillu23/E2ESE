ESPNET_PATH="/data1/user_chiayow/espnet/egs/timit/asr_dys_m_vcfeature"
VC_ESPNET_PATH="/data1/user_chiayow/vc_espnet/espnet/egs/dysarthric-tmsv/vc1"

#Add Dysarthric speech download path command provided by Wen-Chin Huang
DYS_DOWNLOAD="https://drive.google.com/file/d/1V4rxgnm3MsNiBcaLJzyGWBj1kxW8jeXs/view?usp=sharing"

E2E_PATH=$PWD

#1. 
#Add lines in /vc_espnet/espnet/espnet/vc/pytorch_backend/vc.py line548:
#torch.save(model, os.path.join(args.outdir,"TTSmodel.pretrained.entire"),_use_new_zipfile_serialization=False)
#print("save model to", os.path.join(args.outdir,"TTSmodel.pretrained.entire"))
#exit()
#2.
#Add lines in /espnet/espnet/nets/pytorch_backend/e2e_vc_transformer.py
#In def __init__ (line 462): self.args = args
#In def forward (line 723): self.after_outs=after_outs

# step -1: Get the TTS pre-trained model from the VC script
cd $VC_ESPNET_PATH
./run.sh --stage -1 --stop_stage 4 --cmvn downloads/cospro_tts_pt/data/train/cmvn.ark --dysarthric_download_url https://drive.google.com/open?id=$DYS_DOWNLOAD
CUDA_VISIBLE_DEVICES=9 ./run.sh --stage 5 --stop_stage 5 --tag test
cp $VC_ESPNET_PATH/exp/dysarthric_all_pytorch_test/results/model.json data/model.json
cp $VC_ESPNET_PATH/exp/dysarthric_all_pytorch_test/results/PretrainedTTS.model.entire data/VCmodel.loss.best.entire


# step 0: run the first time ESPnet
cd $ESPNET_PATH
rm -r data/ dump/ exp/ fbank/ tensorboard/
./run.sh --stop_stage 2 

cd $E2E_PATH
mkdir out/

# step 2: change data.json 
mkdir out/backup
python change_data_vc.py $VC_ESPNET_PATH/dump/dysarthric_all_eval_cospro/data.json $VC_ESPNET_PATH/dump/dysarthric_all_dev_cospro/data.json $VC_ESPNET_PATH/dump/dysarthric_all_train_cospro/data.json $ESPNET_PATH/dump/test/deltafalse/data.json out/data_test.json
python change_data_vc.py $VC_ESPNET_PATH/dump/dysarthric_all_eval_cospro/data.json $VC_ESPNET_PATH/dump/dysarthric_all_dev_cospro/data.json $VC_ESPNET_PATH/dump/dysarthric_all_train_cospro/data.json $ESPNET_PATH/dump/train_dev/deltafalse/data.json out/data_train_dev.json
python change_data_vc.py $VC_ESPNET_PATH/dump/dysarthric_all_eval_cospro/data.json $VC_ESPNET_PATH/dump/dysarthric_all_dev_cospro/data.json $VC_ESPNET_PATH/dump/dysarthric_all_train_cospro/data.json $ESPNET_PATH/dump/train_nodev/deltafalse/data.json out/data_train_nodev.json

cp $ESPNET_PATH/dump/test/deltafalse/data.json out/backup/test_data.json 
cp $ESPNET_PATH/dump/train_dev/deltafalse/data.json out/backup/train_dev_data.json 
cp $ESPNET_PATH/dump/train_nodev/deltafalse/data.json out/backup/train_nodev_data.json 

cp out/data_test.json $ESPNET_PATH/dump/test/deltafalse/data.json 
cp out/data_train_dev.json $ESPNET_PATH/dump/train_dev/deltafalse/data.json 
cp out/data_train_nodev.json $ESPNET_PATH/dump/train_nodev/deltafalse/data.json 

cp out/data_test.json ../PytorchSE/data/
cp out/data_train_dev.json ../PytorchSE/data/
cp out/data_train_nodev.json ../PytorchSE/data/


# step 3: train & test ASR model
cd $ESPNET_PATH/
./run.sh --stage 3 

# step 4: copy model from espnet and run SE-ASR
cd $E2E_PATH
cd ../PytorchSE
cp $ESPNET_PATH/exp/train_nodev_pytorch_train/results/model.acc.best.entire data/ASRmodel.acc.best.entire
sh run_vc.sh
