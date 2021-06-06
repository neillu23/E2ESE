
stage=1

OUT_PATH="/data1/user_chiayow/data/VC_out_TTSpretrained/"
VC_ESPNET_PATH="/data1/user_chiayow/vc_espnet/espnet/egs/dysarthric-tmsv/vc1"
mkdir $OUT_PATH/save_model/
cp data/model.json $OUT_PATH/save_model/


#stage 1 : Training
if [ ${stage} -le 1 ]; then
    echo "stage 1 : Training"
    python secode/main.py --mode 'train' --out_path $OUT_PATH  --corpus "TMHINT_DYS" --alpha 0.0 \
                          --alpha_epoch 2001 --re_epochs 2000 --epochs 2000 --batch_size 16 --lr 0.001 --Espnet_path $VC_ESPNET_PATH \
                          --VC_test_json $VC_ESPNET_PATH/dump/dysarthric_all_eval_cospro/data.json #--retrain
    #python secode/main.py --mode 'train' --out_path $OUT_PATH  --corpus "TMHINT_DYS" --alpha 0.001 \
    #                      --alpha_epoch 1000 --re_epochs 2000 --epochs 2000 --batch_size 16 --lr 0.001 \ 
    #                      --Espnet_path $VC_ESPNET_PATH \
    #                      --VC_test_json $VC_ESPNET_PATH/dump/dysarthric_all_eval_cospro/data.json
fi

stage 2 : Testing (Decoding & Scoring)
if [ ${stage} -le 2 ]; then
cd $VC_ESPNET_PATH
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 6 --tag test --cmvn downloads/cospro_tts_pt/data/train/cmvn.ark \
    --model $OUT_PATH/save_model/param_VCmodel_MAP_epochs2000_adam_l1_alpha0.0_alpha_epoch2001.0_batch16_lr0.001.pth.tar
fi
