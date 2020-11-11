
stage=0

# Input waveform data
TRAIN_NOISY="/home/neillu/Desktop/Workspace/TIMIT_noisy_40hr_wav/TrNoisy/"
TRAIN_CLEAN="/home/neillu/Downloads/TIMIT_Manner_E2E/TRAIN/"
TEST_NOISY="/home/neillu/Downloads/NewTsNoisy"
TEST_CLEAN="/home/neillu/Downloads/TIMIT_Manner_E2E/TEST/"

# Output spectrum path
OUT_PATH="/home/neillu/Desktop/Workspace/E2E"


if [ ${stage} -eq 0 ]; then
    echo "stage 0 : Data preprocessing"
    python preprocess/gen_npy.py  --data 'trdata' --noisy_wav_path $TRAIN_NOISY --clean_wav_path $TRAIN_CLEAN --out_path $OUT_PATH
    python preprocess/gen_npy.py  --data 'tsdata' --noisy_wav_path $TEST_NOISY --clean_wav_path $TEST_CLEAN --out_path $OUT_PATH
fi


if [ ${stage} -le 1 ]; then
    echo "stage 1 : Training"
    python secode/main.py --mode 'train' --out_path $OUT_PATH --train_clean $TRAIN_CLEAN --train_num 4000
fi

if [ ${stage} -le 2 ]; then
    echo "stage 2 : Testing"
    python secode/main.py --mode 'test' --out_path $OUT_PATH --test_noisy $TEST_NOISY --test_clean $TEST_CLEAN --after_alpha_epoch
fi

if [ ${stage} -le 3 ]; then
    echo "stage 3 : Scoring"
    python preprocess/cal_score.py $OUT_PATH/Result/transformerencoder_03_10000Normal_MAP_epochs150_adam_l1_alpha0.0_after_alpha_epoch0.0_batch4_lr5e-05.csv $OUT_PATH/Score/
fi