
stage=1

# Input waveform data
TRAIN_NOISY="/home/neillu/Desktop/Workspace/TIMIT_noisy_40hr_wav/TrNoisy/"
TRAIN_CLEAN="/home/neillu/Downloads/TIMIT_Manner_E2E/TRAIN/"
TEST_NOISY="/home/neillu/Downloads/NewTsNoisy"
TEST_CLEAN="/home/neillu/Downloads/TIMIT_Manner_E2E/TEST/"

# Output spectrum path
SPEC_PATH="/home/neillu/Desktop/Workspace/TIMIT_Manner_E2E_spec/"



if [ ${stage} -eq 0 ]; then
    echo "stage 0 : Data preprocessing"
    python preprocess/gen_npy.py  --data 'trdata' --noisy_wav_path $TRAIN_NOISY --clean_wav_path $TRAIN_CLEAN --spec_path $SPEC_PATH
    python preprocess/gen_npy.py  --data 'tsdata' --noisy_wav_path $TEST_NOISY --clean_wav_path $TEST_CLEAN --spec_path $SPEC_PATH
fi


if [ ${stage} -le 1 ]; then
    echo "stage 1 : Training"
    python secode/main.py --mode 'train' --spec_path $SPEC_PATH --train_clean $TRAIN_CLEAN 
fi

if [ ${stage} -le 2 ]; then
    echo "stage 2 : Testing"
    python secode/main.py --mode 'test' --test_noisy $TEST_NOISY --test_clean $TEST_CLEAN
fi