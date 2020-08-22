TRAIN_NOISY="datas/TIMIT_noisy_40hr_spec_filelist.txt"
TEST_NOISY="datas/TIMIT_noisy_40hr_wav_filelist.txt"
TEST_CLEAN="datas/TIMIT_Manner_E2E_wav_filelist.txt"

#Train model
python secode/main.py --train_path $TRAIN_NOISY --test_noisy $TEST_NOISY --test_clean $TEST_CLEAN
