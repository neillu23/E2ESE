import torch
import numpy



def spec2fbank(pow_frames):
    sample_rate=16000
    NFFT=400
    nfilt=26
    # pow_frames = torch.load("/mnt/Data/user_vol_2/user_neillu/E2E_Spec/training/train_clean/fsdc0_sx232_0.pt")
    # pow_frames = torch.transpose(pow_frames, 0, 1)
    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    #need to modify numpy.dot to torch
    fbank = torch.tensor(fbank, dtype=torch.float)
    filter_banks = torch.matmul(pow_frames, torch.transpose(fbank, 0, 1))
    #torch.nn.LayerNorm
    m = torch.nn.LayerNorm(filter_banks.size()[0])
    filter_banks = torch.transpose(m(torch.transpose(filter_banks,0,1)),0,1)
    return filter_banks
# filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
# filter_banks = 20 * torch.log(filter_banks)  # dB
# print(filter_banks.shape)
# print(filter_banks)
# print(filter_banks[0])
# print(torch.transpose(filter_banks, 0, 1)[0])
