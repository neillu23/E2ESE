import torch.nn as nn
import numpy

class CombinedModel(nn.Module):
    def __init__(self, SEmodel, ASRmodel, criterion, alpha):
        super(CombinedModel, self).__init__()
        self.SEmodel = SEmodel
        self.ASRmodel = ASRmodel
        self.criterion = criterion
        self.alpha = alpha
        #self.preprocess = nn.Sequential()

        
    def forward(self, a,b,c):
        SEloss=0
        '''
        enhanced = self.SEmodel(noisy)
        SEloss = self.criterion(enhanced, clean)
        x = data_prepare(preprocess(enhanced))
        '''
        ASRloss = self.ASRmodel(a,b,c)
        loss = SEloss + self.alpha * ASRloss
        return loss

class Fbank(nn.Module):
    def __init__(self,sample_rate=16000,NFFT=400,nfilt=26):
        self.sample_rate = sample_rate
        self.NFFT = NFFT
        self.nfilt = nfilt

    def forward(pow_frames):
        low_freq_mel = 0
        high_freq_mel = (2595 * numpy.log10(1 + (self.sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = numpy.floor((self.NFFT + 1) * hz_points / self.sample_rate)

        fbank = numpy.zeros((self.nfilt, int(numpy.floor(self.NFFT / 2 + 1))))
        for m in range(1, self.nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        #need to modify numpy.dot to torch
        filter_banks = numpy.dot(pow_frames, fbank.T)
        filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * numpy.log10(filter_banks)  # dB
        return filter_banks