import torch.nn as nn
import torch
import numpy
import math

class CombinedModel(nn.Module):
    def __init__(self, SEmodel, ASRmodel, SEcriterion, alpha):
        super(CombinedModel, self).__init__()
        self.SEmodel = SEmodel
        self.ASRmodel = ASRmodel
        self.SEcriterion = SEcriterion
        self.alpha = alpha
        self.Fbank = Fbank
             
    def forward(self, noisy, clean, ilen, y):
        enhanced = self.SEmodel(noisy)
        SEloss = self.SEcriterion(enhanced, clean)
        if math.isinf(SEloss):
            print('SEloss is infinity:',SEloss)
            print('Enhanced:',enhanced)
            print('Clean:',clean)

        Fbank=self.Fbank()
        enhanced_fbank = Fbank.forward(enhanced)
        ASRloss = self.ASRmodel(enhanced_fbank,ilen,y)
        
        
        if math.isinf(ASRloss):
            print('enhanced:',enhanced)
            print('enhanced_fbank:',enhanced_fbank)
            print('ASRloss is infinity:',ASRloss)
            
        

        loss = SEloss + self.alpha * ASRloss
        return loss

class Fbank(nn.Module):
    def __init__(self,sample_rate=16000,NFFT=400,nfilt=26,gpu=0):
        super(Fbank, self).__init__()
        self.device    = torch.device(f'cuda:{gpu}')
        self.sample_rate = sample_rate
        self.NFFT = NFFT
        self.nfilt = nfilt

    def forward(self, pow_frames):
        low_freq_mel = 0
        high_freq_mel = (2595 * numpy.log10(1 + (self.sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = numpy.linspace(low_freq_mel, high_freq_mel, self.nfilt + 2)  # Equally spaced in Mel scale
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
        fbank = torch.tensor(fbank, dtype=torch.float)
        filter_banks = torch.matmul(pow_frames, torch.transpose(fbank, 0, 1).to(self.device))
        # filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
        # filter_banks = 20 * torch.log(filter_banks)  # dB
        # m = torch.nn.LayerNorm(filter_banks.size()[1:]).to(self.device)
        # filter_banks = m(filter_banks)
        return torch.tensor(filter_banks)

