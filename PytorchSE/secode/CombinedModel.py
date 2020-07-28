import torch.nn as nn

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

