import torch
import torch.nn as nn


class Blstm(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers=1,dropout=0):
        super().__init__()
        self.blstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout, bidirectional=True)

    def forward(self,x):

        out,_=self.blstm(x)
        out = out[:,:,:int(out.size(-1)/2)]+out[:,:,int(out.size(-1)/2):]  #merge_mode = 'sum'
        return out

class lstm(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers=1,dropout=0):
        super().__init__()
        self.blstm = nn.Sequential(
            nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout, bidirectional=False),
        )

    def forward(self,x):

        out,_=self.blstm(x)

        return out


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)


        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y
    

    
class BLSTM_03(nn.Module):
    
    def __init__(self,):
        super().__init__()
        
        self.lstm_enc = nn.Sequential(
            lstm(input_size=201, hidden_size=201, num_layers=3),
#             nn.ReLU(),
#             TimeDistributed(nn.Linear(500, 201, bias=True))
        )
    
    def forward(self,x):
        x = self.lstm_enc(x)
        
        return x
    
class BLSTM_02(nn.Module):
    
    def __init__(self,):
        super().__init__()
        
        self.lstm_enc = nn.Sequential(
            Blstm(input_size=201, hidden_size=700, num_layers=3),
            TimeDistributed(nn.Linear(700, 201, bias=True))
        )
    
    def forward(self,x):
        x = self.lstm_enc(x)
        
        return x



class BLSTM_01(nn.Module):
    
    def __init__(self,):
        super().__init__()
        
        self.lstm_enc = nn.Sequential(
            Blstm(input_size=201, hidden_size=300, num_layers=2),
            nn.Linear(300, 201, bias=True)
        )
    
    def forward(self,x):
        x = self.lstm_enc(x)
        
        return x
    
