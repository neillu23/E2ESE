import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.normalization import LayerNorm
import pdb
import torch

class Conv(nn.Module):

    def __init__(self, in_chan, out_chan, kernal, dilation=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_chan, out_chan, kernal, dilation=dilation, padding=padding, groups=groups),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        out = self.conv(x)
        return out
    
class first_layer(nn.Module):

    def __init__(self,feature_size,num_convs):
        super().__init__()
        ker = 3
        layer = []
        for i,num_conv in enumerate(num_convs):
            layer.append(nn.ConstantPad1d((2,0), 0))
            if i==0:
                
                layer.append(Conv(feature_size,num_conv,ker))
            else:
                layer.append(Conv(num_convs[i-1],num_convs[i],ker))
            
        self.first = nn.Sequential(*layer)

    def forward(self, x):
        out = self.first(x)
        return out 

class masked_multihead_attention(nn.Module):
    def __init__(self,input_size=256, 
                        size_per_head=64,
                        num_heads=8, 
                        dropout_rate=0.1):
        super(masked_multihead_attention, self).__init__()
        self.size_per_head = size_per_head
        self.num_heads = num_heads
        num_units = size_per_head*num_heads
        self.q_layer = nn.Linear(input_size, num_units, bias=True)
        self.k_layer = nn.Linear(input_size, num_units, bias=True)
        self.v_layer = nn.Linear(input_size, num_units, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(p=dropout_rate)
        
    def masking(self, weight):
#         pdb.set_trace()
        mask = torch.triu(torch.ones_like(weight)).transpose(2,3)
        weight[mask==0]=float('-inf')
        
        return weight
        
    def forward(self,queries,keys):
        
        q = self.q_layer(queries)
        k = self.k_layer(keys)
        v = self.v_layer(keys)
        
        b,t,num_unit = q.shape
        
        q_ = torch.cat(torch.split(q.unsqueeze(1),self.size_per_head,dim=-1),dim=1)
        k_ = torch.cat(torch.split(k.unsqueeze(1),self.size_per_head,dim=-1),dim=1)
        v_ = torch.cat(torch.split(v.unsqueeze(1),self.size_per_head,dim=-1),dim=1)
        
        weights = torch.matmul(q_,k_.permute(0,1,3,2))
        weights = weights*(num_unit**(-0.5))
#         pdb.set_trace()
        weights = self.softmax(self.masking(weights))
        drop_weights = self.drop(weights)
        outputs = torch.matmul(drop_weights, v_)
        
        outputs = torch.cat(torch.split(outputs,1,dim=1),dim=-1).squeeze()
        
        return outputs,drop_weights

        
    
class encoder_layer(nn.Module):
    
    def __init__(self,input_size=256, size_per_head=64, num_heads=8, dropout_rate=0.1, intermediate_size=512):
        super(encoder_layer, self).__init__()
        self.attn = masked_multihead_attention(input_size, size_per_head, num_heads, dropout_rate)
        self.output1 = nn.Sequential(
            nn.Linear(size_per_head*num_heads,input_size),
            nn.Dropout(p=dropout_rate),         
            )
        self.norm1 = LayerNorm(input_size,eps=1e-3)
        self.intermediate = nn.Sequential(
            nn.Linear(input_size,intermediate_size),
            nn.LeakyReLU(negative_slope=0.2),    
            )
        self.output2 = nn.Sequential(
            nn.Linear(intermediate_size,input_size),
            nn.Dropout(p=dropout_rate),         
            )
        self.norm2 = LayerNorm(input_size,eps=1e-3)
        
    def forward(self,x):
        
        out,mask = self.attn(x,x)
        out = self.output1(out)
        out = self.norm1(x+out)
        interout = self.intermediate(out)
        interout = self.output1(interout)
        encoderout = self.norm2(out+interout)
        return encoderout
    
class transformerencoder(nn.Module):
    def __init__(self,):
        super(transformerencoder,self).__init__()

        num_hidden_layers   = 8
        feature_size         = 257
        intermediate_size   = 512
        dropout_rate        = 0.1
        num_heads           = 8
        size_per_head       = 64
        norm_mode           = 'None'
        act_func            = 'lrelu'

        num_convs = [1024,512,128,256]
        self.first_layer = first_layer(feature_size, num_convs)
        layer = []
        for i in range(num_hidden_layers):
            layer.append(encoder_layer(num_convs[-1], size_per_head, num_heads, dropout_rate, intermediate_size))
        self.encoder = nn.Sequential(*layer)
        
        self.last_layer = nn.Sequential(
            nn.Linear(num_convs[-1],feature_size),
            nn.ReLU(),
        )
    
    def forward(self,x):
#         print(x.shape)
        x = self.first_layer(x.permute(0,2,1))
#         print(x.shape)
        x = self.encoder(x.permute(0,2,1))
#         print(x.shape)
        x = self.last_layer(x)
#         print(x.shape)
        
        return x
    
class transformerencoder_02(nn.Module):
    def __init__(self,):
        super(transformerencoder_02,self).__init__()

        num_hidden_layers   = 8
        input_feature_size  = 289
        feature_size        = 257
        intermediate_size   = 512
        dropout_rate        = 0.1
        num_heads           = 8
        size_per_head       = 64
        norm_mode           = 'None'
        act_func            = 'lrelu'

        num_convs = [1024,512,128,256]
        self.first_layer = first_layer(input_feature_size, num_convs)
        layer = []
        for i in range(num_hidden_layers):
            layer.append(encoder_layer(num_convs[-1], size_per_head, num_heads, dropout_rate, intermediate_size))
        self.encoder = nn.Sequential(*layer)
        
        self.last_layer = nn.Sequential(
            nn.Linear(num_convs[-1],feature_size),
            nn.ReLU(),
        )
    
    def forward(self,x):
#         print(x.shape)
        x = self.first_layer(x.permute(0,2,1))
#         print(x.shape)
        x = self.encoder(x.permute(0,2,1))
#         print(x.shape)
        x = self.last_layer(x)
#         print(x.shape)
        
        return x
    
class transformerencoder_03(nn.Module):
    def __init__(self,):
        super(transformerencoder_03,self).__init__()

        num_hidden_layers   = 8
        input_feature_size  = 257
        feature_size        = 257
        intermediate_size   = 512
        dropout_rate        = 0.1
        num_heads           = 8
        size_per_head       = 64
        norm_mode           = 'None'
        act_func            = 'lrelu'

        num_convs = [1024,512,128,256]
        self.first_layer = first_layer(input_feature_size, num_convs)
        layer = []
        for i in range(num_hidden_layers):
            layer.append(encoder_layer(num_convs[-1], size_per_head, num_heads, dropout_rate, intermediate_size))
        self.encoder = nn.Sequential(*layer)
        
        self.last_layer = nn.Sequential(
            nn.Linear(num_convs[-1],feature_size),
            nn.ReLU(),
        )
    
    def forward(self,x):
#         print(x.shape)
        x = self.first_layer(x.permute(0,2,1))
#         print(x.shape)
        x = self.encoder(x.permute(0,2,1))
#         print(x.shape)
        x = self.last_layer(x)
#         print(x.shape)
        
        return x
    
    
class transformerencoder_04(nn.Module):
    def __init__(self,):
        super(transformerencoder_04,self).__init__()

        num_hidden_layers   = 8
        input_feature_size  = 289
        feature_size        = 257
        intermediate_size   = 512
        dropout_rate        = 0.1
        num_heads           = 8
        size_per_head       = 64
        norm_mode           = 'None'
        act_func            = 'lrelu'

        num_convs = [1024,512,128,256]
        self.first_layer = first_layer(input_feature_size, num_convs)
        layer = []
        for i in range(num_hidden_layers):
            layer.append(encoder_layer(num_convs[-1], size_per_head, num_heads, dropout_rate, intermediate_size))
        self.encoder = nn.Sequential(*layer)
        
        self.last_layer = nn.Sequential(
            nn.Linear(num_convs[-1],feature_size),
            nn.ReLU(),
        )
    
    def forward(self,x):
#         print(x.shape)
        x = self.first_layer(x.permute(0,2,1))
#         print(x.shape)
        x = self.encoder(x.permute(0,2,1))
#         print(x.shape)
        x = self.last_layer(x)
#         print(x.shape)
        
        return x
