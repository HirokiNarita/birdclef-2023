import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

import timm
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import AmplitudeToDB
from torchlibrosa.augmentation import DropStripes

from config import CFG

class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.training:
            with torch.no_grad():
                B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
                B_avg = torch.sum(B_avg) / input.size(0)
                # print(B_avg)
                theta_med = torch.median(theta[one_hot == 1])
                self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return output

class BirdCLEF23Net(nn.Module):
    def __init__(self,
                 num_classes=CFG.num_classes,
                 model_name=CFG.model_name,
                 in_chans=CFG.in_chans,
                 sample_rate=CFG.sample_rate,
                 n_fft=CFG.n_fft,
                 hop_length=CFG.hop_length,
                 n_mels=CFG.n_mels,
                 top_db=CFG.top_db):
       
        super(BirdCLEF23Net, self).__init__()
        self.model=timm.create_model(pretrained=True,
                                     model_name=model_name,
                                     in_chans=in_chans,
                                     num_classes=num_classes,)
        self.sr=sample_rate
        self.n_fft=n_fft
        self.hop_length=hop_length
        self.n_mels=n_mels
        self.num_classes=num_classes
        self.adacos = Adacos()
        
        self.to_melspec_fn = MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        self.to_log_scale_fn = AmplitudeToDB(top_db=top_db)
        
        self.freq_dropper = DropStripes(dim=2, drop_width=int(CFG.freq_mask/2), 
            stripes_num=2)
        self.time_dropper = DropStripes(dim=3, drop_width=int(CFG.time_mask/2), 
            stripes_num=2)
        
    
    def wav_to_logmel(self, x):
        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        x = self.to_melspec_fn(x)
        # Convert to decibels
        x = self.to_log_scale_fn(x)
        return x
    
    def mixup(self, data, label, alpha=0.5, device='cuda:0'):
        def _aug(data1, data2, weights):
            batch_size = len(data1)
            mix_data = []
            for i in range(batch_size):
                sample = data1[i]*weights[i] + data2[i]*(1 - weights[i])
                mix_data.append(sample)
            
            mix_data = torch.stack(mix_data)

            return mix_data
        
        batch_size = len(data)
        weights = torch.from_numpy(np.random.uniform(low=0, high=alpha, size=batch_size)).to(device)
        index = np.random.permutation(batch_size)
        x1, x2 = data, data[index]
        y1, y2 = label, label[index]

        x = _aug(x1, x2, weights)
        y = _aug(y1, y2, weights)

        return x, y
    
    def forward(self, wav, targets=None):
        # wav has shape [channel, time]
        wav = wav.unsqueeze(1)
        # wav augment
        #if self.training == True:
        #    wav = 
        x = self.wav_to_logmel(wav)
        #plt.imshow(x[0,0,:,:].to('cpu'))
        #plt.show()
        # log-mel augment
        if self.training == True:
            # TODO mixup
            x, y = self.mixup(x, targets)
            # TODO specaug
            x = self.time_dropper(self.freq_dropper(x))
        # plt.imshow(x[0,0,:,:].to('cpu'))
        # plt.show()
        # feature extractor
        #x = self.model.forward_features(x)
        x = self.model(x)
        if self.training == True:
            return x, y
        else:
            return x
