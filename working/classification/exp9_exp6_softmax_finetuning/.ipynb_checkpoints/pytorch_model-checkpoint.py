import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn

import timm
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import AmplitudeToDB
from torchlibrosa.augmentation import DropStripes

import torchvision.transforms as T

from config import CFG

class BirdCLEF23Net(nn.Module):
    def __init__(self,
                 num_classes=CFG.num_classes,
                 model_name=CFG.model_name,
                 in_chans=CFG.in_chans,
                 sample_rate=CFG.sample_rate,
                 n_fft=CFG.n_fft,
                 hop_length=CFG.hop_length,
                 n_mels=CFG.n_mels,
                 top_db=CFG.top_db,
                 pretrained=True):
       
        super(BirdCLEF23Net, self).__init__()
        self.model=timm.create_model(pretrained=pretrained,
                                     model_name=model_name,
                                     in_chans=in_chans,
                                     num_classes=num_classes,)
        self.sr=sample_rate
        self.n_fft=n_fft
        self.hop_length=hop_length
        self.n_mels=n_mels
        self.num_classes=num_classes
        
        self.torchvision_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5)
            ])
        
        #self.to_melspec_fn = MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        #self.to_log_scale_fn = AmplitudeToDB(top_db=top_db)
        
        self.freq_dropper = DropStripes(dim=2, drop_width=int(CFG.freq_mask/2), 
            stripes_num=2)
        self.time_dropper = DropStripes(dim=3, drop_width=int(CFG.time_mask/2), 
            stripes_num=2)
    
    # def wav_to_logmel(self, x):
    #     # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    #     x = self.to_melspec_fn(x)
    #     # Convert to decibels
    #     x = self.to_log_scale_fn(x)
    #     return x
    
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
    
    def forward(self, x, targets=None):
        # wav has shape [channel, time]
        #wav = wav.unsqueeze(1)
        # wav augment
        #if self.training == True:
        #    wav = 
        #x = self.wav_to_logmel(wav)
        #plt.imshow(x[0,0,:,:].to('cpu'))
        #plt.show()
        # log-mel augment
        if self.training == True:
            x = self.torchvision_transforms(x)
            x, y = self.mixup(x, targets)
            # spec aug
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
