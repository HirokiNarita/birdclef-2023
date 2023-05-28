import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import timm
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import AmplitudeToDB
from torchlibrosa.augmentation import DropStripes

import torchvision.transforms as T

from config import CFG

class Wav2Logmel(nn.Module):
    def __init__(self,
                 sample_rate=CFG.sample_rate,
                 n_fft=CFG.n_fft,
                 hop_length=CFG.hop_length,
                 n_mels=CFG.n_mels,
                 top_db=CFG.top_db,):
       
        super(Wav2Logmel, self).__init__()
        
        self.sr=sample_rate
        self.n_fft=n_fft
        self.hop_length=hop_length
        self.n_mels=n_mels
        
        self.to_melspec_fn = MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        self.to_log_scale_fn = AmplitudeToDB(top_db=top_db)
        
    def wav_to_logmel(self, x):
        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        x = self.to_melspec_fn(x)
        # Convert to decibels
        x = self.to_log_scale_fn(x)
        return x
    
    def forward(self, wav):
        # wav has shape [channel, time]
        #print(wav.shape)
        wav = wav.unsqueeze(1)
        logmel = self.wav_to_logmel(wav)
        #print(logmel[0])
        # print(logmel.shape)
        # plt.imshow(logmel[0,0,:,:].to('cpu'))
        # plt.show()
        return logmel
