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
                 top_db=CFG.top_db,
                 device='cpu'):
       
        super(Wav2Logmel, self).__init__()
        
        self.sr=sample_rate
        self.n_fft=n_fft
        self.hop_length=hop_length
        self.n_mels=n_mels
        
        self.to_melspec_fn = MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        self.to_log_scale_fn = AmplitudeToDB(top_db=top_db)
        # mean = (0.485,)  # R only for RGB
        # std = (0.229,)  # R only for RGB
        # self.normalize = T.Compose([
        #     T.Normalize(mean, std)
        #     ])
        
    def wav_to_logmel(self, x):
        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        x = self.to_melspec_fn(x)
        # Convert to decibels
        x = self.to_log_scale_fn(x)
        return x
    
    def mono_to_color_tensor(self, X, eps=1e-6, mean=None, std=None):
        """
        Converts a one channel tensor in [0, 255]
        Arguments:
            X {torch tensor [B x 1 x H x W]} -- 2D tensor to convert
        Keyword Arguments:
            eps {float} -- To avoid dividing by 0 (default: {1e-6})
            mean {None or torch tensor} -- Mean for normalization (default: {None})
            std {None or torch tensor} -- Std for normalization (default: {None})
        Returns:
            torch tensor [B x 1 x H x W] -- RGB torch tensor
        """
        # X = torch.cat([X, X, X], dim=1)

        # Standardize
        mean = mean or X.mean(dim=[1,2,3], keepdim=True)
        std = std or X.std(dim=[1,2,3], keepdim=True)
        X = (X - mean) / (std + eps)

#         # Normalize to [0, 255
#         _min = X.reshape(X.shape[0], -1).min(dim=1)[0].reshape(X.shape[0], 1, 1, 1)
#         _max = X.reshape(X.shape[0], -1).max(dim=1)[0].reshape(X.shape[0], 1, 1, 1)
#         #print(_min.shape)

#         if ((_max - _min) > eps).all():
#             V = torch.clamp(X, _min, _max)
#             V = 255 * (V - _min) / (_max - _min)
#             V = V.to(torch.uint8)
#         else:
#             V = torch.zeros_like(X, dtype=torch.uint8)

        return X
    
    def forward(self, wav):
        # wav has shape [channel, time]
        #print(wav.shape)
        wav = wav.unsqueeze(1)
        logmel = self.wav_to_logmel(wav)
        # logmel = self.mono_to_color_tensor(logmel)
        # logmel = self.normalize(logmel)
        # print(logmel)
        return logmel
