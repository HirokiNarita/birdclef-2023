# implements by https://www.kaggle.com/code/nischaydnk/birdclef-2023-pytorch-lightning-inference
import copy
import sys

import numpy as np
import matplotlib.pyplot as plt
import librosa as lb
from tinytag import TinyTag

import torch
import torchaudio
from torch.utils.data import Dataset

from config import CFG

def compute_melspec(y, sr, n_fft, hop_length, n_mels, fmin, fmax, power=2.0):
    """
    Computes a mel-spectrogram and puts it at decibel scale
    Arguments:
        y {np array} -- signal
        params {AudioParams} -- Parameters to use for the spectrogram. Expected to have the attributes sr, n_mels, f_min, f_max
    Returns:
        np array -- Mel-spectrogram
    """
    mel_spectrogram = lb.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            power=power,
        )
    # convert melspectrogram to log mel energies
    melspec = (
        20.0 / power * np.log10(np.maximum(mel_spectrogram, sys.float_info.epsilon))
    )
    return melspec

def mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)
    
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V

def crop_or_pad(y, length, is_train=True, start=None):
    if len(y) < length:
        y = np.concatenate([y, np.zeros(length - len(y))])
        
        n_repeats = length // len(y)
        epsilon = length % len(y)
        
        y = np.concatenate([y]*n_repeats + [y[:epsilon]])
        
    elif len(y) > length:
        if not is_train:
            start = start or 0
        else:
            start = start or np.random.randint(len(y) - length)

        y = y[start:start + length]
    
    else:
        return y

    return y


def random_float(low=0.0, high=1.0):
    return np.random.uniform(low, high)

def add_gaussian_noise(array, std):
    noise = np.random.normal(0, std, array.shape)
    return array + noise

def GaussianNoise(audio, std=[0.0025, 0.025], prob=CFG.gn_prob):
    # 指定された範囲内でガウシアンノイズの標準偏差のランダムな値を選択します
    std = random_float(std[0], std[1])

    # 確率`prob`でランダムにガウシアンノイズを適用します
    if random_float() < prob:
        # オーディオ信号にランダムなガウシアンノイズを追加します
        audio = add_gaussian_noise(audio, std)
    return audio

class BirdDataset(Dataset):
    def __init__(self,
                 data,
                 sr=CFG.sample_rate,
                 n_fft=CFG.n_fft,
                 n_mels=CFG.n_mels,
                 hop_length=CFG.hop_length,
                 fmin=0,
                 fmax=None,
                 duration=CFG.duration,
                 step=None,
                 is_train=True,
                 is_test=False):
        
        self.data = data
        
        self.sr = sr
        
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr//2
        self.hop_length = hop_length

        self.duration = duration
        self.audio_length = self.duration*self.sr
        self.is_train = is_train
        self.step = step or self.audio_length
        
        # for test
        self.is_test = is_test

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def normalize(image):
        image = image.astype("float32", copy=False) / 255.0
        return image
    
    # def audio_to_image(self, audio):
    #     image = compute_melspec(audio, self.sr, self.n_fft, self.hop_length, self.n_mels, self.fmin, self.fmax)
    #     #image = mono_to_color(melspec)
    #     #image = self.normalize(image)
    #     return image
    def get_audio_offset(self, filepath, duration=5, start=None):
        tag = TinyTag.get(filepath)
        audio_length = tag.duration
        high = audio_length - duration
        if high < 1:
            start = 0
        elif high > 1:
            if not self.is_train:
                start = start or 0
            else:
                start = start or np.random.randint(high)
        return start

    def read_file(self, filepath):
        # get random offset
        if self.is_train == True:
            offset = self.get_audio_offset(filepath, duration=self.duration)
            audio_org, sr = lb.load(filepath, offset=offset, duration=self.duration, sr=self.sr, mono=True)
            audio_org = GaussianNoise(audio_org)
        else:
            if self.is_test == True:
                # for test
                audio_org, sr = lb.load(filepath, sr=self.sr, mono=True)
            else:
                # for valid
                audio_org, sr = lb.load(filepath, offset=0, duration=self.duration, sr=self.sr, mono=True)
        # train
        if self.is_train == True:
            # adjust audio_org length
            audio = crop_or_pad(audio_org, length=self.audio_length, is_train=self.is_train)
            audio = torch.from_numpy(audio)
        # inference
        else:
            audios = []
            # make subseq
            for idx, i in enumerate(range(0, len(audio_org), self.audio_length)):
                # index
                start = i
                end = start + self.audio_length
                # crop
                audio = audio_org[start:end]
                # 長さが5秒に満たない場合は切り捨て
                if (len(audio) < self.audio_length) and (idx != 0):
                    continue
                audio = crop_or_pad(audio, length=self.audio_length, is_train=self.is_train)
                # to spectrogram
                #image = self.audio_to_image(audio_)
                audio = torch.from_numpy(audio)
                
                audios.append(audio)
            audio = torch.stack(audios, dim=0)
        
        return audio
        
    def __getitem__(self, idx):
        sample_info = self.data.loc[idx]
        features = self.read_file(sample_info['filepath'])
        features = features.float()
        sample_info = sample_info.to_dict()
        
        if self.is_train == False:
            sample_info_tmp = []
            for i in range(len(features)):
                sample_info_ = copy.deepcopy(sample_info)
                sample_info_tmp.append(sample_info_)
            sample_info = sample_info_tmp
            
        return features, sample_info