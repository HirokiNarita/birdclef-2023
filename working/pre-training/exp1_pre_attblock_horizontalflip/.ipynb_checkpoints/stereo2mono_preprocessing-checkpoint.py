# implements by https://www.kaggle.com/code/nischaydnk/birdclef-2023-pytorch-lightning-inference
import copy
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import librosa as lb
import soundfile as sf
from pydub import AudioSegment
import ffmpeg

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

def crop_or_pad(y, length, is_train=True, start=None):
    if len(y) < length:
        y = torch.cat([y, torch.zeros(length - len(y))])
        
        n_repeats = length // len(y)
        epsilon = length % len(y)
        
        y = torch.cat([y]*n_repeats + [y[:epsilon]])
        
    elif len(y) > length:
        if not is_train:
            start = start or 0
        else:
            start = start or np.random.randint(len(y) - length)

        y = y[start:start + length]
    
    else:
        return y

    return y

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
                 is_train=True):
        
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

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def normalize(image):
        image = image.astype("float32", copy=False) / 255.0
        return image
    
    def audio_to_image(self, audio):
        image = compute_melspec(audio, self.sr, self.n_fft, self.hop_length, self.n_mels, self.fmin, self.fmax)
        #image = mono_to_color(melspec)
        #image = self.normalize(image)
        return image

    def read_file(self, filepath):
        audio, sr = lb.load(filepath, sr=self.sr, mono=True)
        #audio, sr = torchaudio.load(filepath)
        # # train
        # if self.is_train == True:
        #     # adjust audio length
        #     audio = crop_or_pad(audio, length=self.audio_length, is_train=self.is_train)
        #     #images = self.audio_to_image(audio)
        #     #images = torch.from_numpy(images)
        # # inference
        # else:
        #     images = []
        #     # make subseq
        #     for idx, i in enumerate(range(0, len(audio), self.audio_length)):
        #         # index
        #         start = i
        #         end = start + self.audio_length
        #         # crop
        #         audio_ = audio[start:end]
        #         # 長さが5秒に満たない場合は切り捨て
        #         if (len(audio_) < self.audio_length) and (idx != 0):
        #             continue
        #         audio_ = crop_or_pad(audio_, length=self.audio_length, is_train=self.is_train)
        #         # to spectrogram
        #         image = self.audio_to_image(audio_)
        #         image = torch.from_numpy(image)
                
        #         images.append(image)
        #     images = torch.stack(images, dim=0)
        
        return audio
    
    def torchaudio_read(self, filepath):
        ext = os.path.splitext(filepath)[1]
        ext = ext.replace('.', '')
        try:
            audio_org, sr = torchaudio.load(filepath, format=ext)
        except:
            print('error_file: ',filepath, flush=True)
        #print(filepath, 'shape:', audio_org.shape, 'offset:', offset)
        # to sr=32k and to mono
        resampler = torchaudio.transforms.Resample(sr, self.sr)
        audio_org = resampler(audio_org)
        return audio_org

    def convert_to_wav(self, input_file, output_file):
        audio = AudioSegment.from_file(input_file)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(self.sr)
        audio.export(output_file, format="wav")
        
    def __getitem__(self, idx):
        sample_info = self.data.iloc[idx]
        try:
            #audio, sr = lb.load(sample_info['filepath'], sr=self.sr, mono=True)
            self.convert_to_wav(sample_info['filepath'], sample_info['out_filepath'])
        except:
            print('error_file: ', sample_info['filepath'], flush=True)
        #out_path = sample_info['out_filepath']
        sample_info = sample_info.to_dict()
        #sf.write(out_path, audio, sr, format='wav', subtype='FLOAT')
        # 音声がモノラルの場合、次元を追加して2次元配列に変換
        # if len(audio.shape) == 1:
        #     audio = audio.reshape((1, -1))

        #sf.write(out_path, audio.T, sr, format='wav', subtype='FLOAT')
        
        
        # if self.is_train == False:
        #     sample_info_tmp = []
        #     for i in range(len(features)):
        #         sample_info_ = copy.deepcopy(sample_info)
        #         sample_info_tmp.append(sample_info_)
        #     sample_info = sample_info_tmp
            
        return np.zeros((100)), sample_info