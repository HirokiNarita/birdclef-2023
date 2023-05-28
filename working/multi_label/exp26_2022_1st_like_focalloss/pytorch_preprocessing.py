# implements by https://www.kaggle.com/code/nischaydnk/birdclef-2023-pytorch-lightning-inference
import copy
import sys
import ast

import numpy as np
import matplotlib.pyplot as plt
import librosa as lb
from tinytag import TinyTag

import torch
import torchaudio
from torch.utils.data import Dataset

import audiomentations as AA

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
        return y.astype(np.float32)

    return y.astype(np.float32)

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
        # for train / val
        self.is_train = is_train
        # for test
        self.is_test = is_test
        
        self.data = data
        
        self.sr = sr
        
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr//2
        self.hop_length = hop_length

        if self.is_test == True:
            self.duration = CFG.test_duration
        else:
            self.duration = duration
        self.audio_length = self.duration*self.sr
        self.step = step or self.audio_length
        if self.is_train == True:
            self.audio_augment = AA.Compose(
                [
                    AA.AddBackgroundNoise(
                        sounds_path=f"{CFG.BACKNOISE_BASE_PATH}/ff1010bird_nocall/nocall", min_snr_in_db=0, max_snr_in_db=3, p=0.5
                    ),
                    AA.AddBackgroundNoise(
                        sounds_path=f"{CFG.BACKNOISE_BASE_PATH}/train_soundscapes/nocall", min_snr_in_db=0, max_snr_in_db=3, p=0.25
                    ),
                    AA.AddBackgroundNoise(
                        sounds_path=f"{CFG.BACKNOISE_BASE_PATH}/aicrowd2020_noise_30sec/noise_30sec", min_snr_in_db=0, max_snr_in_db=3, p=0.25
                    ),
                    AA.Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.5), # DEFAULT
                    AA.Shift(min_fraction=0.2, max_fraction=0.2, p=0.5),   # 15 + (15 * 0.2) = 18 input audio length 
                    AA.LowPassFilter(min_cutoff_freq=100, max_cutoff_freq=10000, p=0.5), # possibly incorrect values
                ]
            )

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
        return start, audio_length
    
    def read_file(self, filepath):
        # get random offset
        offset, audio_duration = self.get_audio_offset(filepath, duration=self.duration)
        if self.is_train == True:
            audio_org, sr = lb.load(filepath, offset=offset, duration=self.duration, sr=self.sr, mono=True)
            #audio_org = GaussianNoise(audio_org)
        else:
            if self.is_test == True:
                # for test
                audio_org, sr = lb.load(filepath, sr=CFG.test_duration, mono=True)
            else:
                # for valid
                audio_org, sr = lb.load(filepath, offset=0, duration=CFG.test_duration, sr=self.sr, mono=True)
        # train
        if self.is_test is not True :
            # adjust audio_org length
            audio = crop_or_pad(audio_org, length=self.audio_length, is_train=self.is_train)
            if self.is_train == True:
                audio = self.audio_augment(audio, sample_rate=self.sr)
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
        
        return audio, audio_duration
    
    def make_multi_onehot(self, sample_info, audio_duration):
        labels = torch.zeros((CFG.num_classes,))
        primary_label = [sample_info['primary_label']]
        secondary_label = ast.literal_eval(sample_info['secondary_labels'])
        # inputのdurationより短い場合は、primaryもsecondaryも確実に入っているので1
        # elseの場合は使わない
        primary_label_prob, secondary_label_prob = 0, 0
        multi_label_names = primary_label + secondary_label
        if self.duration > audio_duration:
            # durationより短い場合、確実に拾えているため、1.0
            primary_label_prob, secondary_label_prob = 1.0, 1.0
        else:
            # durationより長い場合、拾えていない可能性がある
            # primaryは全体に占める割合が多いと考えられるため、0.9995
            # secondaryは入っていない可能性も高いため、0.5000
            # https://www.kaggle.com/competitions/birdclef-2022/discussion/327193
            #primary_label_prob, secondary_label_prob = 0.9995, 0.5000
            primary_label_prob, secondary_label_prob = 1.0, 0
        
        for idx, multi_label_name in enumerate(multi_label_names):
            if idx == 0:
                labels[CFG.name2label[multi_label_name]] = primary_label_prob
            else:
                labels[CFG.name2label[multi_label_name]] = secondary_label_prob
        
        return labels
    
    def __getitem__(self, idx):
        sample_info = self.data.loc[idx]
        features, audio_duration = self.read_file(sample_info['filepath'])
        features = features.float()
        sample_info = sample_info.to_dict()
        if self.is_test == False:
            sample_info['multi_label_target'] = self.make_multi_onehot(sample_info, audio_duration=audio_duration)

        if self.is_test is True:
            sample_info_tmp = []
            for i in range(len(features)):
                sample_info_ = copy.deepcopy(sample_info)
                sec = str((i+1)*5)
                sample_info_['row_id'] = sample_info_['filename'] + f'_{sec}'
                sample_info_tmp.append(sample_info_)
            sample_info = sample_info_tmp
            
        return features, sample_info