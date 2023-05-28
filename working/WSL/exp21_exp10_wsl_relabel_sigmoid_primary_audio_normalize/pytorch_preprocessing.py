# implements by https://www.kaggle.com/code/nischaydnk/birdclef-2023-pytorch-lightning-inference
import copy
import sys
import random

import numpy as np
import matplotlib.pyplot as plt
import librosa as lb
from tinytag import TinyTag
import colorednoise as cn

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

class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray, sr):
        for trns in self.transforms:
            y = trns(y, sr)
        return y


class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray, sr):
        if self.always_apply:
            return self.apply(y, sr=sr)
        else:
            if np.random.rand() < self.p:
                return self.apply(y, sr=sr)
            else:
                return y

    def apply(self, y: np.ndarray, **params):
        raise NotImplementedError

class OneOf(Compose):
    # https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/composition.py
    def __init__(self, transforms, p=0.5):
        super().__init__(transforms)
        self.p = p
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, y: np.ndarray, sr):
        data = y
        if self.transforms_ps and (random.random() < self.p):
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            t = random_state.choice(self.transforms, p=self.transforms_ps)
            data = t(y, sr)
        return data

class NoiseInjection(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_level=0.5):
        super().__init__(always_apply, p)

        self.noise_level = (0.0, max_noise_level)

    def apply(self, y: np.ndarray, **params):
        noise_level = np.random.uniform(*self.noise_level)
        noise = np.random.randn(len(y))
        augmented = (y + noise * noise_level).astype(y.dtype)
        return augmented


class GaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented


class PinkNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented

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
                 is_test=False,
                 oof_df=None,):
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
        self.oof_df = oof_df
        self.add_background_noise = AA.OneOf(
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
            ], p=0.5
        )
        
        self.audio_augment = Compose([
            OneOf([AA.Gain(min_gain_in_db=-15, max_gain_in_db=15, p=0.8),
                   AA.GainTransition(min_gain_in_db=-15, max_gain_in_db=15, p=0.8),]),
            OneOf([NoiseInjection(p=1, max_noise_level=0.04),
                   GaussianNoise(p=1, min_snr=5, max_snr=20),
                   PinkNoise(p=1, min_snr=5, max_snr=20),
                   AA.AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.03, p=0.5),
                   AA.AddGaussianSNR(min_snr_in_db=5, max_snr_in_db=15, p=0.5), ], p=0.3, ),
            self.add_background_noise,
            AA.Normalize(p=1),
            ])
        self.audio_augment_inf = Compose([AA.Normalize(p=1)])
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

    def torchaudio_read(self, filepath, offset=None):
        if offset is not None:
            audio_org, sr = torchaudio.load(filepath, frame_offset=offset, num_frames=self.audio_length)
        else:
            audio_org, sr = torchaudio.load(filepath)
        #print(filepath, 'shape:', audio_org.shape, 'offset:', offset)
        # to sr=32k and to mono
        #resampler = torchaudio.transforms.Resample(sr, self.sr)
        #audio_org = resampler(audio_org)[0, :]
        # to numpy
        audio_org = audio_org[0, :].squeeze().numpy().copy()
        return audio_org
    
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
    
    def read_file(self, sample_info):
        filepath = sample_info['filepath']
        # get random offset
        if self.is_train == True:
            offset=int(sample_info['offset']-5)
            # 音声長よりoffsetの方が長かったら、offset=0を抽出
            # TODO: offset=0にラベルも合わせる
            # _, audio_duration = self.get_audio_offset(sample_info['filepath'], duration=self.duration)
            # if audio_duration < offset:
            #     offset=0
            # print('filepath ', filepath)
            # print('offset ', offset)
            
            audio_org, sr = lb.load(filepath,
                                    offset=offset,
                                    duration=self.duration,
                                    sr=self.sr,
                                    mono=True)
            #audio_org = GaussianNoise(audio_org)
            # augment
            audio_org = self.audio_augment(audio_org, self.sr)
        else:
            if self.is_test == True:
                # for test
                audio_org, sr = lb.load(filepath, sr=self.sr, mono=True)
            else:
                # for valid
                audio_org, sr = lb.load(filepath, offset=0, duration=self.duration, sr=self.sr, mono=True)
            audio_org = self.audio_augment_inf(audio_org, self.sr)
        
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
        sample_info = self.data.iloc[idx]
        if self.is_train == True:
            sample_info_oofs = self.oof_df[self.oof_df['filepath'] == sample_info['filepath']]
            sample_info_oof = sample_info_oofs.sample(n=1, random_state=CFG.seed)
            sample_info = sample_info_oof.iloc[0]
                        
        features = self.read_file(sample_info)
        features = features.float()
        if self.is_train == True:
            onehot_target = sample_info[CFG.class_names].values.tolist()
            onehot_target = torch.Tensor(onehot_target).float()
            
        sample_info = sample_info.to_dict()
        
        if self.is_train == True:
            sample_info['onehot_target'] = onehot_target
        
        if self.is_train == False:
            sample_info_tmp = []
            for i in range(len(features)):
                sample_info_ = copy.deepcopy(sample_info)
                sec = str((i+1)*5)
                sample_info_['row_id'] = sample_info_['filename'] + f'_{sec}'
                sample_info_tmp.append(sample_info_)
            sample_info = sample_info_tmp
            
        return features, sample_info
