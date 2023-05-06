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

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()

def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear").squeeze(1)

    return output



class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)

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

        self.sr=sample_rate
        self.n_fft=n_fft
        self.hop_length=hop_length
        self.n_mels=n_mels
        self.num_classes=num_classes
        
        self.to_melspec_fn = MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        self.to_log_scale_fn = AmplitudeToDB(top_db=top_db)
        
        self.torchvision_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5)
            ])
        self.bn0 = nn.BatchNorm2d(CFG.n_mels)
        self.model=timm.create_model(pretrained=pretrained,
                                     model_name=model_name,
                                     in_chans=in_chans,
                                     num_classes=num_classes,)
        self.fc1 = nn.Linear(1280, 1280, bias=True)
        self.att_block = AttBlockV2(
            1280, num_classes, activation="sigmoid")
               
        self.freq_dropper = DropStripes(dim=2, drop_width=int(CFG.freq_mask/2), 
            stripes_num=2)
        self.time_dropper = DropStripes(dim=3, drop_width=int(CFG.time_mask/2), 
            stripes_num=2)
    
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
        frames_num = x.shape[3]
        # plt.imshow(x[0,0,:,:].to('cpu'))
        # plt.show()
        # log-mel augment
        if self.training == True:
            x = self.torchvision_transforms(x)
            # TODO mixup
            x, y = self.mixup(x, targets)
            # TODO specaug
            x = self.time_dropper(self.freq_dropper(x))
        
        # plt.imshow(x[0,0,:,:].to('cpu'))
        # plt.show()

        x = x.transpose(1, 2)
        x = self.bn0(x)
        x = x.transpose(1, 2)
                
        x = self.model.forward_features(x)
        # Aggregate in frequency axis
        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            'framewise_output': framewise_output,
            'clipwise_output': clipwise_output,
            'logit': logit,
            'framewise_logit': framewise_logit,
        }
        if self.training == True:
            return output_dict, y
        else:
            return output_dict
