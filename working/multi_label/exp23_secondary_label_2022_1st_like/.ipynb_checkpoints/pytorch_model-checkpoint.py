import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.distributions import Beta

import timm
import torchaudio as ta
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import AmplitudeToDB
#from torchlibrosa import DropStripes
from SpecAugment import DropStripes

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

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (
                self.__class__.__name__
                + "("
                + "p="
                + "{:.4f}".format(self.p.data.tolist()[0])
                + ", "
                + "eps="
                + str(self.eps)
                + ")"
        )

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
        
        # self.torchvision_transforms = T.Compose([
        #     T.RandomHorizontalFlip(p=0.5)
        #     ])
        self.bn0 = nn.BatchNorm2d(CFG.n_mels)
        self.model=timm.create_model(pretrained=pretrained,
                                     model_name=model_name,
                                     in_chans=in_chans,
                                     num_classes=num_classes,)
        self.fc1 = nn.Linear(1280, 1280, bias=True)
        # self.att_block = AttBlockV2(
        #     1280, num_classes, activation="sigmoid")
        self.global_pool = GeM(p=3, eps=1e-6)
        self.head = nn.Linear(1280, num_classes)
        # self.freq_dropper = DropStripes(dim=2, drop_width=int(CFG.freq_mask/2), 
        #     stripes_num=2)
        # self.time_dropper = DropStripes(dim=3, drop_width=int(CFG.time_mask/2), 
        #     stripes_num=2)
        self.freq_mask = ta.transforms.FrequencyMasking(24, iid_masks=True)
        self.time_mask = ta.transforms.TimeMasking(64, iid_masks=True)
    
    def mixup(self, data, label, alpha=0.5, prob_th=CFG.mixup_prob, device='cuda:0'):
        def _aug(data1, data2, weights, probs):
            batch_size = len(data1)
            mix_data = []
            for i in range(batch_size):
                if prob_th >= probs[i]:
                    sample = data1[i]*weights[i] + data2[i]*(1 - weights[i])
                else:
                    sample = data1[i]
                mix_data.append(sample)
            
            mix_data = torch.stack(mix_data)

            return mix_data
        
        batch_size = len(data)
        weights = Beta(1., 1.).rsample(torch.Size((batch_size,))).to(device)
        index = np.random.permutation(batch_size)
        x1, x2 = data, data[index]
        y1, y2 = label, label[index]

        probs = np.random.rand(data.shape[0])
        x = _aug(x1, x2, weights, probs)
        y = _aug(y1, y2, weights, probs)

        return x, y
       
    def forward(self, x, targets=None):
        # print(x.shape)
        # plt.imshow(x[0,0,:,:].to('cpu'), aspect='auto')
        # plt.show()
        # log-mel augment
        if self.training == True:
            #x = self.torchvision_transforms(x)
            #x, y = self.mixup(x, targets)
            # spec aug
            x = self.freq_mask(x)
            x = self.time_mask(x)
        
        # plt.imshow(x[0,0,:,:].to('cpu'))
        # plt.show()

        x = self.model.forward_features(x)
        # Aggregate in frequency axis
        x = self.global_pool(x)

        x = x[:, :, 0, 0]
        logit = self.head(x)

        output_dict = {
            'logit': logit,
        }
        if self.training == True:
            return output_dict, y
        else:
            return output_dict