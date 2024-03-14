# VGG imp

import os
from typing import Literal, Union, Sequence

import requests
import h5py
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .params import VGG19_WEIGHTS_LINK

class VGG19(nn.Module):
    def __init__(
        self,
        input_size:Union[int, Sequence[int]]=224,
        activation:str='softmax',
        is_classifier:bool=True,
        use_weight:bool=True,
        dropout:float=0.5) -> None:
        super().__init__()
        self.activation = activation
        self.is_classifier = is_classifier
        if isinstance(input_size, int):
            w = input_size
            h = input_size
        else:
            w, h = input_size
        
        # conv_block_1
        self.block1_conv1 = nn.Conv2d(  3,  64, kernel_size=3, padding=1) #  1
        self.block1_conv2 = nn.Conv2d( 64,  64, kernel_size=3, padding=1) #  2
        self.block1_pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # conv_block_2
        self.block2_conv1 = nn.Conv2d( 64, 128, kernel_size=3, padding=1) #  3
        self.block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1) #  4
        self.block2_pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # conv_block_3
        self.block3_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1) #  5
        self.block3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1) #  6
        self.block3_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1) #  7
        self.block3_conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1) #  8
        self.block3_pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # conv_block_4
        self.block4_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1) #  9
        self.block4_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # 10
        self.block4_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # 11
        self.block4_conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # 12
        self.block4_pool   = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # conv_block_5
        self.block5_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # 13
        self.block5_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # 14
        self.block5_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # 15
        self.block5_conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # 16
        self.block5_pool   = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if is_classifier:
            fc_dim = 512 * (w // 32) * (h // 32)
            
            # fc_block
            self.fc1 = nn.Linear(fc_dim, 4096)                            # 17
            self.fc2 = nn.Linear(  4096, 4096)                            # 18
            self.predictions = nn.Linear(4096, 1000)                      # 19
            self.dropout = nn.Dropout(dropout)
    
    def __call__(
        self,
        x:mx.array,
        train:bool=True,
        out_type:Literal['mx', 'np']='mx') -> mx.array:
        x = mx.array(x)
        
        x = nn.relu(self.block1_conv1(x))
        x = nn.relu(self.block1_conv2(x))
        x = self.block1_pool(x)
        
        x = nn.relu(self.block2_conv1(x))
        x = nn.relu(self.block2_conv2(x))
        x = self.block2_pool(x)
        
        x = nn.relu(self.block3_conv1(x))
        x = nn.relu(self.block3_conv2(x))
        x = nn.relu(self.block3_conv3(x))
        x = nn.relu(self.block3_conv4(x))
        x = self.block3_pool(x)
        
        x = nn.relu(self.block4_conv1(x))
        x = nn.relu(self.block4_conv2(x))
        x = nn.relu(self.block4_conv3(x))
        x = nn.relu(self.block4_conv4(x))
        x = self.block4_pool(x)
        
        x = nn.relu(self.block5_conv1(x))
        x = nn.relu(self.block5_conv2(x))
        x = nn.relu(self.block5_conv3(x))
        x = nn.relu(self.block5_conv4(x))
        x = self.block5_pool(x)
        
        if self.is_classifier:
            x = x.reshape(x.shape[0], -1)
            x = nn.relu(self.fc1(x))
            if train: x = self.dropout(x)
            x = nn.relu(self.fc2(x))
            if train: x = self.dropout(x)
            x = self.predictions(x)
            x = getattr(nn, self.activation)(x)
            
        if out_type == 'mx':
            return x
        elif out_type == 'np':
            return np.array(x)
        else:
            raise ValueError(f'Invalid out_type: {out_type}')

    def load_weights(self, weights_path:str) -> None:
        if weights_path is None:
            weights_path = 'resources/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
            if not os.path.exists(weights_path):
                os.makedirs('resources', exist_ok=True)
                r = requests.get(VGG19_WEIGHTS_LINK, allow_redirects=True)
                open(weights_path, 'wb').write(r.content)
                
        weight = h5py.File(weights_path, 'r')
        for key in weight.keys():
            params = weight[key]
            params_keys = list(params.keys())
            try:
                layer = getattr(self, key)
                if len(layer.weight.shape) == 4:
                    layer.weight = mx.array(params[params_keys[0]][...]).transpose((3, 0, 1, 2))
                elif len(layer.weight.shape) == 2:
                    layer.weight = mx.array(params[params_keys[0]][...]).transpose((1, 0))
                layer.bias = mx.array(params[params_keys[1]][...])
            except:
                pass
        weight.close()
        return self
