# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This module implements the Normalization of gradients as a PreprocessorPyTorch subclass. This is used by the privacy attacker
before passing through the attack model for gender inference.
Including this as a preprocessor class will enable generating perturbations to include this normalization step in addition to the attack model
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple

import numpy as np
from tqdm.auto import tqdm
import pdb
import torchaudio
import pickle
import os
import sys
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
import torch
from pathlib import Path

leak_layer_dict = {'full': ['w0', 'b0', 'w1', 'b1', 'w2', 'b2'],
                   'first': ['w0', 'b0'], 'second': ['w1', 'b1'], 'last': ['w2', 'b2']}
leak_layer_idx_dict = {'w0': 0, 'w1': 2, 'w2': 4, 'b0': 1, 'b1': 3, 'b2': 5}
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

class Normalize_gradients(PreprocessorPyTorch):
    
    """
    Implements the normalization a defense.
    """

    def __init__(self, 
                save_dir, 
                leak_layer, 
                model_type, 
                pred, 
                feature_type, 
                adv_dataset, 
                model_setting_str, 
                num_epochs, 
                apply_fit: bool = False, 
                apply_predict: bool = True) -> None:
        print("Initializing gradient normalizer")
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        # 1. normalization tmp computations
        weight_norm_mean_dict, weight_norm_std_dict = {}, {}
        weight_sum, weight_sum_square = {}, {}
        for key in ['w0', 'w1', 'w2', 'b0', 'b1', 'b2']:
            weight_norm_mean_dict[key], weight_norm_std_dict[key] = 0, 0
            weight_sum[key], weight_sum_square[key] = 0, 0
        # the updates layer name and their idx in gradient file
        weight_name, bias_name = leak_layer_dict[leak_layer][0], leak_layer_dict[leak_layer][1]
        weight_idx, bias_idx = leak_layer_idx_dict[weight_name], leak_layer_idx_dict[bias_name]

        # 1.1 read all data and compute the tmp variables
        # Used for normalizing the inputs. Computes variables using shadow model parameters.
        shadow_training_sample_size, shadow_data_dict = 0, {}
        print('reading file %s' % str(Path(save_dir).joinpath('tmp_model_params', model_type, pred, feature_type, adv_dataset, model_setting_str)))
        for shadow_idx in range(5):
            print('reading shadow model %d' % (shadow_idx))
            for epoch in range(int(num_epochs)):
                adv_federated_model_result_path = Path(save_dir).joinpath('tmp_model_params', model_type, pred, feature_type, adv_dataset, model_setting_str, 'fold'+str(int(shadow_idx+1)))
                file_str = str(adv_federated_model_result_path.joinpath('gradient_hist_'+str(epoch)+'.pkl'))
                # if shadow_idx == 0 and epoch < 10:
                with open(file_str, 'rb') as f:
                    adv_gradient_dict = pickle.load(f)
                for speaker_id in adv_gradient_dict:
                    data_key = str(shadow_idx)+'_'+str(epoch)+'_'+speaker_id
                    gradients = adv_gradient_dict[speaker_id]['gradient']
                    shadow_training_sample_size += 1

                    # calculate running stats for computing std and mean
                    shadow_data_dict[data_key] = {}
                    shadow_data_dict[data_key]['gender'] = adv_gradient_dict[speaker_id]['gender']
                    shadow_data_dict[data_key][weight_name] = gradients[weight_idx]
                    shadow_data_dict[data_key][bias_name] = gradients[bias_idx]
                    for layer_name in leak_layer_dict[leak_layer]:
                        weight_sum[layer_name] += gradients[leak_layer_idx_dict[layer_name]]
                        weight_sum_square[layer_name] += gradients[leak_layer_idx_dict[layer_name]]**2

        # 1.2 calculate std and mean
        for key in leak_layer_dict[leak_layer]:
            weight_norm_mean_dict[key] = weight_sum[key] / shadow_training_sample_size
            tmp_data = weight_sum_square[key] / shadow_training_sample_size - (weight_sum[key] / shadow_training_sample_size)**2
            weight_norm_std_dict[key] = np.sqrt(tmp_data)
        
        self.weight_norm_mean_dict = weight_norm_std_dict
        self.weight_norm_std_dict = weight_norm_std_dict
        self.leak_layer = leak_layer
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.device_gpu = device

    def forward(self, weights_bias, y):
        weights = weights_bias[:,:,:-1]
        bias = weights_bias[:,:,-1]
        tmp_data = (weights - torch.tensor(self.weight_norm_mean_dict[self.weight_name]).to(self.device_gpu)) / torch.tensor(self.weight_norm_std_dict[self.weight_name] + 0.00001).to(self.device_gpu)
        weights = tmp_data.to(self.device_gpu)
        #weights = torch.from_numpy(np.ascontiguousarray(tmp_data)).to(self.device_gpu)
        tmp_data = (bias - torch.tensor(self.weight_norm_mean_dict[self.bias_name]).to(self.device_gpu)) / torch.tensor(self.weight_norm_std_dict[self.bias_name] + 0.00001).to(self.device_gpu)
        bias = tmp_data.to(self.device_gpu)
        #bias = torch.from_numpy(np.ascontiguousarray(tmp_data)).to(self.device_gpu)

        weights_bias = torch.cat((weights,bias.unsqueeze(dim=2)),2)
        return weights_bias, y
