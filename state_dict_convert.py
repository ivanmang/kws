# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:01:46 2020

@author: yuanlei
"""

import torch
import scipy.io as scio
import sys
import os

from quant.quant import conv_bn_sign_numpy_qt,linear_bn_numpy_qt

def convert_state_dict():
    quant_para = {}
    quant_para['conv_qt'] = [0,32]
    quant_para['linear_qt'] = [9, 12, 9, 12]  # [8,32,8,32]
    
    
    conv_qt = quant_para['conv_qt']
    linear_qt = quant_para['linear_qt']
    eps = 1e-5
    
    
    model_folder = '/nfs/project/AIOT/work/lisha/qws-bcnn-32x32_demo_3channels/model/3chns_chiplet_0126/modelepoch-185.pt'
    checkpoint = torch.load(model_folder, map_location='cpu')
#    state_dict = checkpoint.state_dict()
    state_dict = checkpoint

    state_dict_keys = list(state_dict.keys())
    state_dict_values = list(state_dict.values())
    classifier_keys = [t for t in state_dict_keys if 'classifier' in t]
    
    
    layer = int( ( len(state_dict_keys) - len(classifier_keys) )/ 6) # 
    layer_num = 0
    
    convert_list = {}
    
    for i in range(layer):
        conv_weight = state_dict_values[layer_num].numpy()
        bn_weight = state_dict_values[layer_num+1].numpy()
        bn_bias = state_dict_values[layer_num+2].numpy()
        bn_mean = state_dict_values[layer_num+3].numpy()
        bn_var = state_dict_values[layer_num+4].numpy() 
        
        new_conv_weight,new_bias = conv_bn_sign_numpy_qt(conv_weight,
                                                         bn_weight,
                                                         bn_bias,
                                                         bn_mean,
                                                         bn_var,
                                                         quant_para = conv_qt,
                                                         eps = eps)    

        convert_list['conv_' + str(i) + '_weight'] = new_conv_weight
        convert_list['conv_' + str(i) + '_bias'] = new_bias      
            
        layer_num= layer_num + 6

    #load classifier
    conv_weight = state_dict_values[layer_num].numpy()
    conv_bias = state_dict_values[layer_num+1].numpy()
    bn_weight = state_dict_values[layer_num+2].numpy()
    bn_bias = state_dict_values[layer_num+3].numpy()
    bn_mean = state_dict_values[layer_num+4].numpy()
    bn_var = state_dict_values[layer_num+5].numpy() 
    
 
    binary_weights,alpha,beta = linear_bn_numpy_qt(conv_weight,
                                                   conv_bias,     
                                                   bn_weight,
                                                   bn_bias,
                                                   bn_mean,
                                                   bn_var,
                                                   quant_para = linear_qt,
                                                   eps = eps) 

    convert_list['classifier_conv_weight'] = binary_weights
    convert_list['classifier_alpha'] = alpha
    convert_list['classifier_beta'] = beta


    path = os.path.join(os.getcwd()+'/state/', 'bnn_3channels.mat')
    
    scio.savemat(path, convert_list)
    
#    data = scio.loadmat(path)  

    
convert_state_dict()
