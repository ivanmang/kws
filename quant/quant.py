#quant
#foward 

#input quant
#bias quant 
#bn quant 
import math
import torch
import numpy as np
def linear_quantize_torch(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    delta = math.pow(2.0, -sf)
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(input / delta + 0.5)

    clipped_value = torch.clamp(rounded, min_val, max_val) * delta
    return clipped_value

    
def linear_quantize_numpy(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
#        return torch.sign(input) - 1
        return np.sign(input)
    delta = math.pow(2.0, -sf)
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    rounded = np.floor(input / delta + 0.5)

    clipped_value = np.clip(rounded, min_val, max_val) * delta
    return clipped_value


def conv_bn_sign_torch(conv_weight,bn_weight,bn_bias,bn_mean,bn_var,eps = 1e-5):    
    t1 = torch.sqrt(bn_var + eps)
    new_weight = torch.sign(bn_weight)
    weight_abs = torch.abs(bn_weight)
    new_bias = t1*bn_bias/weight_abs - bn_mean*new_weight            
    new_bias = new_bias.reshape([new_bias.shape[0],1,1])
    new_weight = new_weight.reshape([new_weight.shape[0],1,1,1]) 
    
    binary_weights = torch.sign(conv_weight)
    binary_weights[binary_weights == 0] = 1    
    new_conv_weight = binary_weights* new_weight 
           
    return [new_conv_weight,new_bias]
    
def conv_bn_sign_torch_qt(conv_weight,bn_weight,bn_bias,bn_mean,bn_var,eps = 1e-5,quant_para=[]):       
    new_conv_weight,new_bias = conv_bn_sign_torch(conv_weight,bn_weight,bn_bias,bn_mean,bn_var,eps = eps)
    new_bias = linear_quantize_torch(new_bias,quant_para[0],quant_para[1])                  
    return [new_conv_weight,new_bias]                           
 

def linear_bn_torch(linear_weight,linear_bias,bn_weight,bn_bias,bn_mean,bn_var,eps = 1e-5):             
    binary_weights = torch.sign(linear_weight)
    binary_weights[binary_weights == 0] = 1   

    t1 = torch.sqrt(bn_var +  eps)
    
    alpha = bn_weight/t1    
    beta = bn_bias - bn_weight*bn_mean/t1    
    beta = alpha*linear_bias + beta
        
    return [binary_weights,alpha,beta] 


def linear_bn_torch_qt(linear_weight,linear_bias,bn_weight,bn_bias,bn_mean,bn_var,eps = 1e-5,quant_para=[8,16,8,16]):             
    binary_weights,alpha,beta = linear_bn_torch(linear_weight,linear_bias,bn_weight,bn_bias,bn_mean,bn_var,eps = eps)
    alpha_qt = linear_quantize_torch(alpha,quant_para[0],quant_para[1])   
    beta_qt = linear_quantize_torch(beta,quant_para[2],quant_para[3])       
    return [binary_weights,alpha_qt,beta_qt] 




def conv_bn_sign_numpy(conv_weight,bn_weight,bn_bias,bn_mean,bn_var,eps = 1e-5):    
    t1 = np.sqrt(bn_var + eps)
    new_weight = np.sign(bn_weight)
    weight_abs = np.abs(bn_weight)
    new_bias = t1*bn_bias/weight_abs - bn_mean*new_weight            
    new_bias = new_bias.reshape([new_bias.shape[0],1,1])
    new_weight = new_weight.reshape([new_weight.shape[0],1,1,1]) 
    
    binary_weights = np.sign(conv_weight)
    binary_weights[binary_weights == 0] = 1    
    new_conv_weight = binary_weights* new_weight 
           
    return [new_conv_weight,new_bias]
    
def conv_bn_sign_numpy_qt(conv_weight,bn_weight,bn_bias,bn_mean,bn_var,eps = 1e-5,quant_para=[]):       
    new_conv_weight,new_bias = conv_bn_sign_numpy(conv_weight,bn_weight,bn_bias,bn_mean,bn_var,eps = eps)
    new_bias = linear_quantize_numpy(new_bias,quant_para[0],quant_para[1])
    return [new_conv_weight,new_bias]
 

def linear_bn_numpy(linear_weight,linear_bias,bn_weight,bn_bias,bn_mean,bn_var,eps = 1e-5):             
    binary_weights = np.sign(linear_weight)
    binary_weights[binary_weights == 0] = 1   

    t1 = np.sqrt(bn_var + eps)
    
    alpha = bn_weight/t1    
    beta = bn_bias - bn_weight*bn_mean/t1    
    beta = alpha*linear_bias + beta
        
    return [binary_weights,alpha,beta] 


def linear_bn_numpy_qt(linear_weight,linear_bias,bn_weight,bn_bias,bn_mean,bn_var,eps = 1e-5,quant_para=[]):
    binary_weights,alpha,beta = linear_bn_numpy(linear_weight,linear_bias,bn_weight,bn_bias,bn_mean,bn_var,eps = eps)


    alpha_qt = linear_quantize_numpy(alpha,quant_para[0],quant_para[1])   
    beta_qt = linear_quantize_numpy(beta,quant_para[2],quant_para[3])
    print('alpha---', alpha)
    print('alpha_qt---', alpha_qt)
    print('beta -----', beta)
    print('beta_qt---', beta_qt)
    print(max(beta_qt), min(beta_qt))
    return [binary_weights,alpha_qt,beta_qt]

