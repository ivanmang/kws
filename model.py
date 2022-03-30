# -*- coding: utf-8 -*-

import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import re
from manage_audio import AudioPreprocessor
import os
import torch
# model

'''
c model
'''

def quantize_tensor(x, num_bits=8):
    qmin = 0.
    qmax = 2. ** num_bits - 1.
    min_val, max_val = x.min(), x.max()
    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)
    q_x = zero_point + x / scale
    q_x = np.round(np.clip(q_x, qmin, qmax))
    return q_x

class KWS():
    def __init__(self, num_k, idx):
        self.num_k=num_k
        self.idx=idx
        self.test_txt_path = './data/test.csv'
        self.data_test_folder = './data/raw_data_12345/'
        self.model_state_path = './state/sub_part.mat'
        self.parameters = scipy.io.loadmat(self.model_state_path)
        self.structure = [
            {'layer': 'conv_0', 'in_channel': 3, 'out_channel': 64, 'kernal_size': 3, 'stride': 2, 'padding': 1, 'bias': False},
            {'layer': 'conv_1', 'in_channel': 64, 'out_channel': 64, 'kernal_size': 3, 'stride': 1, 'padding': 1, 'bias': False},
            {'layer': 'conv_2', 'in_channel': 64, 'out_channel': 64, 'kernal_size': 3, 'stride':1, 'padding': 1, 'bias': False},
            {'layer': 'conv_3', 'in_channel': 64, 'out_channel': 64, 'kernal_size': 3, 'stride':1, 'padding': 1, 'bias': False},
            {'layer': 'conv_4', 'in_channel': 64, 'out_channel': 64, 'kernal_size': 3, 'stride':2, 'padding': 1, 'bias': False},
            {'layer': 'conv_5', 'in_channel': 64, 'out_channel': 64, 'kernal_size': 3, 'stride':1, 'padding': 1, 'bias': False},
            {'layer': 'classifier', 'in_channel': 4096, 'out_channel': 7, 'kernal_size': None, 'stride':None, 'padding': None, 'bias': True}


        ]
        self.structure=pd.DataFrame(self.structure)

    def dataloader(self):
        data = pd.read_csv(self.test_txt_path, usecols=[1,2,3])
        data["wav_path"]=data["wav_path"].apply(lambda x: os.getcwd()+'/data/raw_data_12345/'+x.split('/')[-1])
        self.data = data
        return data

    def Input_decode(self, x, layer, kernel_size, padding, stride):
        # group == 1
        weight = self.parameters[layer + '_weight']
        bias = self.parameters[layer + '_bias']
        return Conv_Forward(x, weight, bias, stride=stride, pad=padding)

    #
    def Model(self, x, layer, kernel_size, padding, stride, group):
        weight = self.parameters[layer + '_weight']
        bias = self.parameters[layer + '_bias']
        y = Conv_Forward(x, weight, bias, stride=stride, pad=padding)

        return y

    #
    def Classifier(self, x, layer):
        weight = self.parameters[layer + '_conv_weight']

        alpha = self.parameters[layer + '_alpha']
        beta = self.parameters[layer + '_beta']
        return Linear_Forward(x.flatten(), weight, alpha, beta)

    def layer(self, feature, struc, fname):
        pattern=re.compile('conv_'+'[0-9]+')
        data_flow=dict()
        for s in struc['layer']:
            if pattern.findall(s):
                data_flow[s+'_input']=feature
                in_c=int(struc[struc['layer']==s]['in_channel'].values[0])
                ks=int(struc[struc['layer']==s]['kernal_size'].values[0])
                pd=int(struc[struc['layer']==s]['padding'].values[0])
                sd=int(struc[struc['layer']==s]['stride'].values[0])
                gp=1
                feature=self.Model(feature, s, ks,pd,sd,gp)
                data_flow[s + '_output'] = feature
               # np.save('./data/feature_maps/'+'e5d2e09d_nohash_0.wav_'+s+".npy", feature)
            elif 'classifier' in s:
                data_flow[s + '_input'] = feature
                feature = self.Classifier(feature, s)
                data_flow[s + '_output'] = feature
              #  print(s)
            #    np.save('./data/feature_maps/'+'e5d2e09d_nohash_0.wav_'+s+".npy", feature)
            else:
                print('Unknown layer!')
                return 'error'
        if fname!='None':
            scipy.io.savemat(fname, data_flow)
        return feature

    def forward_sigle_wav(self, feature, fo='None', fname='None'):
        if fo == 'None':
            feature = self.layer(feature, self.structure, fname)
            scores=feature[0]

        else:
            feature_flow = {}
            for i, r in self.structure.iterrows():
                feature_flow[r['layer'] + '.input'] = feature

                feature = self.layer(feature, r['layer'])
                feature_flow[r['layer'] + '.output'] = feature
            print(feature_flow)
           # scipy.io.savemat(fo, feature_flow)
        return scores.argmax()

    def forward(self):
        maps=['scilence', 'unknown', 'one', 'two', 'three', 'four', 'five']
        total = 0
        total_x = 0
        assert not self.num_k>len(self.dataloader())
        # import random
        # if len(self.idx)>0:
        #     l=self.idx
        # else:
        #     l = [random.randint(0, len(self.dataloader())) for i in range(self.num_k)]


        # for i, r in self.dataloader().iloc[l].iterrows():
        #     print('data:' + str(i) + ' is processing! Target is :', r['label'])
        kw = ['one', 'two', 'three', 'four', 'five']
        path = '/data/DeepLearning/speech_commands_v0.02/'
        for k in kw:
            print('Label---------', k)
            fi = os.listdir(path + k)[:300]
            for wav in fi:
                audio_data = librosa.core.load(path + k+'/' + wav, sr=8000)[0]
           # audio_data=librosa.core.load(r['wav_path'], sr=8000)[0]
                if len(audio_data)<8000:
                    tmp=list(audio_data)
                    tmp.extend([0]*(8000-len(audio_data)))
                    audio_data=np.array(tmp)
                audio_data = audio_data[35 * 8000 // 1000:-35 * 8000 // 1000]

                audio_processor = AudioPreprocessor(n_mels=32, n_dct_filters=32, hop_ms=30)
                mfcc, d1, d2=audio_processor.compute_mfccs(audio_data)
                mfcc, d1, d2 = mfcc.reshape(1, -1, 32), d1.reshape(1, -1, 32), d2.reshape(1, -1, 32)
                audio_tensor = torch.cat((torch.from_numpy(mfcc), torch.from_numpy(d1), torch.from_numpy(d2)), 0)
                audio_feature=np.array(np.concatenate((mfcc, d1, d2), axis=0))

                predict=self.forward_sigle_wav(audio_feature, fo='None', fname='None')
                pred_kw=maps[predict]

                print('Inference completed! Predict result is :', pred_kw)
                print()

                if pred_kw==k:
                    total_x+=1
                total += 1
        print('Total data: ' + str(total) + ' Correctly Predict:' + str(total_x))
        print('Accuracy: ' + str((total_x/ total)*100) + '%')
        print('-------------------------------------------------------')
        return predict

def Zero_Pad(X, pad):
   # print('pad--', pad)
   # print(X)
    """
    Argument:
    X -- python numpy array of shape (n_C, n_H, n_W) representing a images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    Returns:
    X_pad -- padded image of shape (n_C, n_H + 2*pad, n_W + 2*pad)
    """
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad)), 'constant')
 #  print('x-pad00',X_pad)
    return X_pad


def Conv_Single_Step(a_slice_prev, W):
    """
    Arguments:
    a_slice_prev -- slice of input data of shape (n_C_prev, f, f )
    W -- Weight parameters contained in a window - matrix of shape (n_C_prev, f, f)
    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """
    # Element-wise product .
    s = np.dot(a_slice_prev, W)
    # Sum
    Z = np.sum(s)

    return Z


def Conv_Forward(A_prev, W, b, stride, pad, concat=False, fo='none'):
    """
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (n_C_prev, n_H_prev, n_W_prev)

    W -- Weights after compression and quantization, numpy array of shape (n_C_next, n_C_prev, f, f)
    b -- Biases  after compression and quantization, numpy array of shape (n_C_next)

    stride --
    pad --

    concat -- if concat the input to output

    Z -- conv output, numpy array of shape concat = True :(n_C_next + n_C_prev, n_H, n_W)
                                           concat = False:((n_C_next, n_H, n_W)
    """
    (n_C_prev, n_H_prev, n_W_prev) = A_prev.shape
    (n_C_next, n_C_prev, f, f) = W.shape

    # b = b.reshape(n_C_next,1,1)

    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    # Initialize the output volume Z with zeros. (≈1 line)
    out = np.zeros((n_C_next, n_H, n_W))

    # Create A_prev_pad by padding A_prev
    Xin_pad = np.zeros((n_C_prev, n_H_prev + 2 * pad, n_W_prev + 2 * pad))
    Xin_pad[:, pad: pad + n_H_prev, pad: pad + n_W_prev] = np.copy(A_prev)


    for c in range(n_C_next):
        for h in range(n_H):
            for w in range(n_W):
                vert_start = h * stride
                vert_end = vert_start + f
                horiz_start = w * stride
                horiz_end = horiz_start + f

              #  a_slice_prev = A_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end]
                a_slice_prev = np.copy(np.copy(Xin_pad)[:, vert_start:vert_end, horiz_start:horiz_end])
                sum_var=0

                for ch in range(n_C_prev):
                    for k1 in range(f):
                        for k2 in range(f):
                            sum_var=sum_var+a_slice_prev[ch, k1, k2]*W[c, ch, k1, k2]
                out[c, h, w] = sum_var + b[c]

    # off_out = out.flatten()
    # print('Std---', np.std(off_out), 'Mean --', np.mean(off_out))
    # unique, counts = np.unique(off_out, return_counts=True)
    # print(sorted(dict(zip(unique, counts)).items(),key=lambda x: x[0], reverse=False))
    # print(sorted(dict(zip(unique, counts)).items(),key=lambda x: x[1], reverse=True))
    #
    # add_off = out+ 0.3*max(abs(off_out))
    #
    #
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # ax = sns.distplot(off_out, bins=400, kde=False)
    # #sns.scatterplot(x=[i for i in range(len(off_out))],y=off_out)
    #
    # plt.show()
    out = np.sign(out)
    out[out == 0] = 1
    return out


def Linear_Forward(A_prev, W, alpha, beta, fo='none'):
    '''
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (n_units_prev)

    W -- Weights after compression and quantization, numpy array of shape (n_units_prev,n_units_next)

    alpha --
    beta -- alpha and beta are new parameters after compression fc layer and batch norm layer
    '''
    # # print('A_prev.shape---', A_prev.shape)
    # # print('W.shape--',W.shape)
    # S = np.dot(A_prev, W.T)
    # Q = alpha * S + beta
    # print('Q.argmax()', Q.argmax())


    (co, ci) = W.shape # co=6, ci=4096
    out = np.zeros((co))
    for k in range(co):
        weightedsum=0
        for c in range(ci):
            weightedsum=weightedsum+A_prev[c]*W[k, c]
        out[k]=weightedsum
    out= alpha * out + beta

    return out

def main(num_k, idx=[]):
    model=KWS(num_k, idx)
    model.forward()
if __name__ == '__main__':
    main(20, [i for i in range(139)])