import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import re
from manage_audio_jiao_3channe_vad_based import AudioPreprocessor
import os
import torch
# model

class BNN():
    def __init__(self, num_k, idx):
        self.num_k=num_k
        self.idx=idx
        self.test_txt_path = './data/test.csv'
        self.data_flow_folder = './data/data_flow'
        self.data_test_folder = './data/data_2/'
        self.model_state_path = './state/bnn_3channels.mat'
        self.model_stucture_path='./DenseNet_2x4.csv'
        self.parameters = scipy.io.loadmat(self.model_state_path)

        self.structure = [
            {'layer': 'conv_0', 'in_channel': 3, 'out_channel': 64, 'kernal_size': 3, 'stride': 2, 'padding': 1,
             'bias': False},
            {'layer': 'conv_1', 'in_channel': 64, 'out_channel': 64, 'kernal_size': 3, 'stride': 1, 'padding': 1,
             'bias': False},
            {'layer': 'conv_2', 'in_channel': 64, 'out_channel': 64, 'kernal_size': 3, 'stride': 1, 'padding': 1,
             'bias': False},
            {'layer': 'conv_3', 'in_channel': 64, 'out_channel': 64, 'kernal_size': 3, 'stride': 1, 'padding': 1,
             'bias': False},
            {'layer': 'conv_4', 'in_channel': 64, 'out_channel': 64, 'kernal_size': 3, 'stride': 2, 'padding': 1,
             'bias': False},
            {'layer': 'conv_5', 'in_channel': 64, 'out_channel': 64, 'kernal_size': 3, 'stride': 1, 'padding': 1,
             'bias': False},
            {'layer': 'classifier', 'in_channel': 4096, 'out_channel': 7, 'kernal_size': None, 'stride': None,
             'padding': None, 'bias': True}

        ]
        self.structure = pd.DataFrame(self.structure)



    def Input_decode(self, x, layer, kernel_size, padding, stride):
        # group == 1
        weight = self.parameters[layer + '_weight']
        bias = self.parameters[layer + '_bias']
        return Conv_Forward(x, weight, bias, stride=stride, pad=padding)

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

    def layer(self, feature, r_layer):
        pattern=re.compile('conv_'+'[0-9]+')
        if pattern.findall(r_layer):

            in_c=int(self.structure[self.structure['layer']==r_layer]['in_channel'].values[0])
            ks=int(self.structure[self.structure['layer']==r_layer]['kernal_size'].values[0])
            pd=int(self.structure[self.structure['layer']==r_layer]['padding'].values[0])
            sd=int(self.structure[self.structure['layer']==r_layer]['stride'].values[0])
            gp=1
            feature=self.Model(feature, r_layer, ks,pd,sd,gp)

        elif 'classifier' in r_layer:

            feature = self.Classifier(feature, r_layer)

        else:
            print('Unknown layer!')
            return 'error'
        return feature

    def forward_sigle_wav(self, feature, fo='None'):
        if fo == 'None':
            feature = self.layer(feature, self.structure)
            scores=feature[0]

        else:
            feature_flow = {}
            for i, r in self.structure.iterrows():
                feature_flow[r['layer'] + '.input'] = feature

                feature = self.layer(feature, r['layer'])
                feature_flow[r['layer'] + '.output'] = feature
            scores = feature[0]
            scipy.io.savemat(fo, feature_flow)
        return scores.argmax()

    def vad_kws_forward(self, testdata_path, doc):
        maps = ['scilence', 'unknown', 'one', 'two', 'three', 'four', 'five']


        total = 0
        total_x = 0

        test = pd.read_csv(testdata_path)

        for data in range(test.shape[0]):
            audio_data = test.iloc[data].values
            audio_processor = AudioPreprocessor(n_mels=32, n_dct_filters=32, hop_ms=30)
            mfcc, d1, d2 = audio_processor.compute_mfccs(audio_data)
            mfcc, d1, d2 = mfcc.reshape(1, -1, 32), d1.reshape(1, -1, 32), d2.reshape(1, -1, 32)
            audio_tensor = torch.cat((torch.from_numpy(mfcc), torch.from_numpy(d1), torch.from_numpy(d2)),
                                     0)  # shape (3, 32, 32)
            audio_feature = np.array(np.concatenate((mfcc, d1, d2), axis=0))  # shape (3, 32, 32)
            #  predict=self.forward_sigle_wav(audio_feature, fo=os.path.join(self.data_flow_folder, wav.replace('wav', 'mat')))
            predict = self.forward_sigle_wav(audio_feature, fo=os.path.join(self.data_flow_folder, doc+'_'+str(data)+'.mat'))
            pred_kw = maps[predict]
            #print('Testing wav: ', wav, 'Label: ', label)
            print('Inference completed! Predict result is :', pred_kw)
            print()


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
    if concat and (stride == 1):
        out = np.concatenate((out, A_prev), axis=0)

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

def main(num_k, idx, testdata_path, doc):
    # num_k: sekect n samples randomly
    # idx: extract the sample as index = idx in the test sample list
    # set either num_k or idx as None
    model= BNN(num_k, idx)
    model.vad_kws_forward(testdata_path, doc)



if __name__ == '__main__':
    testdata_path = './data/testdata/Testdata1_single1_pad_8000.csv'
    doc=testdata_path.rstrip('.csv').split('/')[-1]
    print(doc)
    main(None, None, testdata_path, doc)




