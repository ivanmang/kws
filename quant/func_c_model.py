import numpy as np


def Zero_Pad(X, pad):
    """
    Argument:
    X -- python numpy array of shape (n_C, n_H, n_W) representing a images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    Returns:
    X_pad -- padded image of shape (n_C, n_H + 2*pad, n_W + 2*pad)
    """
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
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
    s = np.multiply(a_slice_prev, W) 
    # Sum 
    Z = np.sum(s)
    return Z



def Conv_Forward(A_prev, W, b, stride, pad, concat = True,fo = 'none'):
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

    #b = b.reshape(n_C_next,1,1)

    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    
    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((n_C_next, n_H, n_W))
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = Zero_Pad(A_prev, pad)
              
    for c in range(n_C_next):                               
        for h in range(n_H):                       
            for w in range(n_W):                
                vert_start = h * stride
                vert_end = vert_start + f
                horiz_start = w * stride
                horiz_end = horiz_start + f

                a_slice_prev = A_prev_pad[:,vert_start:vert_end, horiz_start:horiz_end]
                
                Z[c, h, w] = Conv_Single_Step(a_slice_prev, W[c,...]) + b[c]
    
    if concat and (stride == 1)  :
        Z = np.concatenate((Z,A_prev),axis = 0)    
    
    Z = np.sign(Z)
    Z[Z == 0] = 1             
    return Z


def Linear_Forward(A_prev, W, alpha,beta,fo = 'none'):
    '''
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (n_units_prev)
    
    W -- Weights after compression and quantization, numpy array of shape (n_units_prev,n_units_next)
    
    alpha --
    beta -- alpha and beta are new parameters after compression fc layer and batch norm layer
    '''
    
    S = np.dot(A_prev, W.T)   
    Z = alpha*S + beta 
    return Z  














          
            