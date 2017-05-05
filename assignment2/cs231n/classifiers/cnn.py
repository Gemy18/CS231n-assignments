import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32,use_batch=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.num_filters=num_filters
    self.filter_size=filter_size
    self.N=input_dim
    self.C=num_classes
    self.use_batch=use_batch
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################

    C, H, W = input_dim
    F = num_filters

    F_H = filter_size
    F_W = filter_size
    S = 1
    P = (filter_size - 1) / 2
    Hc = 1 + (H + 2 * P - F_H) / S
    Wc = 1 + (W + 2 * P - F_W) / S

    self.params['W1'] = weight_scale * np.random.randn(F,C,F_H,F_W)
    self.params['b1'] = weight_scale * np.random.randn(F)


    pool_w = 2
    pool_h = 2
    S = 2
    Hp = (Hc - pool_h) / S + 1
    Wp = (Wc - pool_w) / S + 1



    self.params['W2'] = weight_scale * np.random.randn(F * Hp * Wp, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)


    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    temp=[num_filters,hidden_dim]
    if use_batch:
      for i in range(2):
        self.params['beta' + str(i+1)] = np.zeros(temp[i])
        self.params['gamma' + str(i+1)] = np.ones(temp[i])

    self.bn_params = []
    if self.use_batch:
      self.bn_params = [{'mode': 'train'} for i in xrange(2)]


    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    if self.use_batch:
      out1, cache2=spatial_batchnorm_forward(out1,self.params['gamma1'],self.params['beta1'],   self.bn_params[0])
    if self.use_batch:
      out3, cache3=affine_batch_rule_forward(out1, W2, b2,self.params['gamma2'],self.params['beta2'],   self.bn_params[1])
    else :
      out3, cache3 = affine_relu_forward(out1, W2, b2)

    scores, cache5 = affine_forward(out3, W3, b3)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))
    loss = data_loss + reg_loss

    dout3, grads['W3'], grads['b3'] = affine_backward(dscores, cache5)
    if self.use_batch:
      dout2, grads['W2'], grads['b2'], grads['gamma2'],    grads['beta2']= affine_norm_relu_backward(dout3,cache3)
    else :
      dout2, grads['W2'], grads['b2'] = affine_relu_backward(dout3, cache3)
    if self.use_batch:
      dout2,grads['gamma1'],    grads['beta1'] = spatial_batchnorm_backward(dout2,cache2)

    dout1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout2, cache1)

    grads['W1'] += self.reg * W1
    grads['W2'] += self.reg * W2
    grads['W3'] += self.reg * W3

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

  
pass
