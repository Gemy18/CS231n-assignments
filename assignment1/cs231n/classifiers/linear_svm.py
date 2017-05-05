import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero


  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = W.dot(X[i,:])
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

        # Compute gradients (one inner and one outer sum)
        # Wonderfully compact and hard to read
        dW[y[i],:] -= X[i].T # this is really a sum over j != y_i
        dW[j,:] += X[i].T # sums each contribution of the x_i's

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Same with gradient
  dW /= num_train

  # Add regularization
  loss += 0.5 * reg * np.sum(W * W)

  # Gradient regularization that carries through per https://piazza.com/class/i37qi08h43qfv?cid=118
  dW += reg*W

  return loss, dW
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################




def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train=X.shape[0]
  num_classes=W.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################


  scores=np.dot(X,W.T)
  cscore=scores[np.arange(num_train), y]


  mat = scores - np.vstack(cscore) + 1  # like above, delta = 1
  mat[ np.arange(num_train),y] = 0  # accounting for the j=y_i term we shouldn't count (subtracting 1 makes up for it since w_j = w_{y_j} in this case)
  # Compute max
  thresh = np.maximum(np.zeros((num_train, num_classes)), mat)

  # Compute loss as double sum
  loss = np.sum(thresh)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
###  thresh=thresh > 0
  # ok=np.sum(thresh,axis=1)
  #thresh[np.arange(num_train),y]=-1*ok[range(num_train)]
  #dW = np.dot(thresh.T, X)
  #dW /= num_train
  #dW += reg*W
###
  thresh = thresh > 0
  thresh= thresh*1
  ok=np.sum(thresh,axis=1)
  thresh[ range(num_train),y] = -ok[range(num_train)]
  dW = np.dot(thresh.T, X)


  dW /= num_train

  dW += reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
