from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    P = np.zeros_like(X@W)
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        P[i]+=p
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss


    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    a = np.arange(num_train)
    P[a,y] -= 1
    dloss_W = 1 / num_train * X.T@P + 2*reg*W
    dloss_X = 1 / num_train * P@W.T
    dW = dloss_W


    return loss, dW

def softmax(z):
    e = np.exp(z)
    nor = np.sum(e,axis=1,keepdims=True)
    return e/nor


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    num_train=X.shape[0]
    loss = 0.0
    dW = np.zeros_like(W)
    z = X@W
    z -= np.max(z,axis=1)[:,None]
    #p = softmax(z)
    p = np.exp(z)
    p /= p.sum(axis=1,keepdims=True)
    P = np.zeros_like(p)
    a = np.arange(num_train)
    P[a,y] -= 1
    P += p
    logp = np.log(p)
    #print("Num train", num_train)
    #print("Shape of y", y.shape)
    loss -= logp[a,y]
    #print(loss.shape)
    #loss = loss.T@
    loss = loss.sum() / num_train + reg * np.sum(W * W)
    #print(loss.shape)
    


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    dW = 1 / num_train * X.T@P + 2*reg*W


    return loss, dW
