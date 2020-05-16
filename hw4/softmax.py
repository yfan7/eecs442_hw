import pickle
import numpy as np
from layers import *

class SoftmaxClassifier(object):
    """
    A fully-connected neural network with softmax loss that uses a modular
    layer design. We assume an input dimension of D, a hidden dimension
    of H, and perform classification over C classes.

    The architecture should be fc - softmax if no hidden layer.
    The architecture should be fc - relu - fc - softmax if one hidden layer

    Note that this class does not implement gradient descent.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3072, hidden_dim=None, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer, None
          if there's no hidden layer.
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with fc weights and biases using the keys        #
        # 'W' and 'b', i.e., W1, b1 for the weights and bias in the first linear   #
        # layer, W2, b2 for the weights and bias in the second linear layer.       #
        ############################################################################
        self.h = True
        if hidden_dim is not None:
            self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim))
            self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
            self.params['b1'] = np.zeros(hidden_dim)
            self.params['b2'] = np.zeros(num_classes)
        else:
            self.h = False
            self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, num_classes))
            self.params['b1'] = np.zeros(num_classes)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def forwards_backwards(self, X, y=None, return_dx = False):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, Din)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass. And
        if return_dx if True, return the gradients of the loss with respect to 
        the input image, otherwise, return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the one-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        c1 = None
        c2 = None
        cr = None
        W1 = None
        W2 = None
        if self.h:
            W1, b1 = self.params['W1'], self.params['b1']
            W2, b2 = self.params['W2'], self.params['b2']
            z, c1 = fc_forward(X, W1, b1)
            h, cr = relu_forward(z)  
            scores, c2 = fc_forward(h , W2, b2)
        else:
            W1, b1 = self.params['W1'], self.params['b1']
            scores, c1 = fc_forward(X, W1, b1)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the one-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   # 
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        reg = self.reg
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * reg * np.sum(W1**2) 
        if self.h:
            loss += 0.5 * reg * np.sum(W2**2)
            dout, grads['W2'], grads['b2'] = fc_backward(dout, c2)
            dout = relu_backward(dout, cr)
            grads['W2'] += reg * W2

        dout, grads['W1'], grads['b1'] = fc_backward(dout, c1)
        if return_dx:
            return dout
            pass
        
        grads['W1'] += reg * W1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def save(self, filepath):
        with open(filepath, "wb") as fp:   
            pickle.dump(self.params, fp, protocol = pickle.HIGHEST_PROTOCOL) 
            
    def load(self, filepath):
        with open(filepath, "rb") as fp:  
            self.params = pickle.load(fp)  
