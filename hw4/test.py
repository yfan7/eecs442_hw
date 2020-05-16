import numpy as np
from layers import *

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def compare(output, r, name):
    wrong = False
    for k, v in output.items():
        if rel_error(v, r[k]) > 1e-5:
            print(name + ' fail! ' + k + ' is wrong.')
            wrong = True
            break
    if not wrong:
        print(name + ' pass!')

def test_fc():
    np.random.seed(442)
    x = np.random.randn(10,5)
    w = np.random.randn(5,3)
    b = np.random.randn(3)
    output = {}
    output['y'], cache = fc_forward(x, w, b)

    dout = np.random.randn(*output['y'].shape)
    output['dx'], output['dw'], output['db'] = fc_backward(dout, cache)
    
    r = np.load('fc.npz')
    compare(output, r, 'fc')


def test_relu():
    np.random.seed(442)
    x = np.random.randn(10)
    output = {}
    output['y'], cache = relu_forward(x)

    dout = np.random.randn(*output['y'].shape)
    output['dx'] = relu_backward(dout, cache)

    r = np.load('relu.npz')
    compare(output, r, 'relu')


def test_l2_loss():
    np.random.seed(442)
    x = np.random.randn(10, 9)
    y = np.random.randn(10, 9)
    output = {}
    output['loss'], output['dx'] = l2_loss(x,y)

    r = np.load('l2_loss.npz')
    compare(output, r, 'l2_loss')

def test_softmax_loss():
    np.random.seed(442)
    x = np.random.randn(10, 9)
    y = np.random.randint(0, 9, size = 10)
    output = {}
    output['loss'], output['dx'] = softmax_loss(x,y)

    r = np.load('softmax_loss.npz')
    compare(output, r, 'softmax_loss')

if __name__=='__main__':
    test_fc()
    test_relu()
    test_l2_loss()
    test_softmax_loss()
