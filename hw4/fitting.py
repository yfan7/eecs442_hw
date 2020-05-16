import numpy as np
import matplotlib.pyplot as plt
from layers import *

def lsq(X,Y, learning_rate = 5e-3):
    """
    Inputs:
    - X: Array, of shape (N,2)
    - Y: Array, of shape (N,2)
    - learning_rate: A scalar for initial learning rate
    """
    S = np.ones((2,2))
    t = np.zeros(2)

    for i in range(10000):
        fwd, cache = fc_forward(X,S,t)
        loss, dloss = l2_loss(fwd,Y)
        dx, dS, dt = fc_backward(dloss,cache)
     
        # You now have the derivative of w in dw and the derivative 
        # of b in dd, update w, b with gradient descent
        S -= dS * learning_rate
        t -= dt * learning_rate
        
    return S, t
        

def main():
    XY = np.load("points_case.npy")
    x, y = XY[:,:2], XY[:,2:]
    # Tune your learning rate here.
    S, t = lsq(x, y,2e-4)
    print(S, t)
    y_hat = x.dot(S) + t
    plt.scatter(x[:,0],x[:,1],c="red")
    plt.scatter(y[:,0],y[:,1],c="green")
    plt.scatter(y_hat[:,0],y_hat[:,1],c="blue",marker='.')
    plt.savefig("./case.jpg")

if __name__ == "__main__":
    main()
