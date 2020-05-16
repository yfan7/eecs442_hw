import os

import numpy as np

from common import read_img, save_img
from filters import edge_detection

def corner_score(image, u=5, v=5, window_size=(5, 5)):
    # Given an input image, x_offset, y_offset, and window_size,
    # return the function E(u,v) for window size W
    # corner detector score for that pixel.
    # Input- image: H x W
    #        u: a scalar for x offset
    #        v: a scalar for y offset
    #        window_size: a tuple for window size
    #
    # Output- results: a image of size H x W
    # Use zero-padding to handle window values outside of the image.
    output = None
    H,W = image.shape
    h,w = window_size
    ph = int(h/2)+abs(u)
    pw = int(w/2)+abs(v)
    im = np.pad(image,((ph,ph),(pw,pw),), 'constant', constant_values = (0,0))
    output = np.zeros(image.shape)
    for i in range(H):
        for j in range(W):
            w1 = im[i+ph-int(h/2):i+ph+int(h/2),j+pw-int(w/2):j+pw+int(w/2)]
            w2 = im[i+ph-int(h/2)+u:i+ph+int(h/2)+u,j+pw-int(w/2)+v:j+pw+int(w/2)+v]
            output[i,j] = np.sum((w1-w2)**2)
    return output


def harris_detector(image, window_size=(5, 5)):
    # Given an input image, calculate the Harris Detector score for all pixels
    # Input- image: H x W
    # Output- results: a image of size H x W
    #
    # You can use same-padding for intensity (or 0-padding for derivatives)
    # to handle window values outside of the image.

    # compute the derivatives

    Ix = None
    Iy = None
    _, Ix, Iy = edge_detection(image)
    H,W = image.shape
    h,w = window_size
    ph = int(h/2)
    pw = int(w/2)
    Ixp = np.pad(Ix,((ph,ph),(pw,pw),), 'constant', constant_values = (0,0))
    Iyp = np.pad(Iy,((ph,ph),(pw,pw),), 'constant', constant_values = (0,0))
    Ixx = np.zeros(image.shape)
    Ixy = np.zeros(image.shape)
    Iyy = np.zeros(image.shape)

    va = 1/(2*np.log(2))
    k = np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            k[i,j] = 1/(2*np.pi*va)*np.exp(-((i-2)**2+(j-2)**2)/(2*va)) 
  
    for i in range(H):
        for j in range(W):
            windowx = Ixp[i:i+h,j:j+w] * k
            windowy = Iyp[i:i+h,j:j+w] * k
            Ixx[i,j] = np.sum(windowx**2)
            Iyy[i,j] = np.sum(windowy**2)
            Ixy[i,j] = np.dot(windowx.reshape(-1),windowy.reshape(-1))


    # For each image location, construct the structure tensor and calculate
    # the Harris response
    response = np.zeros(image.shape)
    for i in range(H):
        for j in range(W):
            M = np.array([[Ixx[i,j], Ixy[i,j]],[Ixy[i,j],Iyy[i,j]]])
            response[i,j] = np.linalg.det(M) - 0.05*(np.trace(M)**2)

    return response


def main():
    img = read_img('./grace_hopper.png')

    # Feature Detection
    if not os.path.exists("./feature_detection"):
        os.makedirs("./feature_detection")

    # Define offsets and window size and calulcate corner score
    u, v, W = 5, 0, (5, 5)

    # score = corner_score(img, u, v, W)
    # save_img(score, "./feature_detection/corner_score_d5.png")

    harris_corners = harris_detector(img)
    save_img(harris_corners, "./feature_detection/harris_response.png")


if __name__ == "__main__":
    main()
