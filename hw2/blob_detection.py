import os

import numpy as np

from common import (find_maxima, read_img, visualize_maxima,
                    visualize_scale_space)

from filters import convolve

def gaussian_filter(image, sigma):
    # Given an image, apply a Gaussian filter with the input kernel size
    # and standard deviation
    # Input
    #   image: image of size HxW
    #   sigma: scalar standard deviation of Gaussian Kernel
    #
    # Output
    #   Gaussian filtered image of size HxW
    H, W = image.shape
    # -- good heuristic way of setting kernel size
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)

    # Ensure that the kernel size isn't too big and is odd
    kernel_size = min(kernel_size, min(H, W) // 2)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1

    # TODO implement gaussian filtering of size kernel_size x kernel_size
    # You can use your convolution function or scipy's convolution function

    k = np.zeros((kernel_size,kernel_size))
    offset = (kernel_size-1)/2
    for i in range(kernel_size):
        for j in range(kernel_size):
            k[i,j] = (1/(2*np.pi*sigma**2))*np.exp(-((i-offset)**2+(j-offset)**2)/(2*sigma**2)) 
    output = convolve(image, k)

    return output


def scale_space(image, min_sigma, k=np.sqrt(2), S=8):
    # Calcualtes a DoG scale space of the image
    # Input
    #   image: image of size HxW
    #   min_sigma: smallest sigma in scale space
    #   k: scalar multiplier for scale space
    #   S: number of scales considers
    #
    # Output
    #   Scale Space of size HxWx(S-1)
    H,W = image.shape
    output = None
    gs = []
    gs.append(gaussian_filter(image, min_sigma))
    cursig = min_sigma
    for i in range(S-1):
        cursig = cursig*k
        gs.append(gaussian_filter(image, cursig))
    diffs = []
    for i in range(S-1):
        diffs.append(gs[i]-gs[i+1])

    output = np.stack(diffs, axis = 2)
    return output


def main():
    image = read_img('./cells/001cell.png')
    image = (image - np.mean(image))/np.std(image)

    # # Create directory for polka_detections
    # if not os.path.exists("./polka_detections"):
    #     os.makedirs("./polka_detections")

    # # -- Detecting Polka Dots
    # print("Detect small polka dots")
    # # -- Detect Small Circles
    # sigma_1, sigma_2 = 10,20
    # gauss_1 = gaussian_filter(image, sigma_1)  # to implenent
    # gauss_2 = gaussian_filter(image, sigma_2)  # to implement

    # # calculate difference of gaussians
    # DoG_small = gauss_2 - gauss_1  # to implement

    # # visualize maxima
    # maxima = find_maxima(DoG_small, k_xy=int(sigma_1))
    # visualize_scale_space(DoG_small, sigma_1, sigma_2 / sigma_1,
    #                       './polka_detections/polka_small_DoG.png')
    # visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
    #                  './polka_detections/polka_small.png')

    # # # -- Detect Large Circles
    # print("Detect large polka dots")
    # sigma_1, sigma_2 = 45, 60
    # gauss_1 = gaussian_filter(image, sigma_1)
    # gauss_2 = gaussian_filter(image, sigma_2)

    # # calculate difference of gaussians
    # DoG_large = gauss_2 - gauss_1

    # # visualize maxima
    # # Value of k_xy is a sugguestion; feel free to change it as you wish.
    # maxima = find_maxima(DoG_large, k_xy=10)
    # visualize_scale_space(DoG_large, sigma_1, sigma_2 / sigma_1,
    #                       './polka_detections/polka_large_DoG.png')
    # visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
    #                  './polka_detections/polka_large.png')

    # TODO Implement scale_space() and try to find both polka dots
    k = 1.3
    min_sigma = 2
    scalespace = scale_space(image, min_sigma, k, 8)
    visualize_scale_space(scalespace, min_sigma, k, './cell_detections/001cell_scalespace.png')
    # TODO Detect the cells in any one (or more) image(s) from vgg_cells
    # # Create directory for polka_detections
    # if not os.path.exists("./cell_detections"):
    #     os.makedirs("./cell_detections")
    kxy = 10
    ks = 4
    maxima = find_maxima(scalespace, kxy, ks)

    visualize_maxima(image, maxima, min_sigma, k,
                     './cell_detections/001_kxy'+str(kxy)+'_ks'+str(ks)+'Detect'+str(len(maxima))+'.png')
    # visualize_maxima(image, maxima, min_sigma, k,
    #                  './cell_detections/polka_kxy'+str(kxy)+'_ks'+str(ks)+'Detect'+str(len(maxima))+'.png')


if __name__ == '__main__':
    main()
