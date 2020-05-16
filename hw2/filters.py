import os

import numpy as np

from common import read_img, save_img


def image_patches(image, patch_size=(16, 16)):
    # Given an input image and patch_size,
    # return the corresponding image patches made
    # by dividing up the image into patch_size sections.
    # Input- image: H x W
    #        patch_size: a scalar tuple M, N
    # Output- results: a list of images of size M x N

    # TODO: Use slicing to complete the function
    output = []
    H,W = image.shape
    h,w = patch_size
    for i in range(0,H,h):
        for j in range(0,W,w):
            pt  = image[i:i+h,j:j+w]
            avg  = np.mean(pt)
            std = np.std(pt)
            pt = (pt-avg)/std
            output.append(pt)
    return output


def convolve(image, kernel):
    # Return the convolution result: image * kernel.
    # Reminder to implement convolution and not cross-correlation!
    # You can use zero- or wrap-padding.
    #
    # Input- image: H x W
    #        kernel: h x w
    #
    # Output- convolve: H x W
    output = None

    H,W = image.shape
    h,w = kernel.shape
    ph = int(h/2)
    pw = int(w/2)
    im = np.pad(image,((ph,ph),(pw,pw),), 'wrap')
    output = np.zeros(image.shape)
    for i in range(H):
        for j in range(W):
            output[i,j] = np.dot(im[i:i+h,j:j+w].reshape(-1), kernel.reshape(-1))
    return output


def edge_detection(image):
    # Return the gradient magnitude of the input image
    # Input- image: H x W
    # Output- grad_magnitude: H x W

    # TODO: Fix kx, ky
    kx = np.array([[-1/2,0,1/2]])  # 1 x 3
    ky = np.array([[-1/2],[0],[1/2]])  # 3 x 1

    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    # TODO: Use Ix, Iy to calculate grad_magnitude
    grad_magnitude = np.sqrt(Ix**2+Iy**2)

    return grad_magnitude, Ix, Iy


def sobel_operator(image):
    # Return Gx, Gy, and the gradient magnitude.
    # Input- image: H x W
    # Output- Gx, Gy, grad_magnitude: H x W

    # TODO: Use convolve() to complete the function
    Gx, Gy, grad_magnitude = None, None, None
    Gx = convolve(image,np.array([[1,0,-1],[2,0,-2],[1,0,-1]]))
    Gy = convolve(image,np.array([[1,2,1],[0,0,0],[-1,-2,-1]]))
    grad_magnitude = np.sqrt(Gx**2+Gy**2)
    return Gx, Gy, grad_magnitude


def steerable_filter(image, angles=(np.pi * np.arange(6, dtype=np.float) / 6)):
    # Given a list of angels used as alpha in the formula,
    # return the corresponding images based on the formula given in pdf.
    # Input- image: H x W
    #        angels: a list of scalars
    # Output- results: a list of images of H x W

    # TODO: Use convolve() to complete the function
    output = []
    Gx = convolve(image,np.array([[1,0,-1],[2,0,-2],[1,0,-1]]))
    Gy = convolve(image,np.array([[1,2,1],[0,0,0],[-1,-2,-1]]))
    for a in angles:
        output.append(Gx*np.cos(a)+Gy*np.sin(a))

    return output


def main():
    # The main function
    img = read_img('./grace_hopper.png')
    """ Image Patches """
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    # -- Image Patches --
    # Q1
    # patches = image_patches(img)
    # TODO choose a few patches and save them
    # chosen_patches = patches[:3]
    
    # save_img(chosen_patches[0], "./image_patches/q1_patch1.png")
    # save_img(chosen_patches[1], "./image_patches/q1_patch2.png")
    # save_img(chosen_patches[2], "./image_patches/q1_patch3.png")

    # Q2: No code
    """ Gaussian Filter """
    # if not os.path.exists("./gaussian_filter"):
    #     os.makedirs("./gaussian_filter")

    # -- Gaussian Filter --
    # Q1: No code

    # Q2

    # TODO: Calculate the kernel described in the question.
    # There is tolerance for the kernel.
    va = 1/(2*np.log(2))
    k = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            k[i,j] = 1/(2*np.pi*va)*np.exp(-((i-1)**2+(j-1)**2)/(2*va)) 
  
    filtered_gaussian = convolve(img, k)
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")

    # # Q3
    edge_detect, _, _ = edge_detection(img)
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")
    edge_with_gaussian, _, _ = edge_detection(filtered_gaussian)
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")
 
    print("Gaussian Filter is done. ")

    # -- Sobel Operator --
    # if not os.path.exists("./sobel_operator"):
    #     os.makedirs("./sobel_operator")

    # # Q1: No code

    # # Q2
    # Gx, Gy, edge_sobel = sobel_operator(img)
    # save_img(Gx, "./sobel_operator/q2_Gx.png")
    # save_img(Gy, "./sobel_operator/q2_Gy.png")
    # save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")
    # # Q3
    # steerable_list = steerable_filter(img)
    # for i, steerable in enumerate(steerable_list):
    #     save_img(steerable, "./sobel_operator/q3_steerable_{}.png".format(i))

    # print("Sobel Operator is done. ")
    # """ LoG Filter """
    # if not os.path.exists("./log_filter"):
    #     os.makedirs("./log_filter")

    # # Q1
    # kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    # kernel_LoG2 = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
    #                         [0, 2, 3, 5, 5, 5, 3, 2, 0],
    #                         [3, 3, 5, 3, 0, 3, 5, 3, 3],
    #                         [2, 5, 3, -12, -23, -12, 3, 5, 2],
    #                         [2, 5, 0, -23, -40, -23, 0, 5, 2],
    #                         [2, 5, 3, -12, -23, -12, 3, 5, 2],
    #                         [3, 3, 5, 3, 0, 3, 5, 3, 3],
    #                         [0, 2, 3, 5, 5, 5, 3, 2, 0],
    #                         [0, 0, 3, 2, 2, 2, 3, 0, 0]])
    # filtered_LoG1 = convolve(img, kernel_LoG1)
    # save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    # filtered_LoG2 = convolve(img, kernel_LoG2)
    # save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")

    # Q2: No code
    # print("LoG Filter is done. ")


if __name__ == "__main__":
    main()
