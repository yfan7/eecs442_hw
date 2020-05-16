"""
Starter code for EECS 442 W20 HW1
"""
from util import *
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import skimage.transform

def rotX(theta):
    # TODO: Return the rotation matrix for angle theta around X axis
    return np.array([[1 ,0, 0 ],[0, np.cos(theta), -np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])


def rotY(theta):
    # TODO: Return the rotation matrix for angle theta around Y axis
    return np.array([[np.cos(theta) ,0, np.sin(theta) ],[0, 1, 0],[-np.sin(theta), 0, np.cos(theta)]])


def projectOthographicLines(R, t, L):
    pass


def part1():
    # R = [rotY(i) for i in np.arange(0,np.pi,0.1)]
    # generate_gif(R)

    theta = np.pi / 4
    X = rotX(theta)
    Y = rotY(theta)
    renderCube(R = Y @ X, file_name = 'x then y')
    renderCube(R = X @ Y, file_name = 'y then x')

    renderCube(R = rotX(np.arcsin(1/np.sqrt(3))) @ rotY(np.pi/4), file_name = 'diag to pt')
    renderCube(f = np.inf, R = rotX(np.arcsin(1/np.sqrt(3))) @ rotY(np.pi/4), file_name = 'inf diag to pt')
    

def split_triptych(trip):
    # TODO: Split a triptych into thirds and return three channels in numpy arrays
    tri = imageio.imread(trip)
    H, W = tri.shape
    h = int(H/3)
    channels = np.dstack((tri[2*h:3*h], tri[h:2*h], tri[:h]))
   
    return channels

def normalized_cross_correlation(ch1, ch2):
    # TODO: Implement the default similarity metric
    n1 = ch1/ np.linalg.norm(ch1)
    n2 = ch2/ np.linalg.norm(ch2)

    return np.dot(n1.reshape(-1),n2.reshape(-1))


def best_offset(ch1, ch2, metric, Xrange=np.arange(-15, 16), Yrange=np.arange(-15, 16)):
    # TODO: Use metric to align ch2 to ch1 and return optimal offsets
    m = 0
    offsets = None
    for xshift in Xrange:
        for yshift in Yrange:

            n2 = np.roll(ch2, shift = (xshift, yshift), axis = (0,1))
            sim = metric(ch1[10:-10,10:-10],n2[10:-10,10:-10])
         
            if m < sim:
                m = sim
                offsets = (xshift, yshift)
                pass
    return offsets

def only_dot(ch1, ch2):
    return np.dot(ch1.reshape(-1),ch2.reshape(-1))

def align_and_combine(R, G, B, metric):
    # TODO: Use metric to align three channels and return the combined RGB image
    o1 = best_offset(R,G,metric, Xrange=np.arange(-10, 10), Yrange=np.arange(-10, 10))
    o2 = best_offset(R,B,metric, Xrange=np.arange(-10, 10), Yrange=np.arange(-10, 10))
    nG = np.roll(G, shift = o1, axis = (0,1))
    nB = np.roll(B, shift = o2, axis = (0,1))
    print("RG offsets: ",o1, " RB offsets: ",o2)
    return np.dstack((R,nG,nB))

def align_and_combine_pyramid(R, G, B, metric):
    # TODO: Use metric to align three channels and return the combined RGB image
    o1 = best_offset(R,G,metric)
    o2 = best_offset(R,B,metric)
    print("RG offsets: ",o1, " RB offsets: ",o2)
    return o1, o2


def task2_2():
    d = 'tableau/'
    pics = [f for f in listdir(d) if isfile(join(d, f))]
    for fname in pics:
        print(fname)
        c = split_triptych(d+ fname)
        imageio.imwrite('rst/'+fname[:-4] + '_unaligned.png', c)
        imageio.imwrite('rst/'+fname[:-4] + '_aligned_normed.png', align_and_combine(c[:,:,0],c[:,:,1],c[:,:,2],normalized_cross_correlation)[20:-20,20:-20])
        imageio.imwrite('rst/'+fname[:-4] + '_aligned_not_normed.png', align_and_combine(c[:,:,0],c[:,:,1],c[:,:,2],only_dot)[20:-20,20:-20])

def task2_3():
    d = 'tableau/'
    fname = 'seoul_tableau.jpg'
    print(fname)
    c = split_triptych(d+ fname)
    H,W,_ = c.shape

    to1 = to2 = (0,0)
    for i in range(2, -1, -1):
        fac = 4**i
        im = skimage.transform.resize(c, (int(H/fac),int(W/fac),3))
        R = im[:,:,0]

        G = np.roll(im[:,:,1], shift = (4*to1[0],4*to1[1]), axis = (0,1))
        B = np.roll(im[:,:,2], shift = (4*to2[0],4*to2[1]), axis = (0,1))
        o1, o2 = align_and_combine_pyramid(R,G,B,normalized_cross_correlation)
        to1 = (4*to1[0]+ o1[0],4*to1[1]+ o1[1])
        to2 = (4*to2[0]+ o2[0],4*to2[1]+ o2[1])

    print("overall RG offsets: ",to1, " overall RB offsets: ",to2)
    R = c[:,:,0]
    G = np.roll(c[:,:,1], shift = to1, axis = (0,1))
    B = np.roll(c[:,:,2], shift = to2, axis = (0,1))
 
    imageio.imwrite('rst/'+fname[:-4] + '_aligned_normed.png', np.dstack((R,G,B)))
  


def part2():
    task2_3()

def part3():
    indoor = imageio.imread('rubik/indoor.png')
    outdoor = imageio.imread('rubik/outdoor.png')
    plt.imshow(indoor[:,:,2], cmap='gray')
    plt.savefig('indoor_B.png')
    plt.imshow(indoor[:,:,1], cmap='gray')
    plt.savefig('indoor_G.png')
    plt.imshow(indoor[:,:,0], cmap='gray')
    plt.savefig('indoor_R.png')
    plt.imshow(outdoor[:,:,2], cmap='gray')
    plt.savefig('outdoor_B.png')
    plt.imshow(outdoor[:,:,1], cmap='gray')
    plt.savefig('outdoor_G.png')
    plt.imshow(outdoor[:,:,0], cmap='gray')
    plt.savefig('outdoor_R.png')




    ni = skimage.color.rgb2lab(indoor[:,:,:3])
    no = skimage.color.rgb2lab(outdoor[:,:,:3])
    plt.imshow(ni[:,:,2], cmap='gray')
    plt.savefig('new_indoor_B.png')
    plt.imshow(ni[:,:,1], cmap='gray')
    plt.savefig('new_indoor_A.png')
    plt.imshow(ni[:,:,0], cmap='gray')
    plt.savefig('new_indoor_L.png')
    plt.imshow(no[:,:,2], cmap='gray')
    plt.savefig('new_outdoor_B.png')
    plt.imshow(no[:,:,1], cmap='gray')
    plt.savefig('new_outdoor_A.png')
    plt.imshow(no[:,:,0], cmap='gray')
    plt.savefig('new_outdoor_L.png')

def diff_lighting():
    tri = imageio.imread("book2.jpg")
    n1 = skimage.transform.rescale(tri, 0.1,multichannel=True)[50:50+256,15:15+256,:]
    plt.imshow(n1)
    imageio.imwrite('im2.jpg', n1)
    plt.show()

def main():
    # part1()
    # part2()
    # part3()
    diff_lighting()


if __name__ == '__main__':
    main()
