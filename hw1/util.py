# Utilities for EECS 442 Winter 2020 HW1
# Part of the code is based on the Dolly Zoom notebook created by David Fouhey
import imageio
import matplotlib.pyplot as plt
from itertools import product
import numpy as np


def generate_gif(R, file_name='cube.gif'):
    """
    Generate a gif of a rotating cube from a list of rotation matrices.

    Input:  R: A list of 3x3 rotation matrices.
            file_name: file_name to save files to.
    Output: None
    """
    with imageio.get_writer(file_name, mode='I') as writer:
        for rot in R:
            fig = renderCube(f=15, t=(0, 0, 3), R=rot)
            # Now we can save it to a numpy array.
            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(),
                                dtype=np.uint8,
                                sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
            writer.append_data(img)


def renderCube(f=15,
               scaleFToSize=None,
               t=np.array([0, 0, -3]),
               R=np.eye(3),
               file_name=None):
    """
    Renders a cube given camera instrinsics and extrinsics.

    Input:  f: focal length
            scaleFToSize: target size on the retrina (sqrt of area)
            t: camera translation (3x1 numpy array)
            R: camera rotation (3x3 numpy array)
            file_name: file path to save image (if provided)

    Output: Matplotlib figure handle
    """
    # Render the cube
    L = generateCube()
    t = np.array(t)
    pL = projectLines(f, R, t, L)

    if scaleFToSize is not None:
        # Then adjust f so that the image is the right size
        xRange, yRange = xyrange(pL)
        geoMean = (xRange * yRange)**0.5
        f = (f / geoMean) * scaleFToSize
        # re-render with the right focal length
        pL = projectLines(f, R, t, L)

    # Generate plot
    fig = plt.figure()
    plt.title("Cube @ [x=%.1f y=%.1f z=%.1f] f=%f" % (t[0], t[1], t[2], f))
    for i in range(pL.shape[0]):
        u1, v1, u2, v2 = pL[i, :]
        plt.plot((u1, u2), (v1, v2), lineWidth=2)

    # Format plot
    plt.axis('square')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    # Save to file if filename is provided
    if file_name:
        plt.savefig(file_name)

    return fig


def generateCube():
    """
    Generates the lines for a unit cube.
    Output: np array of lines.
    """
    lines = []
    for x, y, z in product([0, 1], [0, 1], [0, 1]):
        # all corners, check changing all the dirensions
        # if in the cube, keep, but then center at 0
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                           (0, 0, 1), (0, 0, -1)]:
            xp, yp, zp = x + dx, y + dy, z + dz
            if min([xp, yp, zp]) >= 0 and max([xp, yp, zp]) <= 1:
                lines.append(
                    (x - 0.5, y - 0.5, z - 0.5, xp - 0.5, yp - 0.5, zp - 0.5))
    return np.vstack(lines)


def xyrange(pL):
    # Returns the X and Y ranges for a line
    X, Y = np.vstack([pL[:, 0], pL[:, 2]]), np.vstack([pL[:, 1], pL[:, 3]])
    return np.max(X) - np.min(X), np.max(Y) - np.min(Y)


def projectOthographicLines(R, t, L):
    # TODO: Rewrite this function to project lines with an orthographic camera
    # You may refer to projectLines() for parameter types
    pL = np.zeros((L.shape[0], 4))
    for i in range(L.shape[0]):
        # rotate and translate
        p = np.dot(R, L[i, :3]) + t
        pp = np.dot(R, L[i, 3:]) + t

  
        pL[i, :2] = p[0:2]
        pL[i, 2:] = pp[0:2]

    return np.vstack(pL)


def projectLines(f, R, t, L):
    """
    Projects lines using the camera intrinsic and extrinsic parameters.
    Input:  f: focal length
            R: camera rotation (3x3 numpy array)
            t: camera translation (3x1 numpy array)
            L: Nx6 numpy array depicting N lines in 3D
    Output: 2D projection of lines (Nx4 numpy array)
    """

    if f == np.inf:
        return projectOthographicLines(R, t, L)
    pL = np.zeros((L.shape[0], 4))
    for i in range(L.shape[0]):
        # rotate and translate
        p = np.dot(R, L[i, :3]) + t
        pp = np.dot(R, L[i, 3:]) + t
        # apply projection u = x*f/z; v = y*f/z
        print("p: {}    pp: {}".format(p[2], pp[2]))
        pL[i, :2] = p[0] * f / p[2], p[1] * f / p[2]
        pL[i, 2:] = pp[0] * f / pp[2], pp[1] * f / pp[2]

    return np.vstack(pL)
