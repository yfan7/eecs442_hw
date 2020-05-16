"""
main.py for HW3.

feel free to include libraries needed
"""
from google.colab.patches import cv2_imshow
import numpy as np
from matplotlib import pyplot as plt
import cv2
dpath = '/content/gdrive/My Drive/hw3/'
def homography_transform(X, H):
    # Perform homography transformation on a set of points X
    # using homography matrix H
    #
    # Input - a set of 2D points in an array with size (N,2)
    #         a 3*3 homography matrix
    # Output - a set of 2D points in an array with size (N,2)
    X = np.hstack((X, np.ones((len(X),1))))
    Y_hat = X @ H.T
    Y = Y_hat / Y_hat[:,2:3]
      
   
    return Y


def fit_homography(XY):
    # Given two set of points X, Y in one array,
    # fit a homography matrix from X to Y
    #
    # Input - an array with size(N,4), each row contains two
    #         points in the form[x^T_i,y^T_i]1×4
    # Output - a 3*3 homography matrix
    # print(XY.shape)
    N = len(XY)
    X = XY[:,:2]
    Y = XY[:,2:]
    X = np.hstack((X, np.ones((len(X),1)))) 
    Y = np.hstack((Y, np.ones((len(Y),1)))) 
    A = []
    for i in range(N):
      l1 = np.concatenate(([0,0,0],-1*X[i],Y[i][1]*X[i]))
      l2 = np.concatenate((X[i],[0,0,0],-1*Y[i][0]*X[i]))
      A.append(l1)
      A.append(l2)
    A = np.vstack(A)
    eigenValues, eigenVectors = np.linalg.eig(A.T @ A)
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    v = eigenVectors[:,-1]
    H = np.vstack((v[:3].T, v[3:6].T, v[6:].T))
    return H


def p1():
    # code for Q1.2.3 - Q1.2.5
    # 1. load points X from p1/transform.npy
    data = np.load(dpath + 'p1/transform.npy')
    X = data[:,:2]
    old_X = X
    Y = data[:,2:]
    X = np.hstack((X, np.ones((len(X),1)))) 
    
    # print(rst[0].T)
    # 2. fit a transformation y=Sx+t
    rst = np.linalg.lstsq(X.T @ X, X.T @ Y, rcond = None)
    # 3. transform the points
    Y_hat = X @rst[0]
    # 4. plot the original points and transformed points
    plt.scatter(old_X[:, 0], old_X[:, 1], c="red")  # X
    plt.scatter(Y[:, 0], Y[:, 1], c="green")  # Y
    plt.scatter(Y_hat[:, 0], Y_hat[:, 1], c="blue")  # Y_hat
    plt.savefig(dpath+'125.jpg')
    plt.close()
    # print(transformed)
    # code for Q1.2.6 - Q1.2.8
    case = 8  # you will encounter 8 different transformations
    for i in range(case):
        XY = np.load(dpath+'p1/points_case_'+str(i)+'.npy')
        # 1. generate your Homography matrix H using X and Y
        #
        #    specifically: fill function fit_homography()
        #    such that H = fit_homography(XY)
        H = fit_homography(XY)
        # 2. Report H in your report
        print(H)
        # 3. Transform the points using H
        #
        #    specifically: fill function homography_transsform
        #    such that Y_H = homography_transform(X, H)
        Y_H = homography_transform(XY[:, :2], H)
        # 4. Visualize points as three images in one figure
        # the following code plot figure for you
        plt.scatter(XY[:, 1], XY[:, 0], c="red")  # X
        plt.scatter(XY[:, 3], XY[:, 2], c="green")  # Y
        plt.scatter(Y_H[:, 1], Y_H[:, 0], c="blue")  # Y_hat
        plt.savefig(dpath+'./case_'+str(i))
        plt.close()

def totalls(X,Y):
    Z = np.hstack((X,Y))

    mu = np.mean(Z, axis = 0, keepdims = True)

    eigenValues, eigenVectors = np.linalg.eig((Z - mu).T @ (Z - mu))
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    v = eigenVectors[:,-1:]
    d = mu @ v
  
    return v,d[0,0]

def stitchimage(imgleft, imgright):
    # 1. extract descriptors from images
    #    you may use SIFT/SURF of opencv
    # imgleft = cv2.resize(imgleft,(600, int(600*imgleft.shape[1]/imgleft.shape[0])))
    # imgright = cv2.resize(imgright,(600, int(600*imgright.shape[1]/imgright.shape[0])))
    print(imgleft.shape)
    grayl= cv2.cvtColor(imgleft,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kpl,desl = sift.detectAndCompute(grayl,None)
    # print(desl)
    # imgleft=cv2.drawKeypoints(imgleft,kpl,imgleft)
    # cv2.imwrite(dpath+'bbbleft_keypoints.jpg',imgleft)
    # cv2_imshow(imgleft)
    grayr= cv2.cvtColor(imgright,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kpr,desr = sift.detectAndCompute(grayr,None)
    # imgright=cv2.drawKeypoints(imgright,kpr,imgright)
    # cv2.imwrite(dpath+'bbbright_keypoints.jpg',imgright)
    # cv2_imshow(imgright)

    kpaleft = np.array([[p.pt[0],p.pt[1]] for p in kpl])
    tl = kpaleft
    X = kpaleft
    # X = ((kpaleft - np.mean(kpaleft, axis = 0))/np.std(kpaleft,axis = 0))
    kpaleft = desl[:,None,:]
    kparight = np.array([[p.pt[0],p.pt[1]] for p in kpr])
    tr = kparight
    # kparight = (kparight - np.mean(kparight, axis = 0))/np.std(kparight,axis = 0)
    Y = kparight
    kparight = np.transpose(desr[:,None,:],(1,0,2))
    # desl = ((desl - np.mean(desl, axis = 0, keepdims=True))/np.std(desl,axis = 0, keepdims=True))
    # desr = ((desr - np.mean(desr, axis = 0, keepdims=True))/np.std(desr,axis = 0, keepdims=True))
    lnorm = np.sum(desl**2, axis = 1, keepdims = True)
    rnorm = np.sum(desr**2, axis = 1, keepdims = True)
    dist = (lnorm + rnorm.T -2*(desl @ desr.T))**0.5


    print(np.min(dist),np.max(dist))
    # print(lnorm,rnorm)
    # 2. select paired descriptors

    mins = np.amin(dist,axis = 1)
    print(np.argpartition(dist,2))
    secminsargs = np.argpartition(dist,2)[:,2]
    secmins = dist[np.arange(len(dist)),secminsargs]
    print(mins.shape, secmins.shape)
    s = mins/secmins < 0.5

    print(mins, secmins)
    print("s",np.sum(s))
    # 3. run RANSAC to find a transformation
    #    matrix which has most innerliers
    bestLine, bestCount = None, -1
    bestd = 0
    id_x = np.nonzero(s)[0]

    id_y = np.argmin(dist, axis = 1)[id_x]
    assert(len(id_x) == len(id_y))
    data = np.hstack((X[id_x,:],Y[id_y,:]))
    best_dist = None
    for i in range(400):
        id_s = np.random.randint(0,len(id_x),size = 50)
        H = fit_homography(data)
        Y_h = homography_transform(X[id_x[id_s],:],H)[:,:2]
        # print(n.shape,data.shape,d.shape)
        dist = np.linalg.norm(Y_h - Y[id_y[id_s],:],axis = 1)
        ct = np.sum(dist < 20)
        if ct > bestCount:
            bestLine = H
            bestCount = ct
            best_dist = dist
        # print(dist.shape)
    # print("min dist", np.amin(dist))
    print("best c",bestCount)
    print("bestLine", bestLine)
    print("residual",np.mean(best_dist[best_dist < 20]))
    match_idx = (best_dist < 20)
    kp1 = []
    kp2 = []
    nidx  =  np.nonzero(match_idx)[0]
    # print(nidx.shape,id_x.shape)
    match1to2 = []
    midx = 0
    for i in nidx:
        kp1.append(cv2.KeyPoint(tl[id_x[i]][0],tl[id_x[i]][1],1))
        kp2.append(cv2.KeyPoint(tr[id_y[i]][0],tr[id_y[i]][1],1))
        match1to2.append(cv2.DMatch(midx,midx,dist[i]))
        midx += 1
    
    match = None
    match = cv2.drawMatches(grayl,kp1,grayr,kp2,match1to2 ,match )
    

    # 4. warp one image by your transformation
    #    matrix
    #
    #    Hint:
    #    a. you can use function of opencv to warp image
    #    b. Be careful about final image size
    bestLine = bestLine / bestLine[2,2]
    translate = np.array([[ 1 , 0 , 0],[ 0 , 1 , 0],[ 0 , 0 ,  1 ]])
    H_inv = np.linalg.inv(bestLine)
    print(H_inv)
    warped = cv2.warpPerspective(imgright,H_inv, (600,900))
    save_img(warped, dpath+'warptree.jpg')
    # save_img(match, dpath+'matchbox.jpg')
    cv2_imshow(warped)
    # cv2_imshow(match)
    
    # 5. combine two images, use average of them
    #    in the overlap area
 
    warpleftc = cv2.warpPerspective(imgleft,translate @ bestLine, (2500,1500))
    warprightc = cv2.warpPerspective(imgright,translate.astype(np.float) , (2500,1500))
    # cv2_imshow(warpleftc)
    # cv2_imshow(warprightc)
    img = warpleftc.astype(np.int32) + warprightc.astype(np.int32)
    print("sum",np.sum(warpleftc, axis = 2).shape)
    overlap = np.logical_and(np.sum(warpleftc, axis = 2),np.sum(warprightc, axis = 2))
    print(overlap.shape,img[overlap,:].shape)
    img[overlap,:]  -= (0.5*img[overlap,:]).astype(np.int32)
    cv2_imshow(img)
    return img, bestLine

def p2(p1, p2, savename):
    # read left and right images
    imgleft = read_img(p1)
    imgright = read_img(p2)
    # grayleft = cv2.cvtColor(imgleft, cv2.COLOR_BGR2GRAY).astype(np.double)
    # grayright = cv2.cvtColor(imgright, cv2.COLOR_BGR2GRAY).astype(np.double)
    # save_img(grayleft, dpath+'grayleft.jpg')
    # save_img(grayright, dpath+'grayright.jpg')
    # stitch image
    output, H = stitchimage(imgleft, imgright)
    # save stitched image
    save_img(output, dpath+'./{}.jpg'.format(savename))
    return H


if __name__ == "__main__":
    # Problem 1
    # p1()

    # Problem 2
    
    # p2(dpath+'p2/uttower_left.jpg', dpath+'p2/uttower_right.jpg', 'uttower')
    # p2(dpath+'p2/bbb_left.jpg', dpath+'p2/bbb_right.jpg', 'bbb')
    
    # Problem 3
    # add your code for implementing Problem 3
    #
    front = read_img(dpath+'tree_front.jpg')
    side = read_img(dpath+'tree_left.jpg')
    mark = read_img(dpath+'mark.jpg')
    H = fit_homography(np.array([[0,0,184,408],[400,0,314,410],[400,400,318,539],[0,400,180,539]]))
    H = H/H[2,2]
    print(H)
    changed_mark = cv2.warpPerspective(mark,H, (600,800))
    mark_front = changed_mark*0.6+front
    cv2_imshow(mark_front)
    save_img(mark_front, dpath+'mark_front.jpg')
    Z = p2(dpath+'tree_front.jpg', dpath+'tree_left.jpg', 'tree')
    changed_mark = cv2.warpPerspective(mark,Z@H, (600,800))
    mark_side = changed_mark*0.6+side
    cv2_imshow(mark_side)
    save_img(mark_side, dpath+'mark_side.jpg')
    # Hint:
    # you can reuse functions in Problem 2
