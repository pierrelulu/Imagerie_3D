import cv2
import numpy as np
import glob

def load_calibration():
    # 1. Paramètres de la mire
    nx, ny = 7, 7
    square_size = 0.020

    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) * square_size

    objpoints, imgpoints = [], []
    images = glob.glob('calib_images/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if not ret:
            continue
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Calibration: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f" % (K[0,0], K[1,1], K[0,2], K[1,2]))
    return K, dist, rvecs, tvecs

# --- SPARSE MATCHING TRIANGULATION ---
def detect_and_match_sparse(img1, img2):
    sift = cv2.SIFT_create(nfeatures=10000)
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1
    index_params = {'algorithm':FLANN_INDEX_KDTREE, 'trees':5}
    search_params = {'checks':50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(d1, d2, k=2)
    pts1, pts2 = [], []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            pts1.append(k1[m.queryIdx].pt)
            pts2.append(k2[m.trainIdx].pt)
    return np.float32(pts1), np.float32(pts2)

def estimate_pose(pts1_norm, pts2_norm):
    E, mask = cv2.findEssentialMat(pts1_norm, pts2_norm, focal=1.0, pp=(0.,0.), method=cv2.RANSAC, prob=0.999, threshold=0.8)
    _, R, t, mask_pose = cv2.recoverPose(E, pts1_norm, pts2_norm)
    return R, t, mask_pose

def triangulate_sparse(K, dist, img1, img2):
    pts1, pts2 = detect_and_match_sparse(img1, img2)
    pts1_ud = cv2.undistortPoints(pts1.reshape(-1,1,2), K, dist).reshape(-1,2)
    pts2_ud = cv2.undistortPoints(pts2.reshape(-1,1,2), K, dist).reshape(-1,2)
    R, t, mask = estimate_pose(pts1_ud, pts2_ud)
    mask = mask.ravel().astype(bool)
    pts1_in = pts1_ud[mask]
    pts2_in = pts2_ud[mask]
    # P0,P1 in normalized coords
    P0 = np.hstack((np.eye(3), np.zeros((3,1))))
    P1 = np.hstack((R, t))
    pts1_t = pts1_in.T.astype(np.float64)
    pts2_t = pts2_in.T.astype(np.float64)
    homog = cv2.triangulatePoints(P0, P1, pts1_t, pts2_t)
    pts3D = (homog[:3]/homog[3]).T
    print(f"Sparse: {pts3D.shape[0]} points")
    return pts3D

# --- DENSE STEREO MATCHING TRIANGULATION ---
def triangulate_dense(img1, img2, K, dist, R, t):
    h, w = img1.shape
    # stereoRectify to compute rectify transforms and Q
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, dist, K, dist, (w,h), R, t, flags=0)
    map1x, map1y = cv2.initUndistortRectifyMap(K, dist, R1, P1, (w,h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, dist, R2, P2, (w,h), cv2.CV_32FC1)
    r1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    r2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
    # compute disparity
    num_disp = 16*8
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disp, blockSize=5,
                                   P1=8*3*5**2, P2=32*3*5**2, disp12MaxDiff=1,
                                   uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)
    disp = stereo.compute(r1, r2).astype(np.float32) / 16.0
    # reproject to 3D
    pts3D = cv2.reprojectImageTo3D(disp, Q)
    mask = disp > disp.min()
    pts3D = pts3D[mask]
    print(f"Dense: {pts3D.shape[0]} points")
    return pts3D

# ply exporter
def save_ply(filename, pts):
    with open(filename,'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(pts)}\n')
        f.write('property float x\nproperty float y\nproperty float z\nend_header\n')
        for x,y,z in pts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

if __name__ == '__main__':
    img_paths = sorted(glob.glob('bundle_images/*.jpg'))
    img1 = cv2.imread(img_paths[0], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_paths[1], cv2.IMREAD_GRAYSCALE)
    K, dist, _, _ = load_calibration()

    # sparse reconstruction
    pts_sparse = triangulate_sparse(K, dist, img1, img2)
    save_ply('sparse_cloud.ply', pts_sparse)

    # estimate pose from sparse for dense rectify
    pts1_ud = cv2.undistortPoints(pts_sparse.reshape(-1,1,3)[:,:,:2], K, dist).reshape(-1,2)
    # re-estimate R,t with original feature matching
    pts1, pts2 = detect_and_match_sparse(img1, img2)
    pts1_norm = cv2.undistortPoints(pts1.reshape(-1,1,2),K,dist).reshape(-1,2)
    pts2_norm = cv2.undistortPoints(pts2.reshape(-1,1,2),K,dist).reshape(-1,2)
    R, t, _ = estimate_pose(pts1_norm, pts2_norm)

    # dense reconstruction
    pts_dense = triangulate_dense(img1, img2, K, dist, R, t)
    save_ply('dense_cloud.ply', pts_dense)
    print("Saved sparse_cloud.ply and dense_cloud.ply")
