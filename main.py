import cv2
import numpy as np
from numpy.linalg import inv
import glob
from tqdm import tqdm
import open3d as o3d
from skimage.transform import EssentialMatrixTransform

def load_calibration():
    # 1. Paramètres de la mire
    nx, ny = 6, 6
    square_size = 0.023

    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) * square_size

    objpoints, imgpoints = [], []
    images = glob.glob('calib_image_laet/*.jpg')

    for fname in tqdm(images, desc="Extract features from Calibration images", unit="img"):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if not ret:
            continue

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Calibration matrix: \n", K)
    print("Distorsion coefficient: \n", dist)
    print("Calibration: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f" % (K[0,0], K[1,1], K[0,2], K[1,2]))
    print("Reprojection Error: ", ret)
    return K, dist, rvecs, tvecs

# --- SPARSE MATCHING TRIANGULATION ---
def detect_and_match_sparse(img1, img2,
                            *,
                            ratio: float = 0.7,
                            ransac: bool = True,
                            ransac_thresh: float = 3.0,
                            confidence: float = 0.999,
                            use_roi: bool = True,
                            roi_rect: tuple | None = None,
                            default_roi_size: int = 800,
                            visualize: bool = True,
                            winname: str = "matches"):

    # ---------- 1. détection SIFT ----------
    sift = cv2.SIFT_create(nfeatures=10_000,
                           nOctaveLayers=4,
                           contrastThreshold=0.02,
                           edgeThreshold=8,
                           sigma=1.4)
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)

    # ---------- 2. matching + ratio test ----------
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    raw_matches = flann.knnMatch(d1, d2, k=2)

    good, pts1, pts2 = [], [], []
    for m, n in raw_matches:
        if m.distance < ratio * n.distance:
            good.append(m)
            pts1.append(k1[m.queryIdx].pt)
            pts2.append(k2[m.trainIdx].pt)

    #print(f"[DEBUG] après ratio‑test            : {len(good)} matches")

    # ---------- 3. filtre ROI (facultatif) ----------
    if use_roi and len(good):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if roi_rect is None:
            # carré centré par défaut
            half = default_roi_size // 2
            roi1 = (w1 // 2 - half, w1 // 2 + half, h1 // 2 - half, h1 // 2 + half)
            roi2 = (w2 // 2 - half, w2 // 2 + half, h2 // 2 - half, h2 // 2 + half)
        else:
            x, y, w, h = roi_rect
            # on borne les rectangles aux images pour éviter les IndexError
            roi1 = (max(0, x), min(w1, x + w), max(0, y), min(h1, y + h))
            roi2 = (max(0, x), min(w2, x + w), max(0, y), min(h2, y + h))

        keep = []
        for (x1, y1), (x2, y2) in zip(pts1, pts2):
            ok1 = roi1[0] <= x1 <= roi1[1] and roi1[2] <= y1 <= roi1[3]
            ok2 = roi2[0] <= x2 <= roi2[1] and roi2[2] <= y2 <= roi2[3]
            keep.append(ok1 and ok2)
        keep = np.array(keep, dtype=bool)

        good = [m for m, k in zip(good, keep) if k]
        pts1 = [p for p, k in zip(pts1, keep) if k]
        pts2 = [p for p, k in zip(pts2, keep) if k]

        #print(f"[DEBUG] après filtre ROI           : {len(good)} matches")

    # ---------- 4. RANSAC (facultatif) ----------
    if ransac and len(good) >= 4:
        pts1_np = np.float32(pts1).reshape(-1, 1, 2)
        pts2_np = np.float32(pts2).reshape(-1, 1, 2)
        _, mask_r = cv2.findHomography(pts1_np, pts2_np,
                                       cv2.RANSAC,
                                       ransac_thresh,
                                       confidence=confidence)
        mask_r = (mask_r.ravel() == 1) if mask_r is not None else np.zeros(len(good), bool)
    else:
        mask_r = np.ones(len(good), bool)

    #print(f"[DEBUG] inliers retenus (RANSAC)    : {mask_r.sum()}")

    # ---------- 5. listes finales ----------
    good_inliers = [m for m, keep in zip(good, mask_r) if keep]
    pts1_inliers = np.float32([p for p, keep in zip(pts1, mask_r) if keep])
    pts2_inliers = np.float32([p for p, keep in zip(pts2, mask_r) if keep])

    # ---------- 6. affichage ----------
    if visualize and len(good_inliers):
        img1_vis = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1.copy()
        img2_vis = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if img2.ndim == 2 else img2.copy()

        vis = cv2.drawMatches(img1_vis, k1,
                              img2_vis, k2,
                              good_inliers, None,
                              matchColor=(0, 255, 0),
                              singlePointColor=None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # redimensionne pour affichage (facultatif)
        h_vis, w_vis = vis.shape[:2]
        vis = cv2.resize(vis, (round(w_vis / 2), round(h_vis / 2)), interpolation=cv2.INTER_AREA)
        cv2.imshow(winname, vis)
        cv2.waitKey(0)
        cv2.destroyWindow(winname)

    return pts1_inliers, pts2_inliers, k1, k2, good_inliers
# ply exporter
def save_ply(filename, pts):
    with open(filename,'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(pts)}\n')
        f.write('property float x\nproperty float y\nproperty float z\nend_header\n')
        for x,y,z in pts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

def test_cheirality(R, t, K, pts1, pts2):
    # 1) construire P0 et P1
    P0 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P1 = K @ np.hstack((R, t.reshape(3,1)))
    # 2) trianguler tous les points (OpenCV renvoie projectives : X homogène)
    pts4d_hom = cv2.triangulatePoints(P0, P1, pts1.T, pts2.T)  # shape 4×N
    pts3d = pts4d_hom[:3] / pts4d_hom[3]  # shape 3×N
    # 3) compenser si on veut vérifier z>0 dans chaque repère
    #    Caméra 0 : triviale → z = pts3d[2]
    #    Caméra 1 : coord 3D projetée dans repère 1 = R⋅X + t  => z1 = (R⋅X + t)[2]
    z0 = pts3d[2]
    X_cam1 = R @ pts3d + t.reshape(3,1)
    z1 = X_cam1[2]
    # Nombre de points avec z0>0 ET z1>0
    good = np.logical_and(z0 > 0, z1 > 0)
    return np.count_nonzero(good)

def chain_triangulation(image_paths, K, dist):
     # nuage total
    all_pts_world = []

    R_past = np.eye(3)
    t_past = np.zeros((3, 1))

    # tqdm : une itération par paire (img_i, img_{i+1})
    for i in tqdm(range(len(image_paths) - 1),
                  total=len(image_paths) - 1,
                  desc="Triangulation",
                  unit="pair"):

        img_i = cv2.imread(image_paths[i],     cv2.IMREAD_GRAYSCALE)
        img_j = cv2.imread(image_paths[i + 1], cv2.IMREAD_GRAYSCALE)

        #img_i = cv2.undistort(img_i, K, dist)
        #img_j = cv2.undistort(img_j, K, dist)

        # 1. détection/appairage + undistort
        pts_1, pts_2, kpi, kpj, good = detect_and_match_sparse(
            img_i, img_j,
            visualize=True,
            use_roi=True,
            roi_rect=(785,375, 120,300),
            winname=f"Pair {i}-{i + 1}")

        if len(pts_1) < 8:          # ← seuil minimal d’inliers
            tqdm.write(f"[WARN] Pair {i}-{i+1}: {len(pts_1)} inliers → saut.")
            continue

        E, _ = cv2.findEssentialMat(pts_1, pts_2, K)
        _, R_rel, t_rel, _ = cv2.recoverPose(E, pts_1, pts_2, K)
        #t_rel[2, 0] = 0.0

        P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        #Ppast = K @np.hstack((R_past, t_past))
        Pi = K @ np.hstack((R_rel, t_rel))
        print("R_rel : " + str(R_rel))
        print("t_rel : " + str(t_rel))
        print("pts_1 1 : " + str(pts_1[0]))
        print("pts_2 1 : " + str(pts_2[0]))

        pts4D  = cv2.triangulatePoints(P0, Pi, pts_1.T, pts_2.T)
        pts3D  = (pts4D[:3] / pts4D[3]).T

        save_ply('res_3D/pts3D' + str(i) + '.ply', pts3D)

        # Mise à jour repère global
        #R_now  = R_rel @ R_past
        #  = R_rel @ t_past + t_rel

        # Conversion vers monde :
        #pts_world = pts3D @ R_now.T - (t_now.T @ R_now.T).reshape(1, 3)


        all_pts_world.append(pts3D)

        #R_past, t_past = R_now, t_now

    # tout concaténer
    cloud = np.vstack(all_pts_world)
    return cloud

if __name__ == '__main__':
    img_paths = sorted(glob.glob('bundle_image_laet_small/*.jpg'))
    K, dist, _, _ = load_calibration()
    cloud = chain_triangulation(img_paths, K, dist)
    save_ply('sparse_cloud_all.ply', cloud)
    print(f"Nuage 3D global : {cloud.shape[0]} points")
