import cv2
import numpy as np
import os
import glob
from scipy.spatial.transform import Rotation as R_


feature_type = "orb"  # orb is only feature type avaliable as of current

# |------------------------ Settings ------------------------------|
# image_folder = "Data/rgbd_dataset_freiburg1_floor/rgb"
# depth_folder = "Data/rgbd_dataset_freiburg1_floor/depth"
image_folder = "Data/rgbd_dataset_freiburg1_room/rgb"
depth_folder = "Data/rgbd_dataset_freiburg1_room/depth"

# |------------------------ Camera Intrinsics ------------------------|
K = np.array([[517.3, 0, 318.6],
              [0, 516.5, 255.3],
              [0, 0, 1] ])

# |------------------------ Distortion Coefficients ------------------------|
dist_coeffs = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])


def compute_scale_from_depth(pts1, pts2, depth1, depth2, K):
    """Scale estimation function using depth image"""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    def pixel_to_cam(pt, depth_val):
        x = (pt[0] - cx) * depth_val / fx
        y = (pt[1] - cy) * depth_val / fy
        return np.array([x, y, depth_val])

    pts3d_1, pts3d_2 = [], []

    for p1, p2 in zip(pts1, pts2):
        u1, v1 = int(p1[0]), int(p1[1])
        u2, v2 = int(p2[0]), int(p2[1])

        if (0 <= u1 < depth1.shape[1]) and (0 <= v1 < depth1.shape[0]) and \
           (0 <= u2 < depth2.shape[1]) and (0 <= v2 < depth2.shape[0]):
            d1 = depth1[v1, u1] / 5000.0
            d2 = depth2[v2, u2] / 5000.0

            if 0.1 < d1 < 5.0 and 0.1 < d2 < 5.0:
                pts3d_1.append(pixel_to_cam(p1, d1))
                pts3d_2.append(pixel_to_cam(p2, d2))

    if len(pts3d_1) < 5:
        return 1.0  # fallback scale

    pts3d_1 = np.array(pts3d_1)
    pts3d_2 = np.array(pts3d_2)
    distances = np.linalg.norm(pts3d_2 - pts3d_1, axis=1)
    return np.mean(distances)

# |------------------------ Feature detector ------------------------|
if feature_type == "orb":
    detector = cv2.ORB_create()
else:
    raise ValueError("Unknown feature type: choose 'orb'")

# |------------------------ Load sorted image paths ------------------------|
image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")))
depth_paths = sorted(glob.glob(os.path.join(depth_folder, "*.png")))
output_file = open("vo_output.txt", "w")
timestamps = []

# |------------------------ Initialize pose ------------------------|
pose = np.eye(4)
trajectory = np.zeros((600, 600, 3), dtype=np.uint8)
positions = []



# |------------------------ First frame ------------------------|
prev_img = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
prev_img = cv2.undistort(prev_img, K, dist_coeffs)
prev_kp, prev_des = detector.detectAndCompute(prev_img, None)


#|------------------------ Main loop ------------------------|
""" Main loop of code.
    Calls feature detection on each image
    Matches features to the previous image
    estimates E and pose
    estimates Scale 
    Outputs
"""
for idx in range(1, len(image_paths)):
    curr_img = cv2.imread(image_paths[idx], cv2.IMREAD_GRAYSCALE)
    curr_img = cv2.undistort(curr_img, K, dist_coeffs)
    
    # --- Detects and discribes features ---
    kp, des = detector.detectAndCompute(curr_img, None)
    if prev_des is None or des is None:
        print(f"Skipping frame {idx} due to missing descriptors.")
        continue

    # --- Matches features between images ---
    if feature_type == "orb":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(prev_des, des)
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) < 8:
        print(f"Not enough matches in frame {idx}, skipping.")
        continue

    # --- Estimates Essential matrix and decomposes for translation and rotation vectors --- 
    pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    E, mask = cv2.findEssentialMat(pts2, pts1, K, method=cv2.RANSAC, threshold=1.0)
    _, R, t, mask_pose = cv2.recoverPose(E, pts2, pts1, K)

    # --- Load depth images ---
    depth1 = cv2.imread(depth_paths[idx - 1], cv2.IMREAD_UNCHANGED)
    depth2 = cv2.imread(depth_paths[idx], cv2.IMREAD_UNCHANGED)

    # --- Estimate scale ---
    scale = compute_scale_from_depth(pts1.reshape(-1, 2), pts2.reshape(-1, 2), depth1, depth2, K )

    # --- Update pose from Rot and trans vectors---
    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = (t * scale).squeeze()
    pose = pose @ np.linalg.inv(Rt)

    # --- Visualize ---
    x, z = pose[0, 3], pose[2, 3]
    pt = (int(x) + 300, int(z) + 100)
    positions.append(pt)
    for p in positions:
        cv2.circle(trajectory, p, 1, (0, 255, 0), 1)

    # --- Output to vo_output.txt in TUM RGB-D trajectory format ---
    timestamp = float(os.path.splitext(os.path.basename(image_paths[idx]))[0])
    timestamps.append(timestamp)
    tx, ty, tz = pose[:3, 3]
    quat = R_.from_matrix(pose[:3, :3]).as_quat()
    qx, qy, qz, qw = quat
    output_file.write(f"{timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

    vis = cv2.drawMatches(prev_img, prev_kp, curr_img, kp, matches[:50], None)
    cv2.imshow("Matches", vis)
    cv2.imshow("Trajectory", trajectory)
    if cv2.waitKey(1) == 27:
        break
    prev_img, prev_kp, prev_des = curr_img, kp, des

output_file.close()
cv2.destroyAllWindows()
