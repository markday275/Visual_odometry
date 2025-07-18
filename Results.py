import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def read_tum_trajectory(file_path):
    """Reads TUM groundtruth or VO format: timestamp tx ty tz qx qy qz qw"""
    traj = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            timestamp = float(parts[0])
            pose = np.array([float(x) for x in parts[1:4]])
            traj.append((timestamp, pose))
    return traj

def associate(gt, vo, max_diff=0.02):
    """Associates timestamps from GT and VO within tolerance"""
    matches = []
    gt_dict = dict(gt)
    vo_dict = dict(vo)
    for t_vo in vo_dict:
        closest = min(gt_dict.keys(), key=lambda t_gt: abs(t_gt - t_vo))
        if abs(closest - t_vo) < max_diff:
            matches.append((closest, t_vo))
    matches.sort()
    return matches

def align_trajectories(gt_list, vo_list, matches):
    """Aligns VO to GT using Umeyama algorithm"""
    gt_xyz = np.array([dict(gt_list)[t_gt] for t_gt, t_vo in matches]).T
    vo_xyz = np.array([dict(vo_list)[t_vo] for t_gt, t_vo in matches]).T

    mu_gt = np.mean(gt_xyz, axis=1, keepdims=True)
    mu_vo = np.mean(vo_xyz, axis=1, keepdims=True)

    X = vo_xyz - mu_vo
    Y = gt_xyz - mu_gt

    S = X @ Y.T / X.shape[1]
    U, _, Vt = np.linalg.svd(S)
    R_opt = U @ Vt
    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1
        R_opt = U @ Vt
    t_opt = mu_gt - R_opt @ mu_vo

    vo_aligned = (R_opt @ vo_xyz) + t_opt
    return gt_xyz.T, vo_aligned.T

def compute_rmse(gt, vo):
    error = np.linalg.norm(gt - vo, axis=1)
    return np.sqrt(np.mean(error ** 2)), error

# |------------------------ Load trajectories ------------------------|
# gt_traj = read_tum_trajectory("Data/rgbd_dataset_freiburg1_floor/groundtruth.txt")
gt_traj = read_tum_trajectory("Data/rgbd_dataset_freiburg1_room/groundtruth.txt")
vo_traj = read_tum_trajectory("vo_output.txt")

# |------------------------ Associate and align ------------------------|
matches = associate(gt_traj, vo_traj)
gt_aligned, vo_aligned = align_trajectories(gt_traj, vo_traj, matches)

# |------------------------ Compute RMSE ------------------------|
rmse, errors = compute_rmse(gt_aligned, vo_aligned)
print(f"ATE RMSE: {rmse:.4f} m")

# |------------------------ Plot ------------------------|
plt.plot(gt_aligned[:, 0], gt_aligned[:, 2], label="Ground Truth")
plt.plot(vo_aligned[:, 0], vo_aligned[:, 2], label="VO Trajectory")
plt.legend()
plt.title("Trajectory Comparison")
plt.xlabel("x [m]")
plt.ylabel("z [m]")
plt.axis("equal")
plt.grid()
plt.show()
