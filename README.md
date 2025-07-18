# Visual Odometry for Autonomous Navigation

## Description
This project investigates the use of **visual odometry** for robotic navigation, developed as part of COSC428 at the University of Canterbury. The aim is to estimate camera motion using RGB image sequences and depth information, producing odometry data suitable for robotic systems and integration with frameworks such as ROS.

By detecting and matching visual features an estimated poses are aligned and evaluated against ground truth, providing insight into trajectory accuracy and system performance.

## Features
- Monocular VO using RGB-D data
- ORB feature detection and Hamming-based brute-force matching
- Essential matrix recovery via RANSAC and pose estimation
- Depth-based scale recovery
- Trajectory alignment and Absolute Trajectory Error (ATE) evaluation
- Real-time visualization of matches and trajectory


### Requirements
- Python 3.8+
- OpenCV (cv2)
- NumPy
- SciPy
- Matplotlib

### Setup
1. Clone the repository:
   ```bash
   git clone https://eng-git.canterbury.ac.nz/msd60/cosc428-project.git

2. Download the tgz files from following link
"Computer  Vision  Group  -  Dataset Download,” cvg.cit.tum.de. 
    ```bash 
    - Category: Handheld SLAM
    file -> fr1/floor
    file -> fr1/room
    https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download 

2. Extract datasets into a folder named Data
    ```bash
    tar -xvzf "rgbd_dataset_freiburg1_floor.tgz" -C "Data/"
    tar -xvzf "rgbd_dataset_freiburg1_room.tgz" -C "Data/"

3. Run VOdom_depth. This computes odometry and outputs to 'vo_output.txt'
    ```bash
    python VOdom_depth.py

4. Run Results. This compares results to ground truth and quantifies error.
    ```bash
    python Results.py

### Further Improvements 
Please refer to [Report](<Visual Odometry For Autonomous Navigation.pdf>) for further improvements :)