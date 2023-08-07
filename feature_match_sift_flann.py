import cv2
import numpy as np

def get_camera_poses(frames,intrinsic_matrix,init_camera_pose):

    camera_poses = [init_camera_pose]  # The first frame is the origin (identity matrix)

    threshold = 0.7 # higher it gets it will eliminate weaker matches
    
    # SIFT Feature Detector and Descriptor
    sift = cv2.SIFT_create()

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    for i in range(1, len(frames)):
        prev_frame, curr_frame = frames[i-1], frames[i]

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Find keypoints 
        prev_keypoints, prev_descriptors = sift.detectAndCompute(prev_gray,None)

        # Find SIFT keypoints and descriptors in current frame
        curr_keypoints, curr_descriptors = sift.detectAndCompute(curr_gray,None)

        # Matching
        #bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = flann.knnMatch(prev_descriptors, curr_descriptors,k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)
        # Extract matching keypoints
        prev_matched_pts = np.float32([prev_keypoints[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        curr_matched_pts = np.float32([curr_keypoints[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

        # Calculate the Essential Matrix using intrinsic parameters
        E, _ = cv2.findEssentialMat(curr_matched_pts, prev_matched_pts, focal=intrinsic_matrix[0, 0],
                                    pp=(intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]), method=cv2.RANSAC,
                                    prob=0.999, threshold=1.0, mask=None)

        # Recover the camera poses from the Essential Matrix
        _, R, t, _ = cv2.recoverPose(E, curr_matched_pts, prev_matched_pts, focal=intrinsic_matrix[0, 0],
                                     pp=(intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]))

        # Compose the transformation matrix
        T = np.hstack((R, t))

        # Append the camera pose matrix for the current frame
        pose_matrix = np.vstack((T, [0, 0, 0, 1]))
        camera_poses.append(np.dot(camera_poses[-1], pose_matrix))

        # Visualization of feature matches (optional)
        vis_matches = cv2.drawMatches(prev_frame, prev_keypoints, curr_frame, curr_keypoints, good_matches, None)
        cv2.imshow('Feature Matches', vis_matches)
        cv2.waitKey(0)

    return camera_poses

# Example usage:
# Load your consecutive frames into the 'frames' list
frames = []

for i in range(11):
    image_path = f'image{i}.jpg'
    frame = cv2.imread(image_path)
    frames.append(frame)

K = np.array([[666.8,0,309.26],
     [0,671.15,247.67],
     [0,0,1]])

camera_pose_init = np.eye(4)
camera_poses = get_camera_poses(frames,K,camera_pose_init)

print(camera_poses)

# The 'camera_poses' list now contains the camera pose matrices for each frame.
