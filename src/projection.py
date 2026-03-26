import numpy as np

def project_lidar_to_image(points_3d, P2, R0, Tr):
    """
    Project LiDAR point cloud into camera image space.
    
    Args:
        points_3d : (N, 3) LiDAR points in velodyne frame
        P2        : (3, 4) camera projection matrix
        R0        : (3, 3) rectification matrix
        Tr        : (3, 4) LiDAR to camera transform
    Returns:
        pixels    : (N, 2) pixel coordinates
        depth     : (N,)   depth in meters
        valid     : (N,)   boolean mask of valid points
    """
    N = points_3d.shape[0]
    pts_hom = np.hstack([points_3d, np.ones((N, 1))])
    Tr_full = np.vstack([Tr, [0, 0, 0, 1]])
    R0_full = np.eye(4)
    R0_full[:3, :3] = R0
    pts_cam = R0_full @ Tr_full @ pts_hom.T
    depth   = pts_cam[2, :]
    valid   = depth > 0
    pts_cam = pts_cam[:, valid]
    depth   = depth[valid]
    pts_img = P2 @ pts_cam
    pts_img = pts_img / pts_img[2, :]
    pixels  = pts_img[:2, :].T
    return pixels, depth, valid


def get_lidar_depth_for_box(box_2d, pixels, depth, pts_xyz, valid_mask):
    """
    Extract median LiDAR depth for all points inside a 2D bounding box.
    
    Args:
        box_2d    : [x1, y1, x2, y2]
        pixels    : (N, 2) projected pixel coords
        depth     : (N,)   depth values
        pts_xyz   : (M, 3) original LiDAR points
        valid_mask: (M,)   boolean mask from project_lidar_to_image
    Returns:
        distance  : float (meters) or None
        n_points  : int
    """
    x1, y1, x2, y2 = box_2d
    px, py  = pixels[:, 0], pixels[:, 1]
    in_box  = (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)
    depth_in_box = depth[in_box]
    if len(depth_in_box) == 0:
        return None, 0
    median = np.median(depth_in_box)
    clean  = depth_in_box[np.abs(depth_in_box - median) < 2.0]
    if len(clean) == 0:
        return None, 0
    return round(float(clean.mean()), 2), len(clean)
