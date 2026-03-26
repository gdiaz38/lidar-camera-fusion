import numpy as np

def parse_calib(calib_path):
    """Parse KITTI calibration file into projection matrices."""
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            if ':' in line:
                key, value = line.split(':', 1)
                calib[key.strip()] = np.array(
                    [float(x) for x in value.strip().split()]
                )
    P2 = calib['P2'].reshape(3, 4)
    R0 = calib['R0_rect'].reshape(3, 3)
    Tr = calib['Tr_velo_to_cam'].reshape(3, 4)
    return P2, R0, Tr
