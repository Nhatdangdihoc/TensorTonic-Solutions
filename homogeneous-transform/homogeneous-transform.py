import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).

    Args:
        T: (4,4) array — homogeneous transformation matrix
        points: (3,) for a single point, or (N,3) for N points

    Returns:
        Transformed points, same shape as input (3,) or (N,3)
    """
    T = np.asarray(T, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)

    single_point = (points.ndim == 1)
    if single_point:
        points = points.reshape(1, -1)   # (3,) -> (1,3)

    N = points.shape[0]

    # Step 1: convert to homogeneous coords -> (N,4), append column of 1s
    ones = np.ones((N, 1))
    points_h = np.hstack([points, ones])       # (N,4)

    # Step 2: apply transform -> T @ p for each point
    # points_h.T is (4,N), T @ points_h.T -> (4,N)
    transformed_h = (T @ points_h.T).T          # (N,4)

    # Step 3: drop homogeneous coordinate (divide by w, usually w=1)
    w = transformed_h[:, 3:4]
    transformed = transformed_h[:, :3] / w      # (N,3)

    if single_point:
        return transformed[0]                   # back to (3,)
    return transformed


# --- Example usage ---
if __name__ == "__main__":
    # Transform: rotate 90 deg around Z axis, then translate by (1,2,3)
    theta = np.pi / 2
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    t = np.array([1, 2, 3])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    # single point
    p = np.array([1, 0, 0])
    print(apply_homogeneous_transform(T, p))
    # (1,0,0) xoay 90 quanh Z -> (0,1,0), rồi cộng (1,2,3) -> [1,3,3]

    # multiple points
    pts = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    print(apply_homogeneous_transform(T, pts))