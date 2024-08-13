# Cython file: spt/core.pyx

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, INFINITY
cimport cython

from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

@cython.boundscheck(False)
@cython.wraparound(False)
cdef frame2frame(np.ndarray[double, ndim=2] locs0, np.ndarray[double, ndim=2] locs1, double max_dist):
    """
    Create LAP matrix based on the distances between particle positions in two frames.

    Parameters:
    -----------
    locs0 : 2D numpy array of shape (N0,2)
        Positions of spots in the frame t.
    locs1 : 2D numpy array of shape (N1,2)
        Positions of spots in the frame t+1.
    max_dist : float
        Maximum distance for linking.

    Returns:
    --------
    lap : 2D numpy array of shape (N0+N1, N0+N1)
        LAP matrix.
    """
    # Initialize LAP cost
    cdef int N0 = locs0.shape[0]
    cdef int N1 = locs1.shape[0]
    cdef int N = N0+N1
    cdef np.ndarray[double, ndim=2] lap = np.full((N, N), INFINITY)
    cdef double sqd = max_dist**2
    # cdef double d = sqd
    # cdef double b = sqd
    cdef int i, j

    for i in range(N0):
        lap[i, N1+i] = sqd # Fill "No linking Particle index" (top-right)
        for j in range(N1):
            lap[N0+j, j] = sqd  # Fill "No linking Particle index" (bottom-left)
            dist_sq = (locs0[i, 0] - locs1[j, 0]) ** 2 + (locs0[i, 1] - locs1[j, 1]) ** 2
            if dist_sq < sqd:
                lap[i, j] = dist_sq # Fill "Linking Particle index" (top-left)
            lap[N0+j, N1+i] = lap[i, j] # Fill "LAP topological constrainst" (bottom-right)
                
    return lap

@cython.boundscheck(False)
@cython.wraparound(False)
cdef segment2segment(np.ndarray[double, ndim=2] dfspots, int skip_frames, double max_dist, int n_segments):
    """
    Populate LAP matrix for segment linking.

    Parameters:
    -----------
    dfspots : 2D numpy array of shape (N,6)
            Positions of spots with track id, indexes and sorted by frame as:
                -------------------------------
                idx | description
                -------------------------------
                0  | frame
                1  | frame subindex (fidx)
                2  | x coordinates (x)
                3  | y coordinates (y)
                4  | track id (tid)
                5  | dataframe indexes (idx)
                -------------------------------
    max_dist : float
        Maximum distance for linking.

    Returns:
    --------
    lap : 2D numpy array of shape (2*n_segments, 2*n_segments)
        LAP matrix.
    """
    cdef int n_combos
    cdef np.ndarray[double, ndim=2] dfsegments = np.zeros((n_segments, 6))
    cdef np.ndarray[double, ndim=2] lap = np.full((2*n_segments, 2*n_segments), INFINITY)
    cdef list segment_combos = []
    cdef int idxseg 
    cdef int i, j

    for i in range(n_segments):
        # Find i-th segment
        seg = dfspots[dfspots[:, 4] == i]
        idxseg = seg.shape[0] - 1
        # Populate dfsegments
        dfsegments[i, 0] = seg[0, 0] # frame_start
        dfsegments[i, 1] = seg[0, 2] # x_start
        dfsegments[i, 2] = seg[0, 3] # y_start

        dfsegments[i, 3] = seg[idxseg, 0] # frame_end
        dfsegments[i, 4] = seg[idxseg, 2] # x_end
        dfsegments[i, 5] = seg[idxseg, 3] # y_end
    
    # Identify segment pairs (i, j) that can be linked
    for i in range(n_segments):
        for j in range(n_segments):
            if i != j:
                d_frame = dfsegments[j, 0] - dfsegments[i, 3]
                if 0 < d_frame <= skip_frames + 1:
                    segment_combos.append((i, j))

    # Populate LAP matrix with squared distances
    n_combos = len(segment_combos)
    for i in range(n_combos):
        seg_i = dfsegments[segment_combos[i][0]]
        seg_j = dfsegments[segment_combos[i][1]]
        
        d_frame = seg_j[0] - seg_i[3]
        if 0 < d_frame <= skip_frames + 1:
            d_x = seg_j[1] - seg_i[4]
            d_y = seg_j[2] - seg_i[5]
            distance = d_x * d_x + d_y * d_y
            if distance <= max_dist * max_dist:
                lap[segment_combos[i][0], segment_combos[i][1]] = distance

    # Fill the no-link regions using a loop
    for i in range(n_segments):
        for j in range(n_segments):
            if i == j:
                lap[i, n_segments + j] = max_dist * max_dist
                lap[n_segments + i, j] = max_dist * max_dist
            lap[n_segments + i, n_segments + j] = lap[j, i]

    return lap

@cython.boundscheck(False)
@cython.wraparound(False)
cdef prepare(np.ndarray[double, ndim=2] spots, int N):
    """
    Prepare spots array for tracking algorithm.

    Parameters:
    -----------
    spots : 2D numpy array of shape (N,4)
        Positions of spots where:
            -------------------------------
            idx | description
            -------------------------------
             0  | frame
             1  | frame subindex (fidx)
             2  | x coordinates (x)
             3  | y coordinates (y)
            -------------------------------
    Returns:
    --------
    dfspots : 2D numpy array of shape (N,6)
        Positions of spots with track id, indexes and sorted by frame as:
            -------------------------------
            idx | description
            -------------------------------
             0  | frame
             1  | frame subindex (fidx)
             2  | x coordinates (x)
             3  | y coordinates (y)
             4  | track id (tid)
             5  | dataframe indexes (idx)
            -------------------------------
    """
    cdef np.ndarray[double, ndim=2] dfspots = np.zeros((N, 6))
    cdef int i

    for i in range(N): 
        dfspots[i, :4] = spots[i, :] # Copy spots info
        dfspots[i, 4] = -1 # Overwrite track id
        dfspots[i, 5] = i # Create new column with indexes
    dfspots = dfspots[dfspots[:, 0].argsort()] # Sort by frame

    return dfspots

@cython.boundscheck(False)
@cython.wraparound(False)
def run(np.ndarray[double, ndim=2] spots, int skip_frames, double max_dist):
    """
    Robust single-particle tracking.

    Parameters:
    -----------
    spots : 2D numpy array of shape (N, 4)
        Positions of spots where:
            -------------------------------
            idx | description
            -------------------------------
             0  | frame
             1  | frame subindex (fidx)
             2  | x coordinates (x)
             3  | y coordinates (y)
            -------------------------------
    skip_frame : int
        Maximum number of frames to skip for segment linking.
    max_dist : float
        Maximum distance for linking.

    Returns:
    --------
    spots : 2D numpy array of shape (N,5)
        Positions of spots with track id where:
            -------------------------------
            idx | description
            -------------------------------
             0  | frame
             1  | frame subindex (fidx)
             2  | x coordinates (x)
             3  | y coordinates (y)
             4  | track id (tid)
            -------------------------------
    """
    cdef int N = spots.shape[0]
    cdef int N0, N1
    cdef int n_frames = np.max(spots[:, 0])
    cdef int i, j, tid
    cdef int idx, idx0, idx1
    cdef int frame0, frame1
    cdef np.ndarray[double, ndim=2] costs
    cdef np.ndarray[double, ndim=2] blobs0, blobs1
    cdef np.ndarray[double, ndim=2] locs0, locs1
    cdef np.ndarray[long, ndim=1] row_idxs, col_idxs, idxs
    cdef np.ndarray[double, ndim=1] b0_idxs, b1_idxs

    # Prepare        
    dfspots = prepare(spots, N)

    # Step 1: frame-to-frame linking
    for frame1 in tqdm(range(n_frames+1), desc="Step 1: frame-to-frame linking"):
        frame0 = frame1 - 1

        # Current & future spots
        blobs0 = dfspots[dfspots[:, 0] == frame0]
        N0 = blobs0.shape[0]

        blobs1 = dfspots[dfspots[:, 0] == frame1]
        N1 = blobs1.shape[0]

        if (N0 == 0) & (N1 > 0):
            # Create new blobs1 segment
            for i in range(N1):
                idx = np.where(dfspots[:, 5]==blobs1[i, 5])[0]
                dfspots[idx, 4] = np.max(dfspots[:, 4]) + 1

        # Compute and solve LAP matrix; connect blobs
        if (N0 > 0) & (N1 > 0):
            locs0 = blobs0[:, 2:4]        
            locs1 = blobs1[:, 2:4]
            costs = frame2frame(locs0, locs1, max_dist).T
            row_idxs, col_idxs = linear_sum_assignment(costs)
            b0_idxs = blobs0[blobs0[:, 1].argsort()][:, 5]
            b1_idxs = blobs1[blobs1[:, 1].argsort()][:, 5]
            for i, j in enumerate(col_idxs[:N1]):
                idx1 = np.where(dfspots[:, 5]==b1_idxs[i])[0]
                if j < N0:
                    # Connect
                    idx0 = np.where(dfspots[:, 5]==b0_idxs[j])[0]
                    dfspots[idx1, 4] = dfspots[idx0, 4]
                else:
                    # Create
                    dfspots[idx1, 4] = np.max(dfspots[:, 4]) + 1
    
    # Step 2: link segments.
    cdef int n_segments = np.max(dfspots[:, 4]) + 1
    pbar = tqdm(total=n_segments, desc="Step 2: link segments")
    costs = segment2segment(dfspots, skip_frames, max_dist, n_segments)
    row_idxs, col_idxs = linear_sum_assignment(costs)

    # Update track ids based on LAP result
    for i in range(len(row_idxs)):
        if row_idxs[i] < n_segments and col_idxs[i] < n_segments:
            idxs = dfspots[dfspots[:, 4] == col_idxs[i]][:, 5].astype(int)
            tid = dfspots[dfspots[:, 4] == row_idxs[i]][0, 4]
            for j in idxs:
                idx = np.where(dfspots[:, 5]==j)[0]
                dfspots[idx, 4] = tid
        pbar.update(2)

    return dfspots




