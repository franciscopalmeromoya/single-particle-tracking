# Cython file: spt/core.pyx

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, INFINITY

def frame2frame(np.ndarray[double, ndim=2] locs0, np.ndarray[double, ndim=2] locs1, double max_dist):
    """
    Create LAP matrix based on the distances between particle positions in two frames.

    Parameters:
    locs0 : 2D numpy array of shape (N,2)
        Positions of spots in the frame t.
    locs1 : 2D numpy array of shape (N,2)
        Positions of spots in the frame t+1.
    max_dist : float
        Maximum distance for linking.

    Returns:
    lap : 2D numpy array of shape (2*N, 2*N)
        LAP matrix.
    """

    cdef int N0 = locs0.shape[0]
    cdef int N1 = locs1.shape[0]
    cdef int N = N0+N1
    cdef np.ndarray[double, ndim=2] lap = np.full((N, N))

    return lap

