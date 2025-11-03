# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from cython.parallel import prange

# For numpy arrays - FASTEST
cdef void zero_and_fill_canvas_c(unsigned char[:, :, :] img, 
                                  int[:] x, 
                                  int[:] y, 
                                  int[:] p) noexcept nogil:
    cdef int length = x.shape[0]
    cdef int xi, yi, pi
    cdef int j
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]
    
    # Zero the array
    for j in range(h * w * img.shape[2]):
        (<unsigned char*>&img[0, 0, 0])[j] = 0
    
    # Fill canvas
    for j in range(length):
        xi = x[j]
        yi = y[j]
        pi = p[j]
        img[yi, xi, pi] = 1


def zero_and_fill_canvas(np.ndarray[np.uint8_t, ndim=3] img, 
                         np.ndarray[np.int32_t, ndim=1] x, 
                         np.ndarray[np.int32_t, ndim=1] y, 
                         np.ndarray[np.int32_t, ndim=1] p) -> None:
    zero_and_fill_canvas_c(img, x, y, p)


# For lists - convert once then use fast path
def zero_and_fill_canvas_list(np.ndarray[np.uint8_t, ndim=3] img, 
                              list x, 
                              list y, 
                              list p) -> None:
    cdef np.ndarray[np.int32_t, ndim=1] x_arr = np.array(x, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] y_arr = np.array(y, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] p_arr = np.array(p, dtype=np.int32)
    zero_and_fill_canvas_c(img, x_arr, y_arr, p_arr)


# Alternative: direct list iteration (slower but no conversion)
def zero_and_fill_canvas_list_direct(np.ndarray[np.uint8_t, ndim=3] img, 
                                     list x, 
                                     list y, 
                                     list p) -> None:
    cdef int length = len(x)
    cdef int xi, yi, pi
    cdef int j
    
    img.fill(0)
    
    for j in range(length):
        xi = <int>x[j]
        yi = <int>y[j]
        pi = <int>p[j]
        img[yi, xi, pi] = 1
