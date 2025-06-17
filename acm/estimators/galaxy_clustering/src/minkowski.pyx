#cython: language_level=3
import numpy as np 
import time,sys,os
cimport numpy as np
cimport cython
from cython.parallel import prange
cimport openmp

@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def Minkowski_slice(np.ndarray[np.float32_t, ndim=3] delta_slices, np.ndarray[np.float32_t, ndim=1] thresholds, np.float32_t thres_mask):
    cdef int i,j,k,len_thres,dims_y, dims_z
    cdef long long vol_slice
    cdef np.float32_t t
    cdef np.ndarray[np.float64_t, ndim=2] MFs
    cdef np.ndarray[np.int32_t, ndim=1] n
    cdef np.ndarray[np.float32_t, ndim=3] ds
    cdef np.ndarray[np.float32_t, ndim=2] ds_min
    cdef np.ndarray[np.float32_t, ndim=2] ds_max
    
    vol_slice = 0
    dims_y    = delta_slices.shape[1]
    dims_z    = delta_slices.shape[2]
    len_thres = len(thresholds)
    MFs       = np.zeros((len_thres,4),dtype=np.float64)
    n         = np.zeros(4,dtype=np.int32)
    ds        = np.zeros((8,dims_y,dims_z),dtype=np.float32)
    ds_min    = np.zeros((dims_y,dims_z),dtype=np.float32)
    ds_max    = np.zeros((dims_y,dims_z),dtype=np.float32)
    
    ds[0]  = delta_slices[0]
    ds[1]  = delta_slices[1]
    ds[2]  = np.roll(ds[0],-1,axis=0)
    ds[3]  = np.roll(ds[1],-1,axis=0)
    ds[4]  = np.roll(ds[0],-1,axis=1)
    ds[5]  = np.roll(ds[1],-1,axis=1)
    ds[6]  = np.roll(ds[2],-1,axis=1)
    ds[7]  = np.roll(ds[3],-1,axis=1)
    ds_min = np.min(ds,axis=0)
    ds_max = np.max(ds,axis=0)         

    for i in range(dims_y):
        for j in range(dims_z):
            if ds_min[i,j]>thres_mask:
                vol_slice += 1
                for k in range(len_thres):
                    t = thresholds[k]
                    if t<ds_min[i,j]:
                        MFs[k,0] += 1
                    elif t<ds_max[i,j]:   
                        n[3] = ds[0,i,j]>t
                        n[2] = ((ds[0,i,j]>t or ds[1,i,j]>t)+
                                (ds[0,i,j]>t or ds[2,i,j]>t)+
                                (ds[0,i,j]>t or ds[4,i,j]>t))
                        n[1] = ((ds[0,i,j]>t or ds[1,i,j]>t or ds[2,i,j]>t or ds[3,i,j]>t)+
                                (ds[0,i,j]>t or ds[2,i,j]>t or ds[4,i,j]>t or ds[6,i,j]>t)+
                                (ds[0,i,j]>t or ds[4,i,j]>t or ds[1,i,j]>t or ds[5,i,j]>t))
                        n[0] = (ds[0,i,j]>t or ds[1,i,j]>t or ds[2,i,j]>t or ds[3,i,j]>t or
                                ds[4,i,j]>t or ds[5,i,j]>t or ds[6,i,j]>t or ds[7,i,j]>t)
                        MFs[k,0] +=     n[3]
                        MFs[k,1] += (-3*n[3] + n[2]) *2/9
                        MFs[k,2] += ( 3*n[3]-2*n[2] + n[1]) *2/9
                        MFs[k,3] += ( - n[3] + n[2] - n[1] + n[0])
        
    return (MFs,vol_slice)

################################################################################
# This routine computes the Minkowski functionals of a density field
# delta -------> 3D density field: (dims,dims,dims) numpy array
# CellSize -----> Cell size of the density field
# thres_mask -----> Value of field outside the geometry of galaxy survey should be lower than thres_mask
# thresholds --------> density threshold above which the excursion set is defined

class MFs:
    def __init__(self,delta,CellSize,thres_mask,thresholds):
        start = time.time()
        cdef int dims_x, dims_y, dims_z
        cdef int i,len_thres
        cdef long long vol, vol_slice
        cdef double a
        cdef np.ndarray[np.float64_t, ndim=2] MFs3D 
        cdef np.ndarray[np.float64_t, ndim=2] MFs3D_slice  

        print('\nComputing Minkowski functionals of the field...')
        dims_x,dims_y,dims_z = delta.shape      
        len_thres   = len(thresholds)
        vol         = 0
        vol_slice   = 0
        delta       = np.concatenate((delta,delta[0:1,:,:]),axis=0)
        MFs3D       = np.zeros((len_thres,4), dtype=np.float64)
   
        #calculate the MFs of slices and add up 
        for i in range(dims_x):
            MFs3D_slice,vol_slice = Minkowski_slice(delta[i:i+2,:,:].astype(np.float32),thresholds,thres_mask)
            MFs3D += MFs3D_slice
            vol   += vol_slice
        
        l = 1.0*vol
        a = CellSize
        #Normalize the MFs
        MFs3D *= np.array([1/l,1/(l*a),1/(l*a*a),1/(l*a*a*a)])
        self.MFs3D = MFs3D
        print('Time taken = %.2f seconds'%(time.time()-start))
################################################################################



@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def Minkowski_cube(np.ndarray[np.float32_t, ndim=2] ds,
                   np.ndarray[np.float32_t, ndim=1] ds_min,
                   np.ndarray[np.float32_t, ndim=1] ds_max, 
                   np.float32_t t, np.float32_t thres_mask, np.int32_t nthreads):
    
    cdef int i,dims
    cdef np.ndarray[np.float64_t, ndim=2] MFs3D
    cdef np.ndarray[np.int32_t, ndim=2] n
    dims      = ds_min.shape[0]
    MFs3D     = np.zeros((4,dims),dtype=np.float64)
    n         = np.zeros((4,dims),dtype=np.int32)
    if nthreads > 0:
        openmp.omp_set_num_threads(nthreads)
    for i in prange(dims, nogil=True, schedule='static', num_threads=nthreads):
        if ds_min[i]>thres_mask:
            if t<ds_min[i]:
                MFs3D[0,i] = 1
            elif t<ds_max[i]:   
                n[3,i] = ds[0,i]>t
                n[2,i] = ((ds[0,i]>t or ds[1,i]>t)+
                          (ds[0,i]>t or ds[2,i]>t)+
                          (ds[0,i]>t or ds[4,i]>t))
                n[1,i] = ((ds[0,i]>t or ds[1,i]>t or ds[2,i]>t or ds[3,i]>t)+
                          (ds[0,i]>t or ds[2,i]>t or ds[4,i]>t or ds[6,i]>t)+
                          (ds[0,i]>t or ds[4,i]>t or ds[1,i]>t or ds[5,i]>t))
                n[0,i] = (ds[0,i]>t or ds[1,i]>t or ds[2,i]>t or ds[3,i]>t or
                          ds[4,i]>t or ds[5,i]>t or ds[6,i]>t or ds[7,i]>t)
                MFs3D[0,i] =     n[3,i]
                MFs3D[1,i] = (-3*n[3,i] + n[2,i]) *2/9
                MFs3D[2,i] = ( 3*n[3,i]-2*n[2,i] + n[1,i]) *2/9
                MFs3D[3,i] = ( - n[3,i] + n[2,i] - n[1,i] + n[0,i])
        
    return MFs3D

################################################################################
# This routine computes the Minkowski functionals of a density field with multiple CPUs
# delta -------> 3D density field: (dims,dims,dims) numpy array
# CellSize -----> Cell size of the density field
# thres_mask -----> Value of field outside the geometry of galaxy survey should be lower than thres_mask
# thresholds --------> density threshold above which the excursion set is defined

class MFs_parallel:
    def __init__(self,delta,CellSize,thres_mask,thresholds,nthreads):
        start = time.time()
        cdef int dims_x, dims_y, dims_z
        cdef int i,j,len_thres
        cdef long long vol, vol_slice
        cdef double a
        cdef float thres
        cdef np.ndarray[np.float64_t, ndim=2] MFs
        cdef np.ndarray[np.float64_t, ndim=2] MFs3D  
        # cdef np.ndarray[np.float32_t, ndim=4] ds
        # cdef np.ndarray[np.float32_t, ndim=3] ds_min
        # cdef np.ndarray[np.float32_t, ndim=3] ds_max

        print('\nComputing Minkowski functionals of the field with '+str(nthreads)+' CPUs.')
        dims_x,dims_y,dims_z = delta.shape      
        len_thres   = len(thresholds)
        vol         = 0
        vol_slice   = 0
        delta       = np.array(delta)
        ds          = np.zeros((8,dims_x,dims_y,dims_z),dtype=np.float32)
        ds_min      = np.zeros((dims_x,dims_y,dims_z),dtype=np.float32)
        ds_max      = np.zeros((dims_x,dims_y,dims_z),dtype=np.float32)
        MFs3D       = np.zeros((4,dims_x*dims_y*dims_z), dtype=np.float64)
        MFs         = np.zeros((len_thres,4), dtype=np.float64)
        ds_new = np.reshape(ds,(8,dims_x*dims_y*dims_z))

        ds[0]  = delta
        ds[1]  = np.concatenate((delta[1:,:,:],delta[0:1,:,:]),axis=0)
        ds[2]  = np.roll(ds[0],-1,axis=1)
        ds[3]  = np.roll(ds[1],-1,axis=1)
        ds[4]  = np.roll(ds[0],-1,axis=2)
        ds[5]  = np.roll(ds[1],-1,axis=2)
        ds[6]  = np.roll(ds[2],-1,axis=2)
        ds[7]  = np.roll(ds[3],-1,axis=2)
        ds_min = np.min(ds,axis=0)
        ds_max = np.max(ds,axis=0) 
        
        ds     = np.reshape(ds,(8,dims_x*dims_y*dims_z))
        ds_min = np.reshape(ds_min,(dims_x*dims_y*dims_z,))
        ds_max = np.reshape(ds_max,(dims_x*dims_y*dims_z,))
        if nthreads > 0:
            openmp.omp_set_num_threads(nthreads)
   
        #calculate the MFs of slices and add up 
        for i in range(len_thres):
            thres = thresholds[i]
            MFs3D = Minkowski_cube(ds,ds_min,ds_max,thres,thres_mask,nthreads)
            for j in range(4):
                MFs[i,j] = np.sum(MFs3D[j])
                
        vol = np.sum(ds_min>thres_mask)
        l = 1.0*vol
        a = CellSize
        #Normalize the MFs
        MFs *= np.array([1/l,1/(l*a),1/(l*a*a),1/(l*a*a*a)])
        self.MFs3D = MFs
        print('Time taken = %.2f seconds'%(time.time()-start))
################################################################################
