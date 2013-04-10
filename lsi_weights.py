from __future__ import division
import numpy as np

#-------------------------
# WEIGHTS
#-------------------------

# matrix must be a numpy 2D array
def local_log_weighting( matrix ):
    A = matrix
    (rows,_) = A.shape
    for i in range(rows):
        A[i] = np.log2(A[i]+1)
    return A

# matrix must be a numpy 2D array
def local_aug_weighting( matrix ):
    A = matrix
    (rows,_) = A.shape
    y = 0
    # this is array wise math
    for i in range(rows):
        m = A[i].max(axis=0)
        A[i] = ((A[i]/m)+1) / 2
    return A

# matrix must be a numpy 2D array
def local_binary_weighting( matrix ):
    A = matrix
    (rows,columns) = A.shape
    y = 0
    for i in range(rows):
        for j in range(columns):
            if A[i][j] != 0:
                A[i][j] = 1                
    return A

def global_normal_weighting( matrix ):
    # for each word, square the values then sum them
    # return the inverse of that 
    (x,_) = matrix.shape
    #print matrix.shape
    g = np.zeros(x)

    for i in range(x):
        g[i] = 1/np.sqrt(np.sum((matrix[i])**2))

    return g

def global_gfldf_weighting( matrix ):
    # for each word, square the values then sum them
    # return the inverse of that 
    (x,_) = matrix.shape
    g = np.zeros(x)

    for i in range(x):
        gf = np.sum(matrix[i])
        df = np.count_nonzero(matrix[i])
        g[i] = gf/df
    return g

def global_ldf_weighting( matrix ):
    # for each word, square the values then sum them
    # return the inverse of that 
    (x,_) = matrix.shape
    g = np.zeros(x)

    for i in range(x):
        df = np.count_nonzero(matrix[i])
        g[i] = np.log2(( x /( 1 + df )))                       
    return g

def global_entropy_weighting( matrix ):
    # for each word, square the values then sum themi]
    # return the inverse of that 
    (x,_) = matrix.shape
    g = np.zeros(x)
    
    for i in range(x):
        temp = matrix[i]               
        gf = np.sum(matrix[i])
        temp = temp/gf
        a = temp
        for j in range(len(temp)):
            if temp[j] == 0:
                a[j] = 0
            else:
                a[j] = np.log2( temp[j] )
        temp = temp * a        
        temp = temp / np.log2( x )
        g[i] = 1 + np.sum( temp )    
    return g

def best_weighting( matrix ):
    g = global_entropy_weighting( matrix )
    A = matrix
    (rows,_) = A.shape
    for i in range(rows):
        A[i] = g[i] * np.log2(A[i]+1)
    return A

def weight_matrix(matrix, local_weight, global_weight):
    A = matrix
    B = matrix
    
    if local_weight == 'log' or local_weight == 1:
        A = local_log_weighting( A )
    elif local_weight =='aug' or local_weight == 2:
        A = local_aug_weighting( A )
    elif local_weight =='binary' or local_weight == 3:
        A = local_binary_weighting( A ) 
    else:
        A = A

    if global_weight == 'norm' or global_weight == 1:
        g = global_normal_weighting( B ) 
    elif global_weight == 'gfldf' or global_weight == 2:
        g = global_gfldf_weighting( B ) 
    elif global_weight == 'ldf' or global_weight == 3:
        g = global_ldf_weighting( B )
    elif global_weight == 'entropy' or global_weight == 4:
        g = global_entropy_weighting( B )
    else:
        g = np.array(1)
  
    return np.multiply(A.T, g).T
